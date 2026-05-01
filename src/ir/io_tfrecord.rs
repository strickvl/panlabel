//! TensorFlow TFRecord reader and writer for TFOD-style object detection examples.
//!
//! TFRecord itself is only a binary record container. This adapter supports the
//! common TensorFlow Object Detection API convention where each record payload is
//! a serialized `tf.train.Example` containing image metadata plus normalized
//! bounding-box lists under `image/object/bbox/*`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use prost::Message;

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

pub const ATTR_SOURCE_ID: &str = "tfrecord_source_id";
pub const ATTR_KEY_SHA256: &str = "tfrecord_key_sha256";
pub const ATTR_FORMAT: &str = "tfrecord_format";
pub const ATTR_HAD_ENCODED_IMAGE: &str = "tfrecord_had_encoded_image";
pub const ATTR_CLASS_LABEL: &str = "tfrecord_class_label";
pub const ATTR_AREA: &str = "area";
pub const ATTR_IS_CROWD: &str = "iscrowd";
pub const ATTR_DIFFICULT: &str = "difficult";
pub const ATTR_GROUP_OF: &str = "tfrecord_group_of";
pub const ATTR_WEIGHT: &str = "tfrecord_weight";
pub const ATTR_UNSUPPORTED_FEATURE_COUNT: &str = "tfrecord_unsupported_feature_count";
pub const ATTR_UNSUPPORTED_FEATURE_KEYS: &str = "tfrecord_unsupported_feature_keys";

const BYTES_HELPER_PATH: &str = "<tfrecord bytes>";
const MAX_SNIFF_BYTES: usize = 64 * 1024 * 1024;

const IMAGE_FILENAME: &str = "image/filename";
const IMAGE_HEIGHT: &str = "image/height";
const IMAGE_WIDTH: &str = "image/width";
const IMAGE_SOURCE_ID: &str = "image/source_id";
const IMAGE_KEY_SHA256: &str = "image/key/sha256";
const IMAGE_ENCODED: &str = "image/encoded";
const IMAGE_FORMAT: &str = "image/format";
const BBOX_XMIN: &str = "image/object/bbox/xmin";
const BBOX_XMAX: &str = "image/object/bbox/xmax";
const BBOX_YMIN: &str = "image/object/bbox/ymin";
const BBOX_YMAX: &str = "image/object/bbox/ymax";
const CLASS_TEXT: &str = "image/object/class/text";
const CLASS_LABEL: &str = "image/object/class/label";
const AREA: &str = "image/object/area";
const IS_CROWD: &str = "image/object/is_crowd";
const DIFFICULT: &str = "image/object/difficult";
const GROUP_OF: &str = "image/object/group_of";
const WEIGHT: &str = "image/object/weight";

#[derive(Clone, PartialEq, Message)]
struct Example {
    #[prost(message, optional, tag = "1")]
    features: Option<Features>,
}

#[derive(Clone, PartialEq, Message)]
struct Features {
    #[prost(message, repeated, tag = "1")]
    feature: Vec<FeatureEntry>,
}

#[derive(Clone, PartialEq, Message)]
struct FeatureEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(message, optional, tag = "2")]
    value: Option<Feature>,
}

#[derive(Clone, PartialEq, Message)]
struct Feature {
    #[prost(oneof = "feature::Kind", tags = "1, 2, 3")]
    kind: Option<feature::Kind>,
}

mod feature {
    use super::{BytesList, FloatList, Int64List};
    use prost::Oneof;

    // Variant names mirror the canonical `tf.train.Feature` proto schema
    // (BytesList / FloatList / Int64List); renaming would diverge from
    // feature.proto, which is the schema readers will cross-reference.
    #[allow(clippy::enum_variant_names)]
    #[derive(Clone, PartialEq, Oneof)]
    pub enum Kind {
        #[prost(message, tag = "1")]
        BytesList(BytesList),
        #[prost(message, tag = "2")]
        FloatList(FloatList),
        #[prost(message, tag = "3")]
        Int64List(Int64List),
    }
}

#[derive(Clone, PartialEq, Message)]
struct BytesList {
    #[prost(bytes = "vec", repeated, tag = "1")]
    value: Vec<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct FloatList {
    #[prost(float, repeated, tag = "1")]
    value: Vec<f32>,
}

#[derive(Clone, PartialEq, Message)]
struct Int64List {
    #[prost(int64, repeated, tag = "1")]
    value: Vec<i64>,
}

#[derive(Debug)]
struct ParsedRecord {
    record_index: usize,
    file_name: String,
    width: u32,
    height: u32,
    image_attributes: BTreeMap<String, String>,
    objects: Vec<ParsedObject>,
    unsupported_feature_keys: Vec<String>,
}

#[derive(Debug)]
struct ParsedObject {
    class_name: String,
    bbox_norm: BBoxXYXY<Normalized>,
    attributes: BTreeMap<String, String>,
}

/// Read a single-file, uncompressed TFRecord dataset.
pub fn read_tfrecord(path: &Path) -> Result<Dataset, PanlabelError> {
    let bytes = fs::read(path).map_err(|source| PanlabelError::TfrecordRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    from_tfrecord_slice_with_path(&bytes, path)
}

/// Write a dataset as single-file, uncompressed TFOD-style TFRecords.
pub fn write_tfrecord(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let bytes = to_tfrecord_vec_with_path(dataset, path)?;
    fs::write(path, bytes).map_err(|source| PanlabelError::TfrecordWrite {
        path: path.to_path_buf(),
        message: source.to_string(),
    })
}

/// Parse TFRecord bytes. Useful for tests and fuzzing.
pub fn from_tfrecord_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    from_tfrecord_slice_with_path(bytes, Path::new(BYTES_HELPER_PATH))
}

/// Serialize a dataset to TFRecord bytes. Useful for tests and fixtures.
pub fn to_tfrecord_vec(dataset: &Dataset) -> Result<Vec<u8>, PanlabelError> {
    to_tfrecord_vec_with_path(dataset, Path::new(BYTES_HELPER_PATH))
}

/// Return true if a file starts with a valid TFRecord frame whose payload is a
/// supported TFOD-style `tf.train.Example`.
pub fn is_supported_tfrecord_file(path: &Path) -> Result<bool, PanlabelError> {
    let metadata = fs::metadata(path).map_err(|source| PanlabelError::TfrecordRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    if metadata.len() > MAX_SNIFF_BYTES as u64 {
        // Avoid reading unbounded data during detection. The caller can still
        // specify --from tfrecord explicitly for very large files.
        return Ok(false);
    }

    let bytes = fs::read(path).map_err(|source| PanlabelError::TfrecordRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    is_supported_tfrecord_slice(&bytes, path)
}

fn is_supported_tfrecord_slice(bytes: &[u8], path: &Path) -> Result<bool, PanlabelError> {
    let Some((payload, _next)) = read_one_frame(bytes, 0, path)? else {
        return Ok(false);
    };
    let Ok(example) = Example::decode(payload) else {
        return Ok(false);
    };
    let features = features_to_map(example.features.as_ref());
    Ok(has_tfod_evidence(&features))
}

fn from_tfrecord_slice_with_path(bytes: &[u8], path: &Path) -> Result<Dataset, PanlabelError> {
    let mut offset = 0usize;
    let mut record_index = 0usize;
    let mut parsed = Vec::new();

    while offset < bytes.len() {
        let Some((payload, next_offset)) = read_one_frame(bytes, offset, path)? else {
            break;
        };
        let example =
            Example::decode(payload).map_err(|source| PanlabelError::TfrecordProtobufDecode {
                path: path.to_path_buf(),
                record_index,
                source,
            })?;
        parsed.push(parse_example(example, path, record_index)?);
        offset = next_offset;
        record_index += 1;
    }

    if record_index == 0 {
        return Err(invalid(path, "TFRecord file contains no records"));
    }

    records_to_dataset(parsed, path)
}

fn to_tfrecord_vec_with_path(dataset: &Dataset, path: &Path) -> Result<Vec<u8>, PanlabelError> {
    let mut output = Vec::new();
    for example in dataset_to_examples(dataset, path)? {
        let payload = example.encode_to_vec();
        write_frame(&mut output, &payload);
    }
    Ok(output)
}

fn read_one_frame<'a>(
    bytes: &'a [u8],
    offset: usize,
    path: &Path,
) -> Result<Option<(&'a [u8], usize)>, PanlabelError> {
    if offset == bytes.len() {
        return Ok(None);
    }
    let header_end = offset
        .checked_add(12)
        .ok_or_else(|| invalid(path, "frame offset overflow"))?;
    if header_end > bytes.len() {
        return Err(invalid(path, "truncated TFRecord frame header"));
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[offset..offset + 8]);
    let length = u64::from_le_bytes(len_bytes);
    let length_usize = usize::try_from(length).map_err(|_| {
        invalid(
            path,
            format!("record length {length} does not fit in memory"),
        )
    })?;

    let mut stored_len_crc_bytes = [0u8; 4];
    stored_len_crc_bytes.copy_from_slice(&bytes[offset + 8..offset + 12]);
    let stored_len_crc = u32::from_le_bytes(stored_len_crc_bytes);
    let computed_len_crc = masked_crc32c(&bytes[offset..offset + 8]);
    if stored_len_crc != computed_len_crc {
        return Err(invalid(
            path,
            format!("invalid TFRecord length CRC at byte offset {offset}"),
        ));
    }

    let data_start = header_end;
    let data_end = data_start
        .checked_add(length_usize)
        .ok_or_else(|| invalid(path, "record data offset overflow"))?;
    let footer_end = data_end
        .checked_add(4)
        .ok_or_else(|| invalid(path, "record footer offset overflow"))?;
    if footer_end > bytes.len() {
        return Err(invalid(path, "truncated TFRecord frame data"));
    }

    let mut stored_data_crc_bytes = [0u8; 4];
    stored_data_crc_bytes.copy_from_slice(&bytes[data_end..footer_end]);
    let stored_data_crc = u32::from_le_bytes(stored_data_crc_bytes);
    let payload = &bytes[data_start..data_end];
    let computed_data_crc = masked_crc32c(payload);
    if stored_data_crc != computed_data_crc {
        return Err(invalid(
            path,
            format!("invalid TFRecord data CRC at byte offset {data_start}"),
        ));
    }

    Ok(Some((payload, footer_end)))
}

fn write_frame(output: &mut Vec<u8>, payload: &[u8]) {
    let length = payload.len() as u64;
    let len_bytes = length.to_le_bytes();
    output.extend_from_slice(&len_bytes);
    output.extend_from_slice(&masked_crc32c(&len_bytes).to_le_bytes());
    output.extend_from_slice(payload);
    output.extend_from_slice(&masked_crc32c(payload).to_le_bytes());
}

fn masked_crc32c(bytes: &[u8]) -> u32 {
    let crc = crc32c::crc32c(bytes);
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

fn parse_example(
    example: Example,
    path: &Path,
    record_index: usize,
) -> Result<ParsedRecord, PanlabelError> {
    let features = features_to_map(example.features.as_ref());
    if !has_tfod_evidence(&features) {
        return Err(invalid(
            path,
            format!(
                "TFRecord framing is valid, but record {record_index} is not a supported TensorFlow Object Detection API-style tf.train.Example"
            ),
        ));
    }

    let file_name = required_bytes_string(&features, IMAGE_FILENAME, path, record_index)?;
    let width = required_i64_scalar(&features, IMAGE_WIDTH, path, record_index)?;
    let height = required_i64_scalar(&features, IMAGE_HEIGHT, path, record_index)?;
    let width = positive_u32(width, IMAGE_WIDTH, path, record_index)?;
    let height = positive_u32(height, IMAGE_HEIGHT, path, record_index)?;

    let mut image_attributes = BTreeMap::new();
    copy_optional_bytes_attr(
        &features,
        IMAGE_SOURCE_ID,
        ATTR_SOURCE_ID,
        &mut image_attributes,
    );
    copy_optional_bytes_attr(
        &features,
        IMAGE_KEY_SHA256,
        ATTR_KEY_SHA256,
        &mut image_attributes,
    );
    copy_optional_bytes_attr(&features, IMAGE_FORMAT, ATTR_FORMAT, &mut image_attributes);
    if bytes_values(&features, IMAGE_ENCODED).is_some_and(|values| !values.is_empty()) {
        image_attributes.insert(ATTR_HAD_ENCODED_IMAGE.to_string(), "true".to_string());
    }

    let unsupported_feature_keys = unsupported_feature_keys(&features);
    let objects = parse_objects(&features, path, record_index)?;

    Ok(ParsedRecord {
        record_index,
        file_name,
        width,
        height,
        image_attributes,
        objects,
        unsupported_feature_keys,
    })
}

fn parse_objects(
    features: &BTreeMap<String, Feature>,
    path: &Path,
    record_index: usize,
) -> Result<Vec<ParsedObject>, PanlabelError> {
    let xmin = float_values(features, BBOX_XMIN).unwrap_or_default();
    let xmax = float_values(features, BBOX_XMAX).unwrap_or_default();
    let ymin = float_values(features, BBOX_YMIN).unwrap_or_default();
    let ymax = float_values(features, BBOX_YMAX).unwrap_or_default();
    let lengths = [xmin.len(), xmax.len(), ymin.len(), ymax.len()];
    let object_count = *lengths.iter().max().unwrap_or(&0);
    if object_count == 0 {
        return Ok(Vec::new());
    }
    if lengths.iter().any(|&len| len != object_count) {
        return Err(invalid(
            path,
            format!(
                "record {record_index} has mismatched bbox list lengths: xmin={}, xmax={}, ymin={}, ymax={}",
                xmin.len(), xmax.len(), ymin.len(), ymax.len()
            ),
        ));
    }

    let class_text = bytes_values(features, CLASS_TEXT).unwrap_or_default();
    let class_label = int64_values(features, CLASS_LABEL).unwrap_or_default();
    if !class_text.is_empty() && class_text.len() != object_count {
        return Err(invalid(
            path,
            format!(
                "record {record_index} has {} class text values for {object_count} boxes",
                class_text.len()
            ),
        ));
    }
    if class_text.is_empty() && class_label.len() != object_count {
        return Err(invalid(
            path,
            format!(
                "record {record_index} has no class text and {} class label values for {object_count} boxes",
                class_label.len()
            ),
        ));
    }

    let area = float_values(features, AREA).unwrap_or_default();
    let is_crowd = int64_values(features, IS_CROWD).unwrap_or_default();
    let difficult = int64_values(features, DIFFICULT).unwrap_or_default();
    let group_of = int64_values(features, GROUP_OF).unwrap_or_default();
    let weight = float_values(features, WEIGHT).unwrap_or_default();

    let mut objects = Vec::with_capacity(object_count);
    for idx in 0..object_count {
        let label = class_label.get(idx).copied();
        let class_name = if let Some(raw) = class_text.get(idx) {
            decode_utf8(raw, path, CLASS_TEXT, record_index)?.to_string()
        } else {
            label.expect("class label length checked").to_string()
        };

        let mut attributes = BTreeMap::new();
        if let Some(value) = label {
            attributes.insert(ATTR_CLASS_LABEL.to_string(), value.to_string());
        }
        copy_optional_indexed_f32(&area, idx, ATTR_AREA, &mut attributes);
        copy_optional_indexed_i64(&is_crowd, idx, ATTR_IS_CROWD, &mut attributes);
        copy_optional_indexed_i64(&difficult, idx, ATTR_DIFFICULT, &mut attributes);
        copy_optional_indexed_i64(&group_of, idx, ATTR_GROUP_OF, &mut attributes);
        copy_optional_indexed_f32(&weight, idx, ATTR_WEIGHT, &mut attributes);

        objects.push(ParsedObject {
            class_name,
            bbox_norm: BBoxXYXY::<Normalized>::from_xyxy(
                xmin[idx] as f64,
                ymin[idx] as f64,
                xmax[idx] as f64,
                ymax[idx] as f64,
            ),
            attributes,
        });
    }

    Ok(objects)
}

fn records_to_dataset(records: Vec<ParsedRecord>, path: &Path) -> Result<Dataset, PanlabelError> {
    let mut image_dims: BTreeMap<String, (u32, u32, BTreeMap<String, String>)> = BTreeMap::new();
    let mut category_names = BTreeSet::new();
    let mut unsupported = BTreeSet::new();

    for record in &records {
        match image_dims.get_mut(&record.file_name) {
            Some((width, height, attrs)) => {
                if *width != record.width || *height != record.height {
                    return Err(invalid(
                        path,
                        format!(
                            "inconsistent dimensions for '{}': existing {}x{}, record {} has {}x{}",
                            record.file_name,
                            *width,
                            *height,
                            record.record_index,
                            record.width,
                            record.height
                        ),
                    ));
                }
                for (key, value) in &record.image_attributes {
                    attrs.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
            None => {
                image_dims.insert(
                    record.file_name.clone(),
                    (record.width, record.height, record.image_attributes.clone()),
                );
            }
        }
        for object in &record.objects {
            category_names.insert(object.class_name.clone());
        }
        for key in &record.unsupported_feature_keys {
            unsupported.insert(key.clone());
        }
    }

    let image_map: BTreeMap<String, ImageId> = image_dims
        .keys()
        .enumerate()
        .map(|(idx, file_name)| (file_name.clone(), ImageId::new((idx + 1) as u64)))
        .collect();
    let category_map: BTreeMap<String, CategoryId> = category_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), CategoryId::new((idx + 1) as u64)))
        .collect();

    let images = image_dims
        .iter()
        .map(|(file_name, (width, height, attrs))| {
            let mut image = Image::new(image_map[file_name], file_name.clone(), *width, *height);
            image.attributes = attrs.clone();
            image
        })
        .collect();
    let categories = category_names
        .iter()
        .map(|name| Category::new(category_map[name], name.clone()))
        .collect();

    let mut annotations = Vec::new();
    for record in &records {
        let image_id = image_map[&record.file_name];
        for object in &record.objects {
            let mut annotation = Annotation::new(
                AnnotationId::new((annotations.len() + 1) as u64),
                image_id,
                category_map[&object.class_name],
                object
                    .bbox_norm
                    .to_pixel(record.width as f64, record.height as f64),
            );
            annotation.attributes = object.attributes.clone();
            annotations.push(annotation);
        }
    }

    let mut info = DatasetInfo::default();
    if !unsupported.is_empty() {
        info.attributes.insert(
            ATTR_UNSUPPORTED_FEATURE_COUNT.to_string(),
            unsupported.len().to_string(),
        );
        info.attributes.insert(
            ATTR_UNSUPPORTED_FEATURE_KEYS.to_string(),
            unsupported.into_iter().collect::<Vec<_>>().join(","),
        );
    }

    Ok(Dataset {
        info,
        licenses: Vec::new(),
        images,
        categories,
        annotations,
    })
}

fn dataset_to_examples(dataset: &Dataset, path: &Path) -> Result<Vec<Example>, PanlabelError> {
    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|i| (i.id, i)).collect();
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|c| (c.id, c)).collect();
    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();

    for ann in &dataset.annotations {
        if !image_lookup.contains_key(&ann.image_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if !category_lookup.contains_key(&ann.category_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            ));
        }
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut images: Vec<&Image> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name).then(a.id.cmp(&b.id)));

    let mut examples = Vec::with_capacity(images.len());
    for image in images {
        if image.width == 0 || image.height == 0 {
            return Err(invalid(
                path,
                format!("image '{}' has zero width or height", image.file_name),
            ));
        }
        let mut annotations = anns_by_image.remove(&image.id).unwrap_or_default();
        annotations.sort_by_key(|ann| ann.id);
        examples.push(image_to_example(image, &annotations, &category_lookup));
    }

    Ok(examples)
}

fn image_to_example(
    image: &Image,
    annotations: &[&Annotation],
    category_lookup: &BTreeMap<CategoryId, &Category>,
) -> Example {
    let mut features = BTreeMap::new();
    features.insert(
        IMAGE_FILENAME.to_string(),
        bytes_feature_one(image.file_name.as_bytes()),
    );
    features.insert(
        IMAGE_WIDTH.to_string(),
        int64_feature(vec![image.width as i64]),
    );
    features.insert(
        IMAGE_HEIGHT.to_string(),
        int64_feature(vec![image.height as i64]),
    );

    if let Some(value) = image.attributes.get(ATTR_SOURCE_ID) {
        features.insert(
            IMAGE_SOURCE_ID.to_string(),
            bytes_feature_one(value.as_bytes()),
        );
    }
    if let Some(value) = image.attributes.get(ATTR_KEY_SHA256) {
        features.insert(
            IMAGE_KEY_SHA256.to_string(),
            bytes_feature_one(value.as_bytes()),
        );
    }
    if let Some(value) = image.attributes.get(ATTR_FORMAT) {
        features.insert(
            IMAGE_FORMAT.to_string(),
            bytes_feature_one(value.as_bytes()),
        );
    }

    let mut xmin = Vec::new();
    let mut xmax = Vec::new();
    let mut ymin = Vec::new();
    let mut ymax = Vec::new();
    let mut class_text = Vec::new();
    let mut class_label = Vec::new();
    for ann in annotations {
        let category = category_lookup[&ann.category_id];
        let bbox = ann
            .bbox
            .to_normalized(image.width as f64, image.height as f64);
        xmin.push(bbox.xmin() as f32);
        xmax.push(bbox.xmax() as f32);
        ymin.push(bbox.ymin() as f32);
        ymax.push(bbox.ymax() as f32);
        class_text.push(category.name.as_bytes().to_vec());
        class_label.push(
            ann.attributes
                .get(ATTR_CLASS_LABEL)
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or_else(|| category.id.as_u64() as i64),
        );
    }

    features.insert(BBOX_XMIN.to_string(), float_feature(xmin));
    features.insert(BBOX_XMAX.to_string(), float_feature(xmax));
    features.insert(BBOX_YMIN.to_string(), float_feature(ymin));
    features.insert(BBOX_YMAX.to_string(), float_feature(ymax));
    features.insert(CLASS_TEXT.to_string(), bytes_feature(class_text));
    features.insert(CLASS_LABEL.to_string(), int64_feature(class_label));
    insert_complete_f32_attr(&mut features, annotations, AREA, ATTR_AREA);
    insert_complete_i64_attr(&mut features, annotations, IS_CROWD, ATTR_IS_CROWD);
    insert_complete_i64_attr(&mut features, annotations, DIFFICULT, ATTR_DIFFICULT);
    insert_complete_i64_attr(&mut features, annotations, GROUP_OF, ATTR_GROUP_OF);
    insert_complete_f32_attr(&mut features, annotations, WEIGHT, ATTR_WEIGHT);

    Example {
        features: Some(map_to_features(features)),
    }
}

fn features_to_map(features: Option<&Features>) -> BTreeMap<String, Feature> {
    let mut map = BTreeMap::new();
    if let Some(features) = features {
        for entry in &features.feature {
            if let Some(value) = &entry.value {
                map.insert(entry.key.clone(), value.clone());
            }
        }
    }
    map
}

fn map_to_features(map: BTreeMap<String, Feature>) -> Features {
    Features {
        feature: map
            .into_iter()
            .map(|(key, value)| FeatureEntry {
                key,
                value: Some(value),
            })
            .collect(),
    }
}

fn has_tfod_evidence(features: &BTreeMap<String, Feature>) -> bool {
    features.contains_key(IMAGE_WIDTH)
        && features.contains_key(IMAGE_HEIGHT)
        && (features.contains_key(IMAGE_FILENAME)
            || features.contains_key(IMAGE_ENCODED)
            || features.contains_key(BBOX_XMIN)
            || features.contains_key(BBOX_YMIN)
            || features.contains_key(BBOX_XMAX)
            || features.contains_key(BBOX_YMAX))
}

fn unsupported_feature_keys(features: &BTreeMap<String, Feature>) -> Vec<String> {
    features
        .keys()
        .filter(|key| {
            key.contains("mask")
                || key.contains("keypoint")
                || key.contains("segmentation")
                || key.contains("densepose")
        })
        .cloned()
        .collect()
}

fn required_bytes_string(
    features: &BTreeMap<String, Feature>,
    key: &str,
    path: &Path,
    record_index: usize,
) -> Result<String, PanlabelError> {
    let values = bytes_values(features, key).ok_or_else(|| {
        invalid(
            path,
            format!("record {record_index} missing required bytes feature '{key}'"),
        )
    })?;
    let value = values.first().ok_or_else(|| {
        invalid(
            path,
            format!("record {record_index} feature '{key}' is empty"),
        )
    })?;
    Ok(decode_utf8(value, path, key, record_index)?.to_string())
}

fn required_i64_scalar(
    features: &BTreeMap<String, Feature>,
    key: &str,
    path: &Path,
    record_index: usize,
) -> Result<i64, PanlabelError> {
    let values = int64_values(features, key).ok_or_else(|| {
        invalid(
            path,
            format!("record {record_index} missing required int64 feature '{key}'"),
        )
    })?;
    values.first().copied().ok_or_else(|| {
        invalid(
            path,
            format!("record {record_index} feature '{key}' is empty"),
        )
    })
}

fn positive_u32(
    value: i64,
    key: &str,
    path: &Path,
    record_index: usize,
) -> Result<u32, PanlabelError> {
    if value <= 0 || value > u32::MAX as i64 {
        return Err(invalid(
            path,
            format!("record {record_index} feature '{key}' must be a positive u32, got {value}"),
        ));
    }
    Ok(value as u32)
}

fn bytes_values(features: &BTreeMap<String, Feature>, key: &str) -> Option<Vec<Vec<u8>>> {
    match features.get(key)?.kind.as_ref()? {
        feature::Kind::BytesList(list) => Some(list.value.clone()),
        _ => None,
    }
}

fn float_values(features: &BTreeMap<String, Feature>, key: &str) -> Option<Vec<f32>> {
    match features.get(key)?.kind.as_ref()? {
        feature::Kind::FloatList(list) => Some(list.value.clone()),
        _ => None,
    }
}

fn int64_values(features: &BTreeMap<String, Feature>, key: &str) -> Option<Vec<i64>> {
    match features.get(key)?.kind.as_ref()? {
        feature::Kind::Int64List(list) => Some(list.value.clone()),
        _ => None,
    }
}

fn decode_utf8<'a>(
    value: &'a [u8],
    path: &Path,
    key: &str,
    record_index: usize,
) -> Result<&'a str, PanlabelError> {
    std::str::from_utf8(value).map_err(|source| {
        invalid(
            path,
            format!("record {record_index} feature '{key}' is not valid UTF-8: {source}"),
        )
    })
}

fn copy_optional_bytes_attr(
    features: &BTreeMap<String, Feature>,
    feature_key: &str,
    attr_key: &str,
    attrs: &mut BTreeMap<String, String>,
) {
    if let Some(values) = bytes_values(features, feature_key) {
        if let Some(value) = values.first() {
            if let Ok(text) = std::str::from_utf8(value) {
                attrs.insert(attr_key.to_string(), text.to_string());
            }
        }
    }
}

fn copy_optional_indexed_f32(
    values: &[f32],
    idx: usize,
    key: &str,
    attrs: &mut BTreeMap<String, String>,
) {
    if let Some(value) = values.get(idx) {
        attrs.insert(key.to_string(), value.to_string());
    }
}

fn copy_optional_indexed_i64(
    values: &[i64],
    idx: usize,
    key: &str,
    attrs: &mut BTreeMap<String, String>,
) {
    if let Some(value) = values.get(idx) {
        attrs.insert(key.to_string(), value.to_string());
    }
}

fn insert_complete_f32_attr(
    features: &mut BTreeMap<String, Feature>,
    annotations: &[&Annotation],
    feature_key: &str,
    attr_key: &str,
) {
    let values: Option<Vec<f32>> = annotations
        .iter()
        .map(|ann| ann.attributes.get(attr_key)?.parse::<f32>().ok())
        .collect();
    if let Some(values) = values.filter(|values| !values.is_empty()) {
        features.insert(feature_key.to_string(), float_feature(values));
    }
}

fn insert_complete_i64_attr(
    features: &mut BTreeMap<String, Feature>,
    annotations: &[&Annotation],
    feature_key: &str,
    attr_key: &str,
) {
    let values: Option<Vec<i64>> = annotations
        .iter()
        .map(|ann| ann.attributes.get(attr_key)?.parse::<i64>().ok())
        .collect();
    if let Some(values) = values.filter(|values| !values.is_empty()) {
        features.insert(feature_key.to_string(), int64_feature(values));
    }
}

fn bytes_feature_one(value: &[u8]) -> Feature {
    bytes_feature(vec![value.to_vec()])
}

fn bytes_feature(values: Vec<Vec<u8>>) -> Feature {
    Feature {
        kind: Some(feature::Kind::BytesList(BytesList { value: values })),
    }
}

fn float_feature(values: Vec<f32>) -> Feature {
    Feature {
        kind: Some(feature::Kind::FloatList(FloatList { value: values })),
    }
}

fn int64_feature(values: Vec<i64>) -> Feature {
    Feature {
        kind: Some(feature::Kind::Int64List(Int64List { value: values })),
    }
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::TfrecordInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    fn sample_dataset() -> Dataset {
        Dataset {
            images: vec![
                Image::new(2u64, "z.jpg", 200, 100),
                Image::new(1u64, "a.jpg", 100, 50),
            ],
            categories: vec![Category::new(2u64, "dog"), Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 5.0, 60.0, 45.0),
            )],
            ..Default::default()
        }
    }

    #[test]
    fn frame_roundtrip_validates_crc() {
        let payload = b"hello";
        let mut framed = Vec::new();
        write_frame(&mut framed, payload);
        let (restored, next) = read_one_frame(&framed, 0, Path::new("test.tfrecord"))
            .expect("frame should parse")
            .expect("frame should exist");
        assert_eq!(restored, payload);
        assert_eq!(next, framed.len());
    }

    #[test]
    fn corrupt_crc_fails() {
        let mut framed = Vec::new();
        write_frame(&mut framed, b"hello");
        let last = framed.len() - 1;
        framed[last] ^= 0xff;
        let err = read_one_frame(&framed, 0, Path::new("bad.tfrecord")).unwrap_err();
        assert!(err.to_string().contains("data CRC"));
    }

    #[test]
    fn dataset_roundtrip_preserves_unannotated_images() {
        let original = sample_dataset();
        let bytes = to_tfrecord_vec(&original).expect("serialize");
        let restored = from_tfrecord_slice(&bytes).expect("parse");
        assert_eq!(restored.images.len(), 2);
        assert_eq!(restored.annotations.len(), 1);
        assert!(restored.images.iter().any(|img| img.file_name == "z.jpg"));
    }

    #[test]
    fn output_is_deterministic() {
        let dataset = sample_dataset();
        let first = to_tfrecord_vec(&dataset).expect("first");
        let second = to_tfrecord_vec(&dataset).expect("second");
        assert_eq!(first, second);
    }

    #[test]
    fn numeric_labels_work_without_class_text() {
        let mut features = BTreeMap::new();
        features.insert(IMAGE_FILENAME.to_string(), bytes_feature_one(b"img.jpg"));
        features.insert(IMAGE_WIDTH.to_string(), int64_feature(vec![100]));
        features.insert(IMAGE_HEIGHT.to_string(), int64_feature(vec![100]));
        features.insert(BBOX_XMIN.to_string(), float_feature(vec![0.1]));
        features.insert(BBOX_XMAX.to_string(), float_feature(vec![0.5]));
        features.insert(BBOX_YMIN.to_string(), float_feature(vec![0.2]));
        features.insert(BBOX_YMAX.to_string(), float_feature(vec![0.6]));
        features.insert(CLASS_LABEL.to_string(), int64_feature(vec![7]));
        let example = Example {
            features: Some(map_to_features(features)),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &example.encode_to_vec());

        let dataset = from_tfrecord_slice(&bytes).expect("parse");
        assert_eq!(dataset.categories[0].name, "7");
        assert_eq!(dataset.annotations[0].attributes[ATTR_CLASS_LABEL], "7");
    }

    #[test]
    fn encoded_image_bytes_are_not_stored_in_ir() {
        let mut features = BTreeMap::new();
        features.insert(IMAGE_FILENAME.to_string(), bytes_feature_one(b"img.jpg"));
        features.insert(IMAGE_WIDTH.to_string(), int64_feature(vec![100]));
        features.insert(IMAGE_HEIGHT.to_string(), int64_feature(vec![80]));
        features.insert(
            IMAGE_ENCODED.to_string(),
            bytes_feature_one(b"fake image bytes"),
        );
        let example = Example {
            features: Some(map_to_features(features)),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &example.encode_to_vec());

        let dataset = from_tfrecord_slice(&bytes).expect("parse");
        assert_eq!(dataset.images[0].attributes[ATTR_HAD_ENCODED_IMAGE], "true");
        assert!(!dataset.images[0]
            .attributes
            .values()
            .any(|value| value.contains("fake image bytes")));
    }
}
