//! Marmot XML reader and writer.
//!
//! Marmot stores document-layout annotations as one XML page per companion
//! image. Numeric rectangles are four 16-hex-token big-endian f64 values in
//! page space: `x_left y_top x_right y_bottom`. Panlabel's IR uses pixel-space
//! XYXY with a top-left origin, so the reader scales through the page CropBox
//! and flips the Y axis.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use roxmltree::Node;

use super::io_adapter_common::{
    has_extension, is_safe_relative_image_ref, normalize_path_separators,
};
use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

pub const ATTR_XML_PATH: &str = "marmot_xml_path";
pub const ATTR_CROPBOX_HEX: &str = "marmot_cropbox_hex";
pub const ATTR_BBOX_HEX: &str = "marmot_bbox_hex";
pub const ATTR_BBOX_SOURCE: &str = "marmot_bbox_source";
pub const BBOX_SOURCE_CROPBOX_TRANSFORM: &str = "cropbox_transform";

const MEMORY_PATH: &str = "<memory>";
const DEFAULT_LABEL: &str = "Composite";
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tif", "tiff"];

#[derive(Clone, Copy, Debug)]
struct MarmotRect {
    x_left: f64,
    y_top: f64,
    x_right: f64,
    y_bottom: f64,
}

#[derive(Debug)]
struct ParsedPage {
    xml_path: PathBuf,
    xml_rel_path: Option<String>,
    image_file_name: String,
    width: u32,
    height: u32,
    cropbox_hex: String,
    composites: Vec<ParsedComposite>,
}

#[derive(Debug)]
struct ParsedComposite {
    label: String,
    bbox: BBoxXYXY<Pixel>,
    attributes: BTreeMap<String, String>,
}

/// Read Marmot XML from a single XML file or a directory of XML files.
pub fn read_marmot_xml(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        let parsed = parse_marmot_file(path, None)?;
        dataset_from_pages(vec![parsed])
    } else if path.is_dir() {
        read_marmot_dir(path)
    } else {
        Err(invalid(
            path,
            "path must be a Marmot XML file or a directory containing Marmot XML files",
        ))
    }
}

/// Write panlabel IR as deterministic minimal Marmot XML.
pub fn write_marmot_xml(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    if has_extension(path, "xml") {
        if dataset.images.len() != 1 {
            return Err(write_error(
                path,
                format!(
                    "single-file Marmot XML output requires exactly 1 image, got {}",
                    dataset.images.len()
                ),
            ));
        }
        write_single_file(path, dataset, &dataset.images[0])
    } else {
        write_directory(path, dataset)
    }
}

/// Parse a single Marmot XML string into IR.
pub fn from_marmot_xml_str(
    xml: &str,
    image_file_name: &str,
    width: u32,
    height: u32,
) -> Result<Dataset, PanlabelError> {
    let parsed = parse_marmot_xml_str(
        xml,
        Path::new(MEMORY_PATH),
        None,
        image_file_name.to_string(),
        width,
        height,
    )?;
    dataset_from_pages(vec![parsed])
}

/// Serialize a single-image dataset to Marmot XML text.
pub fn to_marmot_xml_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    if dataset.images.len() != 1 {
        return Err(write_error(
            Path::new(MEMORY_PATH),
            format!(
                "single-file Marmot XML output requires exactly 1 image, got {}",
                dataset.images.len()
            ),
        ));
    }
    dataset_image_to_xml(dataset, &dataset.images[0], Path::new(MEMORY_PATH))
}

/// Quick structural check used by CLI autodetection.
pub fn is_likely_marmot_xml_file(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_file() || !has_extension(path, "xml") {
        return Ok(false);
    }
    let xml = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    is_likely_marmot_xml_str(&xml, path)
}

/// Returns true when the XML root is `<Page>` with a valid Marmot CropBox.
pub fn is_likely_marmot_xml_str(xml: &str, path: &Path) -> Result<bool, PanlabelError> {
    let doc = match roxmltree::Document::parse(xml) {
        Ok(doc) => doc,
        Err(_) => return Ok(false),
    };
    let root = doc.root_element();
    if root.tag_name().name() != "Page" {
        return Ok(false);
    }
    let Some(cropbox) = root.attribute("CropBox") else {
        return Ok(false);
    };
    Ok(decode_hex_rect(cropbox, path, "CropBox").is_ok())
}

/// Returns true when a Marmot XML file has a same-stem companion image.
pub fn has_companion_image(path: &Path) -> bool {
    resolve_companion_image(path).is_some()
}

fn is_marmot_page_xml_file(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_file() || !has_extension(path, "xml") {
        return Ok(false);
    }
    let xml = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let doc = match roxmltree::Document::parse(&xml) {
        Ok(doc) => doc,
        Err(_) => return Ok(false),
    };
    Ok(doc.root_element().tag_name().name() == "Page")
}

fn read_marmot_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let xml_files = collect_marmot_xml_files(path)?;
    if xml_files.is_empty() {
        return Err(invalid(path, "no Marmot XML files found"));
    }

    let mut pages = Vec::with_capacity(xml_files.len());
    for xml_path in xml_files {
        pages.push(parse_marmot_file(&xml_path, Some(path))?);
    }
    dataset_from_pages(pages)
}

fn collect_marmot_xml_files(root: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(true) {
        let entry = entry.map_err(|source| {
            invalid(root, format!("failed while traversing directory: {source}"))
        })?;
        let path = entry.path();
        if entry.file_type().is_file()
            && has_extension(path, "xml")
            && is_marmot_page_xml_file(path)?
        {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn parse_marmot_file(
    path: &Path,
    dataset_root: Option<&Path>,
) -> Result<ParsedPage, PanlabelError> {
    let image_path =
        resolve_companion_image(path).ok_or_else(|| PanlabelError::MarmotImageNotFound {
            path: path.to_path_buf(),
            searched: companion_image_candidates(path)
                .iter()
                .map(|candidate| candidate.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        })?;
    let size =
        imagesize::size(&image_path).map_err(|source| PanlabelError::MarmotImageDimensionRead {
            path: image_path.clone(),
            source,
        })?;
    let image_file_name = image_file_name(&image_path, dataset_root);
    let xml_rel_path = dataset_root.map(|root| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    });

    let xml = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    parse_marmot_xml_str(
        &xml,
        path,
        xml_rel_path,
        image_file_name,
        size.width as u32,
        size.height as u32,
    )
}

fn parse_marmot_xml_str(
    xml: &str,
    path: &Path,
    xml_rel_path: Option<String>,
    image_file_name: String,
    width: u32,
    height: u32,
) -> Result<ParsedPage, PanlabelError> {
    let doc = roxmltree::Document::parse(xml).map_err(|source| PanlabelError::MarmotXmlParse {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    let root = doc.root_element();
    if root.tag_name().name() != "Page" {
        return Err(invalid(path, "Marmot XML root must be <Page>"));
    }

    let cropbox_hex = root
        .attribute("CropBox")
        .ok_or_else(|| invalid(path, "missing required Page@CropBox attribute"))?
        .trim()
        .to_string();
    let cropbox = decode_hex_rect(&cropbox_hex, path, "CropBox")?;
    validate_cropbox(cropbox, path)?;

    let mut composites = Vec::new();
    for node in root
        .descendants()
        .filter(|node| is_composite_under_composites(*node))
    {
        let bbox_hex = node
            .attribute("BBox")
            .ok_or_else(|| invalid(path, "Composite is missing required BBox attribute"))?
            .trim()
            .to_string();
        let rect = decode_hex_rect(&bbox_hex, path, "BBox")?;
        let bbox = page_rect_to_pixel_bbox(rect, cropbox, width, height);
        let parent_label = node.parent().and_then(|parent| parent.attribute("Label"));
        let label = node
            .attribute("Label")
            .or(parent_label)
            .unwrap_or(DEFAULT_LABEL)
            .trim();
        let label = if label.is_empty() {
            DEFAULT_LABEL
        } else {
            label
        }
        .to_string();

        let mut attributes = BTreeMap::new();
        attributes.insert(ATTR_BBOX_HEX.to_string(), bbox_hex.clone());
        attributes.insert(
            ATTR_BBOX_SOURCE.to_string(),
            BBOX_SOURCE_CROPBOX_TRANSFORM.to_string(),
        );
        copy_attr(&mut attributes, &node, "LID", "marmot_lid");
        copy_attr(&mut attributes, &node, "PLID", "marmot_plid");
        copy_attr(&mut attributes, &node, "CLIDs", "marmot_clids");

        composites.push(ParsedComposite {
            label,
            bbox,
            attributes,
        });
    }

    Ok(ParsedPage {
        xml_path: path.to_path_buf(),
        xml_rel_path,
        image_file_name,
        width,
        height,
        cropbox_hex,
        composites,
    })
}

fn dataset_from_pages(pages: Vec<ParsedPage>) -> Result<Dataset, PanlabelError> {
    let mut image_defs = BTreeMap::new();
    let mut category_names = BTreeSet::new();
    for page in &pages {
        if image_defs.contains_key(&page.image_file_name) {
            return Err(invalid(
                &page.xml_path,
                format!(
                    "duplicate companion image '{}' found in multiple Marmot XML files",
                    page.image_file_name
                ),
            ));
        }
        image_defs.insert(
            page.image_file_name.clone(),
            (
                page.width,
                page.height,
                page.xml_rel_path.clone(),
                page.cropbox_hex.clone(),
            ),
        );
        for composite in &page.composites {
            category_names.insert(composite.label.clone());
        }
    }

    let categories: Vec<Category> = category_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name))
        .collect();
    let category_id_by_name: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let images: Vec<Image> = image_defs
        .into_iter()
        .enumerate()
        .map(
            |(idx, (file_name, (width, height, xml_rel_path, cropbox_hex)))| {
                let mut image = Image::new((idx + 1) as u64, file_name, width, height);
                if let Some(xml_rel_path) = xml_rel_path {
                    image
                        .attributes
                        .insert(ATTR_XML_PATH.to_string(), xml_rel_path);
                }
                image
                    .attributes
                    .insert(ATTR_CROPBOX_HEX.to_string(), cropbox_hex);
                image
            },
        )
        .collect();
    let image_id_by_name: BTreeMap<String, ImageId> = images
        .iter()
        .map(|image| (image.file_name.clone(), image.id))
        .collect();

    let mut annotations = Vec::new();
    let mut next_annotation_id = 1u64;
    for page in pages {
        let image_id = image_id_by_name[&page.image_file_name];
        for composite in page.composites {
            let category_id = category_id_by_name[&composite.label];
            let mut annotation = Annotation::new(
                AnnotationId::new(next_annotation_id),
                image_id,
                category_id,
                composite.bbox,
            );
            annotation.attributes = composite.attributes;
            annotations.push(annotation);
            next_annotation_id += 1;
        }
    }

    Ok(Dataset {
        info: DatasetInfo::default(),
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

fn write_single_file(path: &Path, dataset: &Dataset, image: &Image) -> Result<(), PanlabelError> {
    let xml = dataset_image_to_xml(dataset, image, path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }
    fs::write(path, xml).map_err(PanlabelError::Io)
}

fn write_directory(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    for image in images_sorted {
        let xml_rel_path = marmot_output_rel_path(&image.file_name, path)?;
        let xml_path = path.join(xml_rel_path);
        write_single_file(&xml_path, dataset, image)?;
    }
    Ok(())
}

fn marmot_output_rel_path(
    image_file_name: &str,
    error_path: &Path,
) -> Result<PathBuf, PanlabelError> {
    if !is_safe_relative_image_ref(image_file_name) {
        return Err(write_error(
            error_path,
            format!(
                "image file_name '{image_file_name}' cannot be used for Marmot directory output because it must be a safe relative path"
            ),
        ));
    }
    Ok(Path::new(image_file_name).with_extension("xml"))
}

fn dataset_image_to_xml(
    dataset: &Dataset,
    image: &Image,
    error_path: &Path,
) -> Result<String, PanlabelError> {
    let image_ids: BTreeSet<ImageId> = dataset.images.iter().map(|img| img.id).collect();
    let category_name_by_id: BTreeMap<CategoryId, &str> = dataset
        .categories
        .iter()
        .map(|category| (category.id, category.name.as_str()))
        .collect();

    let mut annotations_by_label: BTreeMap<String, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_ids.contains(&ann.image_id) {
            return Err(write_error(
                error_path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if ann.image_id != image.id {
            continue;
        }
        let Some(category_name) = category_name_by_id.get(&ann.category_id) else {
            return Err(write_error(
                error_path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            ));
        };
        annotations_by_label
            .entry((*category_name).to_string())
            .or_default()
            .push(ann);
    }
    for anns in annotations_by_label.values_mut() {
        anns.sort_by_key(|ann| ann.id);
    }

    let cropbox = MarmotRect {
        x_left: 0.0,
        y_top: image.height as f64,
        x_right: image.width as f64,
        y_bottom: 0.0,
    };
    let cropbox_hex = encode_hex_rect(cropbox);

    let mut xml = String::new();
    writeln!(&mut xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>").unwrap();
    writeln!(&mut xml, "<Page CropBox=\"{}\">", cropbox_hex).unwrap();
    for (label, anns) in annotations_by_label {
        writeln!(
            &mut xml,
            "  <Composites Label=\"{}\">",
            escape_xml_attr(&label)
        )
        .unwrap();
        for ann in anns {
            let rect = pixel_bbox_to_page_rect(ann.bbox, cropbox, image.width, image.height);
            let bbox_hex = encode_hex_rect(rect);
            writeln!(
                &mut xml,
                "    <Composite BBox=\"{}\" LID=\"{}\" Label=\"{}\" />",
                bbox_hex,
                ann.id.as_u64(),
                escape_xml_attr(&label)
            )
            .unwrap();
        }
        writeln!(&mut xml, "  </Composites>").unwrap();
    }
    writeln!(&mut xml, "</Page>").unwrap();
    Ok(xml)
}

fn page_rect_to_pixel_bbox(
    rect: MarmotRect,
    cropbox: MarmotRect,
    image_width: u32,
    image_height: u32,
) -> BBoxXYXY<Pixel> {
    let crop_width = cropbox.x_right - cropbox.x_left;
    let crop_height = cropbox.y_top - cropbox.y_bottom;
    let x0 = ((rect.x_left - cropbox.x_left) / crop_width) * image_width as f64;
    let x1 = ((rect.x_right - cropbox.x_left) / crop_width) * image_width as f64;
    let y0 = ((cropbox.y_top - rect.y_top) / crop_height) * image_height as f64;
    let y1 = ((cropbox.y_top - rect.y_bottom) / crop_height) * image_height as f64;
    BBoxXYXY::<Pixel>::from_xyxy(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
}

fn pixel_bbox_to_page_rect(
    bbox: BBoxXYXY<Pixel>,
    cropbox: MarmotRect,
    image_width: u32,
    image_height: u32,
) -> MarmotRect {
    let crop_width = cropbox.x_right - cropbox.x_left;
    let crop_height = cropbox.y_top - cropbox.y_bottom;
    MarmotRect {
        x_left: cropbox.x_left + (bbox.xmin() / image_width as f64) * crop_width,
        y_top: cropbox.y_top - (bbox.ymin() / image_height as f64) * crop_height,
        x_right: cropbox.x_left + (bbox.xmax() / image_width as f64) * crop_width,
        y_bottom: cropbox.y_top - (bbox.ymax() / image_height as f64) * crop_height,
    }
}

fn is_composite_under_composites(node: Node<'_, '_>) -> bool {
    node.is_element()
        && node.tag_name().name() == "Composite"
        && node
            .parent()
            .map(|parent| parent.is_element() && parent.tag_name().name() == "Composites")
            .unwrap_or(false)
}

fn copy_attr(
    attrs: &mut BTreeMap<String, String>,
    node: &Node<'_, '_>,
    source: &str,
    target: &str,
) {
    if let Some(value) = node.attribute(source) {
        attrs.insert(target.to_string(), value.to_string());
    }
}

fn decode_hex_rect(raw: &str, path: &Path, attr_name: &str) -> Result<MarmotRect, PanlabelError> {
    let values: Vec<f64> = raw
        .split_whitespace()
        .map(|token| decode_hex_f64(token, path, attr_name))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 4 {
        return Err(PanlabelError::MarmotXmlParse {
            path: path.to_path_buf(),
            message: format!(
                "{attr_name} must contain exactly four 16-hex-token f64 values, got {} token(s)",
                values.len()
            ),
        });
    }
    Ok(MarmotRect {
        x_left: values[0],
        y_top: values[1],
        x_right: values[2],
        y_bottom: values[3],
    })
}

fn decode_hex_f64(token: &str, path: &Path, attr_name: &str) -> Result<f64, PanlabelError> {
    if token.len() != 16 || !token.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return Err(PanlabelError::MarmotXmlParse {
            path: path.to_path_buf(),
            message: format!(
                "{attr_name} token '{token}' must be exactly 16 hexadecimal characters"
            ),
        });
    }
    let mut bytes = [0u8; 8];
    for idx in 0..8 {
        bytes[idx] = u8::from_str_radix(&token[idx * 2..idx * 2 + 2], 16).map_err(|_| {
            PanlabelError::MarmotXmlParse {
                path: path.to_path_buf(),
                message: format!("{attr_name} token '{token}' is not valid hexadecimal"),
            }
        })?;
    }
    let value = f64::from_be_bytes(bytes);
    if !value.is_finite() {
        return Err(PanlabelError::MarmotXmlParse {
            path: path.to_path_buf(),
            message: format!("{attr_name} token '{token}' decoded to a non-finite f64"),
        });
    }
    Ok(value)
}

fn encode_hex_rect(rect: MarmotRect) -> String {
    [rect.x_left, rect.y_top, rect.x_right, rect.y_bottom]
        .into_iter()
        .map(encode_hex_f64)
        .collect::<Vec<_>>()
        .join(" ")
}

fn encode_hex_f64(value: f64) -> String {
    value
        .to_be_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn validate_cropbox(cropbox: MarmotRect, path: &Path) -> Result<(), PanlabelError> {
    if cropbox.x_right == cropbox.x_left || cropbox.y_top == cropbox.y_bottom {
        return Err(PanlabelError::MarmotXmlParse {
            path: path.to_path_buf(),
            message: "Page@CropBox must have non-zero width and height".to_string(),
        });
    }
    Ok(())
}

fn resolve_companion_image(xml_path: &Path) -> Option<PathBuf> {
    companion_image_candidates(xml_path)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn companion_image_candidates(xml_path: &Path) -> Vec<PathBuf> {
    let parent = xml_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = xml_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    let mut candidates = Vec::new();
    for ext in IMAGE_EXTENSIONS {
        candidates.push(parent.join(format!("{stem}.{ext}")));
        candidates.push(parent.join("images").join(format!("{stem}.{ext}")));
    }
    if let Some(grandparent) = parent.parent() {
        for ext in IMAGE_EXTENSIONS {
            candidates.push(grandparent.join("images").join(format!("{stem}.{ext}")));
        }
    }
    candidates
}

fn image_file_name(image_path: &Path, dataset_root: Option<&Path>) -> String {
    if let Some(root) = dataset_root {
        if let Ok(rel) = image_path.strip_prefix(root) {
            return normalize_path_separators(&rel.to_string_lossy());
        }
    }
    image_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| normalize_path_separators(&image_path.to_string_lossy()))
}

fn escape_xml_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::MarmotLayoutInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn write_error(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::MarmotWriteError {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(value: f64) -> String {
        encode_hex_f64(value)
    }

    #[test]
    fn decodes_big_endian_hex_doubles() {
        let rect = decode_hex_rect(
            &format!("{} {} {} {}", hex(0.0), hex(100.0), hex(200.0), hex(0.0)),
            Path::new("sample.xml"),
            "CropBox",
        )
        .expect("decode rect");
        assert_eq!(rect.x_left, 0.0);
        assert_eq!(rect.y_top, 100.0);
        assert_eq!(rect.x_right, 200.0);
        assert_eq!(rect.y_bottom, 0.0);
    }

    #[test]
    fn rejects_wrong_hex_token_count() {
        let err = decode_hex_rect("0000000000000000", Path::new("sample.xml"), "BBox")
            .unwrap_err()
            .to_string();
        assert!(err.contains("exactly four"));
    }

    #[test]
    fn transforms_page_rect_to_pixel_xyxy() {
        let cropbox = MarmotRect {
            x_left: 0.0,
            y_top: 100.0,
            x_right: 200.0,
            y_bottom: 0.0,
        };
        let rect = MarmotRect {
            x_left: 20.0,
            y_top: 80.0,
            x_right: 120.0,
            y_bottom: 30.0,
        };
        let bbox = page_rect_to_pixel_bbox(rect, cropbox, 200, 100);
        assert_eq!(bbox.xmin(), 20.0);
        assert_eq!(bbox.ymin(), 20.0);
        assert_eq!(bbox.xmax(), 120.0);
        assert_eq!(bbox.ymax(), 70.0);
    }
}
