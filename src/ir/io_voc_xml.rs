//! Pascal VOC XML reader and writer.
//!
//! This module supports the common VOC layout with an `Annotations/` directory
//! containing one XML file per image. The canonical IR remains pixel-space XYXY.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use roxmltree::Node;
use walkdir::WalkDir;

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const VOC_XML_EXTENSION: &str = "xml";
const JPEG_IMAGES_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original images here to complete the VOC dataset layout.\n";

/// Read a Pascal VOC dataset directory into IR.
///
/// `path` may be the dataset root containing `Annotations/`, or the
/// `Annotations/` directory itself.
pub fn read_voc_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let layout = discover_layout(path)?;
    let mut xml_files = collect_xml_files(&layout.annotations_dir)?;
    xml_files.sort_by_cached_key(|xml_path| rel_string(&layout.annotations_dir, xml_path));

    let mut parsed_files = Vec::with_capacity(xml_files.len());
    for xml_path in xml_files {
        let parsed = parse_voc_xml(&xml_path)?;
        parsed_files.push((xml_path, parsed));
    }

    let mut image_defs: BTreeMap<String, (u32, u32, Option<u32>)> = BTreeMap::new();
    let mut category_names = BTreeSet::new();

    for (xml_path, parsed) in &parsed_files {
        if image_defs.contains_key(&parsed.filename) {
            return Err(PanlabelError::VocXmlParse {
                path: xml_path.clone(),
                message: format!(
                    "duplicate <filename> '{}' found in multiple XML files",
                    parsed.filename
                ),
            });
        }

        image_defs.insert(
            parsed.filename.clone(),
            (parsed.width, parsed.height, parsed.depth),
        );

        for object in &parsed.objects {
            category_names.insert(object.name.clone());
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
        .map(|(idx, (file_name, (width, height, depth)))| {
            let mut image = Image::new((idx + 1) as u64, file_name, width, height);
            if let Some(depth) = depth {
                image
                    .attributes
                    .insert("depth".to_string(), depth.to_string());
            }
            image
        })
        .collect();

    let image_id_by_name: BTreeMap<String, ImageId> = images
        .iter()
        .map(|image| (image.file_name.clone(), image.id))
        .collect();

    let mut annotations = Vec::new();
    let mut next_annotation_id: u64 = 1;

    for (xml_path, parsed) in parsed_files {
        let image_id = image_id_by_name
            .get(&parsed.filename)
            .copied()
            .ok_or_else(|| PanlabelError::VocXmlParse {
                path: xml_path.clone(),
                message: format!(
                    "internal error: image '{}' missing from lookup",
                    parsed.filename
                ),
            })?;

        for object in parsed.objects {
            let category_id = category_id_by_name
                .get(&object.name)
                .copied()
                .ok_or_else(|| PanlabelError::VocXmlParse {
                    path: xml_path.clone(),
                    message: format!(
                        "internal error: category '{}' missing from lookup",
                        object.name
                    ),
                })?;

            let mut annotation = Annotation::new(
                AnnotationId::new(next_annotation_id),
                image_id,
                category_id,
                BBoxXYXY::<Pixel>::from_xyxy(object.xmin, object.ymin, object.xmax, object.ymax),
            );
            annotation.attributes = object.attrs;

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

/// Write an IR dataset as a Pascal VOC directory.
///
/// Creates `Annotations/` and `JPEGImages/README.txt` under `path`.
pub fn write_voc_dir(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;

    let annotations_dir = path.join("Annotations");
    let jpeg_images_dir = path.join("JPEGImages");

    fs::create_dir_all(&annotations_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&jpeg_images_dir).map_err(PanlabelError::Io)?;
    fs::write(jpeg_images_dir.join("README.txt"), JPEG_IMAGES_README).map_err(PanlabelError::Io)?;

    let image_by_id: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_name_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|category| (category.id, category.name.clone()))
        .collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for annotation in &dataset.annotations {
        if !image_by_id.contains_key(&annotation.image_id) {
            return Err(PanlabelError::VocWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "annotation {} references missing image {}",
                    annotation.id.as_u64(),
                    annotation.image_id.as_u64()
                ),
            });
        }

        if !category_name_by_id.contains_key(&annotation.category_id) {
            return Err(PanlabelError::VocWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "annotation {} references missing category {}",
                    annotation.id.as_u64(),
                    annotation.category_id.as_u64()
                ),
            });
        }

        annotations_by_image
            .entry(annotation.image_id)
            .or_default()
            .push(annotation);
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|left, right| left.file_name.cmp(&right.file_name));

    for image in images_sorted {
        let xml_rel_path = Path::new(&image.file_name).with_extension(VOC_XML_EXTENSION);
        let xml_path = annotations_dir.join(&xml_rel_path);

        if let Some(parent) = xml_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }

        let mut image_annotations = annotations_by_image.remove(&image.id).unwrap_or_default();
        image_annotations.sort_by_key(|annotation| annotation.id);

        write_voc_xml(
            &xml_path,
            image,
            &image_annotations,
            &category_name_by_id,
            path,
        )?;
    }

    Ok(())
}

/// Parse VOC XML from a UTF-8 string.
///
/// This helper is primarily useful for testing/fuzzing parse behavior in-memory.
pub fn from_voc_xml_str(xml: &str) -> Result<(), PanlabelError> {
    parse_voc_xml_str(xml, Path::new("<memory>"))?;
    Ok(())
}

/// Parse VOC XML from bytes.
///
/// The input must be valid UTF-8.
pub fn from_voc_xml_slice(bytes: &[u8]) -> Result<(), PanlabelError> {
    let xml = std::str::from_utf8(bytes).map_err(|source| PanlabelError::VocXmlParse {
        path: PathBuf::from("<memory>"),
        message: format!("input is not valid UTF-8: {source}"),
    })?;
    from_voc_xml_str(xml)
}

#[derive(Clone, Debug)]
struct VocLayout {
    #[cfg_attr(not(test), allow(dead_code))]
    root: PathBuf,
    annotations_dir: PathBuf,
    #[cfg_attr(not(test), allow(dead_code))]
    images_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct ParsedVocAnnotation {
    filename: String,
    width: u32,
    height: u32,
    depth: Option<u32>,
    objects: Vec<ParsedVocObject>,
}

#[derive(Debug)]
struct ParsedVocObject {
    name: String,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    attrs: BTreeMap<String, String>,
}

fn discover_layout(input: &Path) -> Result<VocLayout, PanlabelError> {
    if !input.is_dir() {
        return Err(PanlabelError::VocLayoutInvalid {
            path: input.to_path_buf(),
            message: "input must be a directory".to_string(),
        });
    }

    let (root, annotations_dir) = if input.join("Annotations").is_dir() {
        (input.to_path_buf(), input.join("Annotations"))
    } else if is_dir_named(input, "Annotations") {
        let root = input
            .parent()
            .ok_or_else(|| PanlabelError::VocLayoutInvalid {
                path: input.to_path_buf(),
                message: "Annotations directory has no parent directory".to_string(),
            })?
            .to_path_buf();
        (root, input.to_path_buf())
    } else {
        return Err(PanlabelError::VocLayoutInvalid {
            path: input.to_path_buf(),
            message: "expected a VOC dataset root containing Annotations/ or an Annotations/ directory itself"
                .to_string(),
        });
    };

    let images_dir = root.join("JPEGImages");
    let images_dir = images_dir.is_dir().then_some(images_dir);

    Ok(VocLayout {
        root,
        annotations_dir,
        images_dir,
    })
}

fn collect_xml_files(dir: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let path = entry.path();
        if path.is_file() && has_xml_extension(&path) {
            files.push(path);
        }
    }

    files.sort_by_cached_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| rel_string(dir, path))
    });

    let mut nested_xml = Vec::new();
    for entry in WalkDir::new(dir).follow_links(true).min_depth(2) {
        let entry = entry.map_err(|source| PanlabelError::VocLayoutInvalid {
            path: dir.to_path_buf(),
            message: format!("failed while traversing annotations directory: {source}"),
        })?;

        if entry.file_type().is_file() && has_xml_extension(entry.path()) {
            nested_xml.push(entry.path().to_path_buf());
        }
    }

    if !nested_xml.is_empty() {
        nested_xml.sort_by_cached_key(|path| rel_string(dir, path));
        let sample = rel_string(dir, &nested_xml[0]);
        eprintln!(
            "Warning: VOC reader scans Annotations/ flat (non-recursive); skipping {} nested .xml file(s), e.g. {}",
            nested_xml.len(),
            sample
        );
    }

    Ok(files)
}

fn parse_voc_xml(path: &Path) -> Result<ParsedVocAnnotation, PanlabelError> {
    let xml = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    parse_voc_xml_str(&xml, path)
}

fn parse_voc_xml_str(xml: &str, path: &Path) -> Result<ParsedVocAnnotation, PanlabelError> {
    let document =
        roxmltree::Document::parse(xml).map_err(|source| PanlabelError::VocXmlParse {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;

    let annotation = document.root_element();
    if annotation.tag_name().name() != "annotation" {
        return Err(PanlabelError::VocXmlParse {
            path: path.to_path_buf(),
            message: "missing <annotation> root element".to_string(),
        });
    }

    let filename = required_child_text(annotation, "filename", path, "<annotation>")?;

    let size = required_child_element(annotation, "size", path, "<annotation>")?;
    let width = parse_required_u32(size, "width", path, "<size>")?;
    let height = parse_required_u32(size, "height", path, "<size>")?;
    let depth = optional_child_text(size, "depth")
        .map(|raw| {
            raw.parse::<u32>().map_err(|_| PanlabelError::VocXmlParse {
                path: path.to_path_buf(),
                message: format!("invalid <depth> value '{raw}' in <size>; expected u32"),
            })
        })
        .transpose()?;

    let mut objects = Vec::new();
    for object in annotation
        .children()
        .filter(|node| node.is_element() && node.tag_name().name() == "object")
    {
        let name = required_child_text(object, "name", path, "<object>")?;
        let bndbox = required_child_element(object, "bndbox", path, "<object>")?;

        let xmin = parse_required_f64(bndbox, "xmin", path, "<bndbox>")?;
        let ymin = parse_required_f64(bndbox, "ymin", path, "<bndbox>")?;
        let xmax = parse_required_f64(bndbox, "xmax", path, "<bndbox>")?;
        let ymax = parse_required_f64(bndbox, "ymax", path, "<bndbox>")?;

        let mut attrs = BTreeMap::new();
        for key in ["pose", "truncated", "difficult", "occluded"] {
            if let Some(value) = optional_child_text(object, key) {
                attrs.insert(key.to_string(), value);
            }
        }

        objects.push(ParsedVocObject {
            name,
            xmin,
            ymin,
            xmax,
            ymax,
            attrs,
        });
    }

    Ok(ParsedVocAnnotation {
        filename,
        width,
        height,
        depth,
        objects,
    })
}

fn required_child_element<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &str,
    path: &Path,
    context: &str,
) -> Result<Node<'a, 'input>, PanlabelError> {
    child_element(node, tag).ok_or_else(|| PanlabelError::VocXmlParse {
        path: path.to_path_buf(),
        message: format!("missing <{tag}> in {context}"),
    })
}

fn required_child_text(
    node: Node<'_, '_>,
    tag: &str,
    path: &Path,
    context: &str,
) -> Result<String, PanlabelError> {
    optional_child_text(node, tag).ok_or_else(|| PanlabelError::VocXmlParse {
        path: path.to_path_buf(),
        message: format!("missing <{tag}> in {context}"),
    })
}

fn parse_required_u32(
    node: Node<'_, '_>,
    tag: &str,
    path: &Path,
    context: &str,
) -> Result<u32, PanlabelError> {
    let raw = required_child_text(node, tag, path, context)?;
    raw.parse::<u32>().map_err(|_| PanlabelError::VocXmlParse {
        path: path.to_path_buf(),
        message: format!("invalid <{tag}> value '{raw}' in {context}; expected u32"),
    })
}

fn parse_required_f64(
    node: Node<'_, '_>,
    tag: &str,
    path: &Path,
    context: &str,
) -> Result<f64, PanlabelError> {
    let raw = required_child_text(node, tag, path, context)?;
    raw.parse::<f64>().map_err(|_| PanlabelError::VocXmlParse {
        path: path.to_path_buf(),
        message: format!(
            "invalid <{tag}> value '{raw}' in {context}; expected floating-point number"
        ),
    })
}

fn child_element<'a, 'input>(node: Node<'a, 'input>, tag: &str) -> Option<Node<'a, 'input>> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == tag)
}

fn optional_child_text(node: Node<'_, '_>, tag: &str) -> Option<String> {
    child_element(node, tag)
        .and_then(|child| child.text())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn write_voc_xml(
    xml_path: &Path,
    image: &Image,
    annotations: &[&Annotation],
    category_name_by_id: &BTreeMap<CategoryId, String>,
    output_root: &Path,
) -> Result<(), PanlabelError> {
    let mut xml = String::new();

    writeln!(xml, "<?xml version=\"1.0\" encoding=\"utf-8\"?>").expect("write to string");
    writeln!(xml, "<annotation>").expect("write to string");
    writeln!(xml, "  <folder>JPEGImages</folder>").expect("write to string");
    writeln!(
        xml,
        "  <filename>{}</filename>",
        xml_escape(&image.file_name)
    )
    .expect("write to string");
    writeln!(xml, "  <size>").expect("write to string");
    writeln!(xml, "    <width>{}</width>", image.width).expect("write to string");
    writeln!(xml, "    <height>{}</height>", image.height).expect("write to string");

    if let Some(depth_raw) = image.attributes.get("depth") {
        if let Ok(depth) = depth_raw.trim().parse::<u32>() {
            writeln!(xml, "    <depth>{}</depth>", depth).expect("write to string");
        }
    }

    writeln!(xml, "  </size>").expect("write to string");

    for annotation in annotations {
        let category_name = category_name_by_id
            .get(&annotation.category_id)
            .ok_or_else(|| PanlabelError::VocWriteError {
                path: output_root.to_path_buf(),
                message: format!(
                    "annotation {} references missing category {}",
                    annotation.id.as_u64(),
                    annotation.category_id.as_u64()
                ),
            })?;

        writeln!(xml, "  <object>").expect("write to string");
        writeln!(xml, "    <name>{}</name>", xml_escape(category_name)).expect("write to string");

        if let Some(pose) = annotation
            .attributes
            .get("pose")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            writeln!(xml, "    <pose>{}</pose>", xml_escape(pose)).expect("write to string");
        }

        for key in ["truncated", "difficult", "occluded"] {
            if let Some(raw) = annotation.attributes.get(key) {
                if let Some(normalized) = normalize_bool_attr(raw) {
                    writeln!(xml, "    <{0}>{1}</{0}>", key, normalized).expect("write to string");
                }
            }
        }

        writeln!(xml, "    <bndbox>").expect("write to string");
        writeln!(xml, "      <xmin>{}</xmin>", annotation.bbox.xmin()).expect("write to string");
        writeln!(xml, "      <ymin>{}</ymin>", annotation.bbox.ymin()).expect("write to string");
        writeln!(xml, "      <xmax>{}</xmax>", annotation.bbox.xmax()).expect("write to string");
        writeln!(xml, "      <ymax>{}</ymax>", annotation.bbox.ymax()).expect("write to string");
        writeln!(xml, "    </bndbox>").expect("write to string");
        writeln!(xml, "  </object>").expect("write to string");
    }

    writeln!(xml, "</annotation>").expect("write to string");

    fs::write(xml_path, xml).map_err(PanlabelError::Io)
}

fn xml_escape(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn normalize_bool_attr(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Some("1"),
        "false" | "no" | "0" => Some("0"),
        _ => None,
    }
}

fn has_xml_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(VOC_XML_EXTENSION))
        .unwrap_or(false)
}

fn is_dir_named(path: &Path, dir_name: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(dir_name))
        .unwrap_or(false)
}

fn rel_string(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_layout_accepts_root_or_annotations_dir() {
        let temp = tempfile::tempdir().expect("create temp dir");
        fs::create_dir_all(temp.path().join("Annotations")).expect("create annotations dir");
        fs::create_dir_all(temp.path().join("JPEGImages")).expect("create images dir");

        let root_layout = discover_layout(temp.path()).expect("discover from root");
        assert_eq!(root_layout.root, temp.path());
        assert_eq!(root_layout.annotations_dir, temp.path().join("Annotations"));
        assert_eq!(root_layout.images_dir, Some(temp.path().join("JPEGImages")));

        let ann_layout =
            discover_layout(&temp.path().join("Annotations")).expect("discover from annotations");
        assert_eq!(ann_layout.root, temp.path());
        assert_eq!(ann_layout.annotations_dir, temp.path().join("Annotations"));
        assert_eq!(ann_layout.images_dir, Some(temp.path().join("JPEGImages")));
    }

    #[test]
    fn parse_voc_xml_extracts_bbox_and_attrs() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>img1.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <pose>Unspecified</pose>
    <truncated>1</truncated>
    <difficult>0</difficult>
    <occluded>yes</occluded>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>30</xmax>
      <ymax>40</ymax>
    </bndbox>
  </object>
</annotation>"#;

        let parsed = parse_voc_xml_str(xml, Path::new("sample.xml")).expect("parse xml");
        assert_eq!(parsed.filename, "img1.jpg");
        assert_eq!(parsed.width, 640);
        assert_eq!(parsed.height, 480);
        assert_eq!(parsed.depth, Some(3));
        assert_eq!(parsed.objects.len(), 1);
        assert_eq!(parsed.objects[0].name, "cat");
        assert_eq!(
            parsed.objects[0].attrs.get("pose"),
            Some(&"Unspecified".to_string())
        );
        assert_eq!(
            parsed.objects[0].attrs.get("occluded"),
            Some(&"yes".to_string())
        );
    }

    #[test]
    fn normalize_bool_attr_maps_expected_values() {
        assert_eq!(normalize_bool_attr("true"), Some("1"));
        assert_eq!(normalize_bool_attr("yes"), Some("1"));
        assert_eq!(normalize_bool_attr("1"), Some("1"));
        assert_eq!(normalize_bool_attr("false"), Some("0"));
        assert_eq!(normalize_bool_attr("no"), Some("0"));
        assert_eq!(normalize_bool_attr("0"), Some("0"));
        assert_eq!(normalize_bool_attr("maybe"), None);
    }
}
