//! CVAT XML reader and writer.
//!
//! This adapter supports CVAT "for images" task-export XML:
//! - single `annotations.xml` file
//! - root `<annotations>` containing `<image>` entries
//! - `<box>` elements only (object-detection bboxes)

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const CVAT_XML_FILE_NAME: &str = "annotations.xml";

/// Read a CVAT XML file or directory containing `annotations.xml` into IR.
pub fn read_cvat_xml(path: &Path) -> Result<Dataset, PanlabelError> {
    let resolved = resolve_cvat_xml_path(path)?;
    let xml = fs::read_to_string(&resolved).map_err(PanlabelError::Io)?;
    parse_cvat_xml_str(&xml, &resolved)
}

/// Write an IR dataset as CVAT XML.
///
/// - If `path` ends with `.xml`, writes directly to that file.
/// - Otherwise, treats `path` as a directory and writes `annotations.xml` inside it.
pub fn write_cvat_xml(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let (_out_dir, out_file) = resolve_cvat_output_path(path);
    if let Some(parent) = out_file.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    let xml = build_cvat_xml(dataset, &out_file)?;
    fs::write(&out_file, xml).map_err(PanlabelError::Io)
}

/// Parse CVAT XML from a string.
pub fn from_cvat_xml_str(xml: &str) -> Result<Dataset, PanlabelError> {
    parse_cvat_xml_str(xml, Path::new("<string>"))
}

/// Parse CVAT XML from bytes (must be valid UTF-8).
pub fn from_cvat_xml_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let xml = std::str::from_utf8(bytes).map_err(|source| PanlabelError::CvatXmlParse {
        path: PathBuf::from("<bytes>"),
        message: format!("input is not valid UTF-8: {source}"),
    })?;
    parse_cvat_xml_str(xml, Path::new("<bytes>"))
}

/// Serialize an IR dataset to a CVAT XML string.
pub fn to_cvat_xml_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    build_cvat_xml(dataset, Path::new("<string>"))
}

#[derive(Debug)]
struct ParsedBox {
    label: String,
    bbox: BBoxXYXY<Pixel>,
    occluded: bool,
    z_order: Option<i32>,
    source: Option<String>,
    attributes: BTreeMap<String, String>,
}

#[derive(Debug)]
struct ParsedImage {
    name: String,
    width: u32,
    height: u32,
    cvat_id: Option<u64>,
    boxes: Vec<ParsedBox>,
}

#[derive(Clone, Debug)]
struct MetaLabels {
    all: BTreeSet<String>,
    bbox_or_unknown: BTreeSet<String>,
}

fn parse_cvat_xml_str(xml: &str, path: &Path) -> Result<Dataset, PanlabelError> {
    let document = Document::parse(xml).map_err(|source| PanlabelError::CvatXmlParse {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;

    let root = document.root_element();
    if root.tag_name().name() != "annotations" {
        return Err(PanlabelError::CvatXmlParse {
            path: path.to_path_buf(),
            message: "missing <annotations> root element".to_string(),
        });
    }

    let meta_labels = extract_meta_labels(root, path)?;

    let mut seen_image_names = BTreeSet::new();
    let mut parsed_images = Vec::new();
    let mut referenced_labels: BTreeSet<String> = BTreeSet::new();

    for image_node in root
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "image")
    {
        let parsed = parse_image_element(image_node, path, meta_labels.as_ref())?;
        if !seen_image_names.insert(parsed.name.clone()) {
            return Err(PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "duplicate image name: '{}' appears in multiple <image> elements",
                    parsed.name
                ),
            });
        }

        for b in &parsed.boxes {
            referenced_labels.insert(b.label.clone());
        }

        parsed_images.push(parsed);
    }

    let category_names: BTreeSet<String> = match &meta_labels {
        Some(meta) => {
            let mut out = meta.bbox_or_unknown.clone();
            out.extend(referenced_labels);
            out
        }
        None => referenced_labels,
    };

    let categories: Vec<Category> = category_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name))
        .collect();

    let category_id_by_name: BTreeMap<String, CategoryId> =
        categories.iter().map(|c| (c.name.clone(), c.id)).collect();

    parsed_images.sort_by(|a, b| a.name.cmp(&b.name));

    let mut images = Vec::with_capacity(parsed_images.len());
    let mut image_id_by_name: BTreeMap<String, ImageId> = BTreeMap::new();

    for (idx, parsed) in parsed_images.iter().enumerate() {
        let mut image = Image::new(
            (idx + 1) as u64,
            parsed.name.clone(),
            parsed.width,
            parsed.height,
        );
        if let Some(cvat_id) = parsed.cvat_id {
            image
                .attributes
                .insert("cvat_image_id".to_string(), cvat_id.to_string());
        }
        image_id_by_name.insert(parsed.name.clone(), image.id);
        images.push(image);
    }

    let mut annotations = Vec::new();
    let mut next_ann_id: u64 = 1;

    for parsed_img in parsed_images {
        let image_id = image_id_by_name
            .get(&parsed_img.name)
            .copied()
            .ok_or_else(|| PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "internal error: missing image mapping for '{}'",
                    parsed_img.name
                ),
            })?;

        for parsed_box in parsed_img.boxes {
            let category_id = category_id_by_name
                .get(&parsed_box.label)
                .copied()
                .ok_or_else(|| PanlabelError::CvatXmlParse {
                    path: path.to_path_buf(),
                    message: format!(
                        "internal error: missing category mapping for '{}'",
                        parsed_box.label
                    ),
                })?;

            let mut ann = Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                category_id,
                parsed_box.bbox,
            );

            let mut attrs = parsed_box.attributes;
            if parsed_box.occluded {
                attrs.insert("occluded".to_string(), "1".to_string());
            }
            if let Some(z) = parsed_box.z_order.filter(|z| *z != 0) {
                attrs.insert("z_order".to_string(), z.to_string());
            }
            if let Some(source) = parsed_box.source.as_ref().filter(|s| !s.trim().is_empty()) {
                attrs.insert("source".to_string(), source.trim().to_string());
            }

            ann.attributes = attrs;
            annotations.push(ann);
            next_ann_id += 1;
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

fn parse_image_element(
    node: Node<'_, '_>,
    path: &Path,
    meta: Option<&MetaLabels>,
) -> Result<ParsedImage, PanlabelError> {
    let name = required_attr(node, "name", path, "<image>")?.to_string();
    let width = parse_required_u32_attr(node, "width", path, "<image>")?;
    let height = parse_required_u32_attr(node, "height", path, "<image>")?;
    let cvat_id = node
        .attribute("id")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|raw| {
            raw.parse::<u64>().map_err(|_| PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!("invalid <image id> value '{raw}'; expected u64"),
            })
        })
        .transpose()?;

    let mut boxes = Vec::new();
    for child in node.children().filter(|n| n.is_element()) {
        let tag = child.tag_name().name();
        if tag != "box" {
            return Err(PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "image '{}' contains unsupported annotation type <{tag}>; only <box> is supported",
                    name
                ),
            });
        }

        boxes.push(parse_box_element(child, path, &name, meta)?);
    }

    Ok(ParsedImage {
        name,
        width,
        height,
        cvat_id,
        boxes,
    })
}

fn parse_box_element(
    node: Node<'_, '_>,
    path: &Path,
    image_name: &str,
    meta: Option<&MetaLabels>,
) -> Result<ParsedBox, PanlabelError> {
    let label = required_attr(node, "label", path, "<box>")?.to_string();

    if let Some(meta) = meta {
        if !meta.all.contains(&label) {
            return Err(PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "<box> in image '{}' references unknown label '{}' not in <meta><task><labels>",
                    image_name, label
                ),
            });
        }
    }

    let xtl = parse_required_f64_attr(node, "xtl", path, "<box>", image_name)?;
    let ytl = parse_required_f64_attr(node, "ytl", path, "<box>", image_name)?;
    let xbr = parse_required_f64_attr(node, "xbr", path, "<box>", image_name)?;
    let ybr = parse_required_f64_attr(node, "ybr", path, "<box>", image_name)?;

    let occluded = node
        .attribute("occluded")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|raw| match raw {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "<box> in image '{}' has invalid occluded='{raw}'; expected '0' or '1'",
                    image_name
                ),
            }),
        })
        .transpose()?
        .unwrap_or(false);

    let z_order = node
        .attribute("z_order")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|raw| {
            raw.parse::<i32>().map_err(|_| PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!(
                    "<box> in image '{}' has invalid z_order='{raw}'; expected i32",
                    image_name
                ),
            })
        })
        .transpose()?;

    let source = node
        .attribute("source")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToOwned::to_owned);

    let mut attributes = BTreeMap::new();
    for attr_node in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "attribute")
    {
        let name = required_attr(attr_node, "name", path, "<attribute>")?.trim();
        if name.is_empty() {
            return Err(PanlabelError::CvatXmlParse {
                path: path.to_path_buf(),
                message: format!("<attribute> in image '{}' has empty name", image_name),
            });
        }

        let value = attr_node.text().map(str::trim).unwrap_or("").to_string();

        attributes.insert(format!("cvat_attr_{name}"), value);
    }

    Ok(ParsedBox {
        label,
        bbox: BBoxXYXY::<Pixel>::from_xyxy(xtl, ytl, xbr, ybr),
        occluded,
        z_order,
        source,
        attributes,
    })
}

fn extract_meta_labels(
    root: Node<'_, '_>,
    path: &Path,
) -> Result<Option<MetaLabels>, PanlabelError> {
    let Some(meta) = child_element(root, "meta") else {
        return Ok(None);
    };

    if child_element(meta, "project").is_some() {
        return Err(PanlabelError::CvatXmlParse {
            path: path.to_path_buf(),
            message: "project export not supported; export a task instead".to_string(),
        });
    }

    let Some(task) = child_element(meta, "task") else {
        return Ok(None);
    };
    let Some(labels) = child_element(task, "labels") else {
        return Ok(None);
    };

    let mut all = BTreeSet::new();
    let mut bbox_or_unknown = BTreeSet::new();

    for label_node in labels
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "label")
    {
        let name = required_child_text(label_node, "name", path, "<label>")?;
        all.insert(name.clone());

        let typ = optional_child_text(label_node, "type").unwrap_or_else(|| "bbox".to_string());
        if typ.trim().eq_ignore_ascii_case("bbox") {
            bbox_or_unknown.insert(name);
        }
    }

    Ok(Some(MetaLabels {
        all,
        bbox_or_unknown,
    }))
}

fn build_cvat_xml(dataset: &Dataset, output_path: &Path) -> Result<String, PanlabelError> {
    let image_by_id: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_by_id: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_by_id.contains_key(&ann.image_id) {
            return Err(PanlabelError::CvatWriteError {
                path: output_path.to_path_buf(),
                message: format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            });
        }

        if !category_by_id.contains_key(&ann.category_id) {
            return Err(PanlabelError::CvatWriteError {
                path: output_path.to_path_buf(),
                message: format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            });
        }

        annotations_by_image
            .entry(ann.image_id)
            .or_default()
            .push(ann);
    }

    for anns in annotations_by_image.values_mut() {
        anns.sort_by_key(|ann| ann.id);
    }

    let used_category_ids: HashSet<CategoryId> =
        dataset.annotations.iter().map(|a| a.category_id).collect();
    let mut categories: Vec<&Category> = dataset
        .categories
        .iter()
        .filter(|cat| used_category_ids.contains(&cat.id))
        .collect();
    categories.sort_by(|a, b| a.name.cmp(&b.name));

    let category_name_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut xml = String::new();
    writeln!(xml, "<?xml version=\"1.0\" encoding=\"utf-8\"?>").expect("write to string");
    writeln!(xml, "<annotations>").expect("write to string");
    writeln!(xml, "  <version>1.1</version>").expect("write to string");
    writeln!(xml, "  <meta>").expect("write to string");
    writeln!(xml, "    <task>").expect("write to string");
    writeln!(xml, "      <name>panlabel export</name>").expect("write to string");
    writeln!(xml, "      <size>{}</size>", images_sorted.len()).expect("write to string");
    writeln!(xml, "      <mode>annotation</mode>").expect("write to string");
    writeln!(xml, "      <labels>").expect("write to string");
    for cat in categories {
        writeln!(xml, "        <label>").expect("write to string");
        writeln!(xml, "          <name>{}</name>", xml_escape(&cat.name)).expect("write to string");
        writeln!(xml, "        </label>").expect("write to string");
    }
    writeln!(xml, "      </labels>").expect("write to string");
    writeln!(xml, "    </task>").expect("write to string");
    writeln!(xml, "  </meta>").expect("write to string");

    for (idx, image) in images_sorted.into_iter().enumerate() {
        let image_idx = idx as u64;
        writeln!(
            xml,
            "  <image id=\"{}\" name=\"{}\" width=\"{}\" height=\"{}\">",
            image_idx,
            xml_escape(&image.file_name),
            image.width,
            image.height
        )
        .expect("write to string");

        let anns = annotations_by_image.remove(&image.id).unwrap_or_default();
        for ann in anns {
            let label = category_name_by_id.get(&ann.category_id).ok_or_else(|| {
                PanlabelError::CvatWriteError {
                    path: output_path.to_path_buf(),
                    message: format!(
                        "internal error: missing category {} while writing",
                        ann.category_id.as_u64()
                    ),
                }
            })?;

            let occluded = ann
                .attributes
                .get("occluded")
                .and_then(|value| normalize_bool_attr(value))
                .unwrap_or("0");

            let z_order = ann
                .attributes
                .get("z_order")
                .and_then(|v| v.trim().parse::<i32>().ok())
                .unwrap_or(0);

            let source = ann
                .attributes
                .get("source")
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .unwrap_or("manual");

            writeln!(
                xml,
                "    <box label=\"{}\" occluded=\"{}\" xtl=\"{}\" ytl=\"{}\" xbr=\"{}\" ybr=\"{}\" z_order=\"{}\" source=\"{}\">",
                xml_escape(label),
                occluded,
                ann.bbox.xmin(),
                ann.bbox.ymin(),
                ann.bbox.xmax(),
                ann.bbox.ymax(),
                z_order,
                xml_escape(source),
            )
            .expect("write to string");

            for (key, value) in &ann.attributes {
                let Some(raw_name) = key.strip_prefix("cvat_attr_") else {
                    continue;
                };
                let raw_name = raw_name.trim();
                if raw_name.is_empty() {
                    continue;
                }

                writeln!(
                    xml,
                    "      <attribute name=\"{}\">{}</attribute>",
                    xml_escape(raw_name),
                    xml_escape(value)
                )
                .expect("write to string");
            }

            writeln!(xml, "    </box>").expect("write to string");
        }

        writeln!(xml, "  </image>").expect("write to string");
    }

    writeln!(xml, "</annotations>").expect("write to string");
    Ok(xml)
}

fn resolve_cvat_xml_path(path: &Path) -> Result<PathBuf, PanlabelError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    if !path.is_dir() {
        return Err(PanlabelError::CvatLayoutInvalid {
            path: path.to_path_buf(),
            message: "input must be a file or directory".to_string(),
        });
    }

    let candidate = path.join(CVAT_XML_FILE_NAME);
    if candidate.is_file() {
        return Ok(candidate);
    }

    Err(PanlabelError::CvatLayoutInvalid {
        path: path.to_path_buf(),
        message: format!("expected '{CVAT_XML_FILE_NAME}' at directory root"),
    })
}

fn resolve_cvat_output_path(path: &Path) -> (PathBuf, PathBuf) {
    let is_xml_file = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("xml"))
        .unwrap_or(false);

    if is_xml_file {
        let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        return (parent, path.to_path_buf());
    }

    (path.to_path_buf(), path.join(CVAT_XML_FILE_NAME))
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

fn required_child_text(
    node: Node<'_, '_>,
    tag: &str,
    path: &Path,
    context: &str,
) -> Result<String, PanlabelError> {
    optional_child_text(node, tag).ok_or_else(|| PanlabelError::CvatXmlParse {
        path: path.to_path_buf(),
        message: format!("missing <{tag}> in {context}"),
    })
}

fn required_attr<'a>(
    node: Node<'a, '_>,
    attr: &str,
    path: &Path,
    context: &str,
) -> Result<&'a str, PanlabelError> {
    node.attribute(attr)
        .ok_or_else(|| PanlabelError::CvatXmlParse {
            path: path.to_path_buf(),
            message: format!("missing '{attr}' attribute in {context}"),
        })
}

fn parse_required_u32_attr(
    node: Node<'_, '_>,
    attr: &str,
    path: &Path,
    context: &str,
) -> Result<u32, PanlabelError> {
    let raw = required_attr(node, attr, path, context)?;
    raw.trim()
        .parse::<u32>()
        .map_err(|_| PanlabelError::CvatXmlParse {
            path: path.to_path_buf(),
            message: format!("invalid '{attr}' value '{raw}' in {context}; expected u32"),
        })
}

fn parse_required_f64_attr(
    node: Node<'_, '_>,
    attr: &str,
    path: &Path,
    context: &str,
    image_name: &str,
) -> Result<f64, PanlabelError> {
    let raw = required_attr(node, attr, path, context)?;
    raw.trim()
        .parse::<f64>()
        .map_err(|_| PanlabelError::CvatXmlParse {
            path: path.to_path_buf(),
            message: format!(
                "<box> in image '{}' has invalid {attr}='{raw}'; expected floating-point number",
                image_name
            ),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_invalid_root() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?><annotation></annotation>"#;
        let err = from_cvat_xml_str(xml).unwrap_err();
        match err {
            PanlabelError::CvatXmlParse { message, .. } => {
                assert!(message.contains("<annotations>"))
            }
            other => panic!("expected CvatXmlParse, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_polygon() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="img.jpg" width="10" height="10">
    <polygon label="cat" points="1,1;2,2"/>
  </image>
</annotations>"#;
        let err = from_cvat_xml_str(xml).unwrap_err();
        match err {
            PanlabelError::CvatXmlParse { message, .. } => {
                assert!(message.contains("unsupported annotation type"))
            }
            other => panic!("expected CvatXmlParse, got {other:?}"),
        }
    }

    #[test]
    fn write_then_read_roundtrip_semantic() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <labels>
        <label><name>cat</name><type>bbox</type></label>
        <label><name>dog</name><type>bbox</type></label>
        <label><name>unused</name><type>bbox</type></label>
      </labels>
    </task>
  </meta>
  <image id="5" name="b.jpg" width="20" height="10">
    <box label="cat" occluded="1" xtl="1.0" ytl="2.0" xbr="3.0" ybr="4.0" z_order="2" source="manual">
      <attribute name="truncated">no</attribute>
    </box>
  </image>
  <image id="2" name="a.jpg" width="20" height="10"></image>
</annotations>"#;

        let dataset = from_cvat_xml_str(xml).expect("parse");
        assert!(dataset.categories.iter().any(|c| c.name == "unused"));

        let out = to_cvat_xml_string(&dataset).expect("write");
        let restored = from_cvat_xml_str(&out).expect("parse restored");

        assert_eq!(restored.images.len(), 2);
        assert_eq!(restored.annotations.len(), 1);
        assert!(!restored.categories.iter().any(|c| c.name == "unused"));

        let ann = &restored.annotations[0];
        assert_eq!(ann.attributes.get("occluded"), Some(&"1".to_string()));
        assert_eq!(ann.attributes.get("z_order"), Some(&"2".to_string()));
        assert_eq!(ann.attributes.get("source"), Some(&"manual".to_string()));
        assert_eq!(
            ann.attributes.get("cvat_attr_truncated"),
            Some(&"no".to_string())
        );
    }
}
