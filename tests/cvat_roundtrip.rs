//! Integration tests for CVAT XML format support.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use panlabel::ir::io_cvat_xml::{read_cvat_xml, write_cvat_xml};

fn create_sample_cvat_export(root: &Path) {
    fs::create_dir_all(root).expect("create root");

    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>cat</name><type>bbox</type></label>
        <label><name>dog</name><type>bbox</type></label>
        <label><name>unused</name><type>bbox</type></label>
      </labels>
    </task>
  </meta>
  <image id="10" name="img_b.jpg" width="120" height="80">
    <box label="dog" occluded="1" xtl="10.0" ytl="12.0" xbr="60.0" ybr="70.0" z_order="0" source="manual">
      <attribute name="truncated">no</attribute>
    </box>
  </image>
  <image id="2" name="img_a.jpg" width="100" height="50">
    <box label="cat" occluded="0" xtl="1" ytl="2" xbr="30" ybr="40" z_order="2" source="manual"/>
  </image>
  <image id="3" name="img_c.jpg" width="64" height="64"></image>
</annotations>
"#;

    fs::write(root.join("annotations.xml"), xml).expect("write annotations.xml");
}

#[test]
fn read_cvat_from_dir_succeeds_and_assigns_deterministic_ids() {
    let temp = tempfile::tempdir().expect("tempdir");
    create_sample_cvat_export(temp.path());

    let dataset = read_cvat_xml(temp.path()).expect("read cvat dir");
    assert_eq!(dataset.images.len(), 3);
    assert_eq!(dataset.annotations.len(), 2);

    assert_eq!(dataset.images[0].file_name, "img_a.jpg");
    assert_eq!(dataset.images[0].id.as_u64(), 1);
    assert_eq!(dataset.images[1].file_name, "img_b.jpg");
    assert_eq!(dataset.images[1].id.as_u64(), 2);
    assert_eq!(dataset.images[2].file_name, "img_c.jpg");
    assert_eq!(dataset.images[2].id.as_u64(), 3);

    assert_eq!(
        dataset.images[1].attributes.get("cvat_image_id"),
        Some(&"10".to_string())
    );
}

#[test]
fn cvat_write_then_read_roundtrip_semantic() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input_root = temp.path().join("input");
    let output_root = temp.path().join("output");

    create_sample_cvat_export(&input_root);

    let input_dataset = read_cvat_xml(&input_root).expect("read input cvat");
    write_cvat_xml(&output_root, &input_dataset).expect("write cvat");

    assert!(output_root.join("annotations.xml").is_file());
    let restored = read_cvat_xml(&output_root).expect("read restored cvat");

    assert_eq!(restored.images.len(), input_dataset.images.len());
    assert_eq!(restored.annotations.len(), input_dataset.annotations.len());

    let restored_category_names: BTreeSet<_> =
        restored.categories.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        restored_category_names,
        ["cat".to_string(), "dog".to_string()].into_iter().collect()
    );

    assert!(restored.annotations.iter().any(|ann| {
        ann.attributes.get("occluded") == Some(&"1".to_string())
            && ann.attributes.get("cvat_attr_truncated") == Some(&"no".to_string())
    }));
}
