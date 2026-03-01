//! Dataset semantic diffing.

mod report;

pub use report::{DiffAnnotationCounts, DiffCounts, DiffDetail, DiffReport, ModifiedAnnotation};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::ir::{Annotation, AnnotationId, CategoryId, Dataset, Image, ImageId};

/// Annotation matching strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchBy {
    /// Match annotations by annotation ID (within shared images).
    Id,
    /// Match annotations greedily by IoU within shared image + category.
    Iou,
}

/// Diff options.
#[derive(Clone, Debug)]
pub struct DiffOptions {
    pub match_by: MatchBy,
    pub iou_threshold: f64,
    pub detail: bool,
    pub max_items: usize,
    pub bbox_eps: f64,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            match_by: MatchBy::Id,
            iou_threshold: 0.5,
            detail: false,
            max_items: 20,
            bbox_eps: 1e-6,
        }
    }
}

/// Compute a semantic diff between two datasets.
pub fn diff_datasets(a: &Dataset, b: &Dataset, opts: &DiffOptions) -> DiffReport {
    let images_a = image_map_by_name(a);
    let images_b = image_map_by_name(b);

    let names_a: BTreeSet<String> = images_a.keys().cloned().collect();
    let names_b: BTreeSet<String> = images_b.keys().cloned().collect();

    let shared_image_names: Vec<String> = names_a.intersection(&names_b).cloned().collect();
    let images_only_in_a: Vec<String> = names_a.difference(&names_b).cloned().collect();
    let images_only_in_b: Vec<String> = names_b.difference(&names_a).cloned().collect();

    let categories_a: BTreeSet<String> = a.categories.iter().map(|c| c.name.clone()).collect();
    let categories_b: BTreeSet<String> = b.categories.iter().map(|c| c.name.clone()).collect();

    let mut report = DiffReport {
        images: DiffCounts {
            shared: shared_image_names.len(),
            only_in_a: images_only_in_a.len(),
            only_in_b: images_only_in_b.len(),
        },
        categories: DiffCounts {
            shared: categories_a.intersection(&categories_b).count(),
            only_in_a: categories_a.difference(&categories_b).count(),
            only_in_b: categories_b.difference(&categories_a).count(),
        },
        ..Default::default()
    };

    let anns_a = annotations_by_image(a);
    let anns_b = annotations_by_image(b);

    let cat_names_a: HashMap<CategoryId, String> = a
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();
    let cat_names_b: HashMap<CategoryId, String> = b
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let mut detail = if opts.detail {
        Some(DiffDetail {
            images_only_in_a: images_only_in_a.clone(),
            images_only_in_b: images_only_in_b.clone(),
            modified_annotations: Vec::new(),
            max_items: opts.max_items,
        })
    } else {
        None
    };

    for name in &shared_image_names {
        let image_a = images_a.get(name).expect("shared image exists in A map");
        let image_b = images_b.get(name).expect("shared image exists in B map");

        let list_a: Vec<&Annotation> = anns_a.get(&image_a.id).cloned().unwrap_or_default();
        let list_b: Vec<&Annotation> = anns_b.get(&image_b.id).cloned().unwrap_or_default();

        match opts.match_by {
            MatchBy::Id => diff_annotations_by_id(
                name,
                &list_a,
                &list_b,
                &cat_names_a,
                &cat_names_b,
                &mut report.annotations,
                detail.as_mut(),
                opts,
            ),
            MatchBy::Iou => diff_annotations_by_iou(
                &list_a,
                &list_b,
                &cat_names_a,
                &cat_names_b,
                &mut report.annotations,
                opts,
            ),
        }
    }

    for name in &images_only_in_a {
        if let Some(image) = images_a.get(name) {
            report.annotations.only_in_a += anns_a.get(&image.id).map(|v| v.len()).unwrap_or(0);
        }
    }
    for name in &images_only_in_b {
        if let Some(image) = images_b.get(name) {
            report.annotations.only_in_b += anns_b.get(&image.id).map(|v| v.len()).unwrap_or(0);
        }
    }

    report.detail = detail;
    report
}

fn image_map_by_name(dataset: &Dataset) -> BTreeMap<String, &Image> {
    let mut map = BTreeMap::new();
    for image in &dataset.images {
        map.insert(image.file_name.clone(), image);
    }
    map
}

fn annotations_by_image(dataset: &Dataset) -> HashMap<ImageId, Vec<&Annotation>> {
    let mut map: HashMap<ImageId, Vec<&Annotation>> = HashMap::new();
    for ann in &dataset.annotations {
        map.entry(ann.image_id).or_default().push(ann);
    }
    map
}

fn category_name(category_names: &HashMap<CategoryId, String>, category_id: CategoryId) -> String {
    category_names
        .get(&category_id)
        .cloned()
        .unwrap_or_else(|| format!("<missing cat {}>", category_id))
}

fn bbox_eq_eps(
    a: &crate::ir::BBoxXYXY<crate::ir::Pixel>,
    b: &crate::ir::BBoxXYXY<crate::ir::Pixel>,
    eps: f64,
) -> bool {
    (a.xmin() - b.xmin()).abs() <= eps
        && (a.ymin() - b.ymin()).abs() <= eps
        && (a.xmax() - b.xmax()).abs() <= eps
        && (a.ymax() - b.ymax()).abs() <= eps
}

#[allow(clippy::too_many_arguments)]
fn diff_annotations_by_id(
    file_name: &str,
    anns_a: &[&Annotation],
    anns_b: &[&Annotation],
    cat_names_a: &HashMap<CategoryId, String>,
    cat_names_b: &HashMap<CategoryId, String>,
    counts: &mut DiffAnnotationCounts,
    detail: Option<&mut DiffDetail>,
    opts: &DiffOptions,
) {
    let map_a: BTreeMap<AnnotationId, &Annotation> =
        anns_a.iter().map(|ann| (ann.id, *ann)).collect();
    let map_b: BTreeMap<AnnotationId, &Annotation> =
        anns_b.iter().map(|ann| (ann.id, *ann)).collect();

    let ids_a: BTreeSet<AnnotationId> = map_a.keys().copied().collect();
    let ids_b: BTreeSet<AnnotationId> = map_b.keys().copied().collect();

    let shared_ids: Vec<AnnotationId> = ids_a.intersection(&ids_b).copied().collect();

    let mut detail_ref = detail;

    for ann_id in shared_ids {
        counts.shared += 1;

        let ann_a = map_a
            .get(&ann_id)
            .expect("shared annotation id exists in A");
        let ann_b = map_b
            .get(&ann_id)
            .expect("shared annotation id exists in B");

        let cat_a = category_name(cat_names_a, ann_a.category_id);
        let cat_b = category_name(cat_names_b, ann_b.category_id);

        let mut reasons: Vec<&str> = Vec::new();
        if cat_a != cat_b {
            reasons.push("category changed");
        }
        if !bbox_eq_eps(&ann_a.bbox, &ann_b.bbox, opts.bbox_eps) {
            reasons.push("bbox changed");
        }

        if !reasons.is_empty() {
            counts.modified += 1;
            if let Some(detail) = detail_ref.as_deref_mut() {
                if detail.modified_annotations.len() < opts.max_items {
                    detail.modified_annotations.push(ModifiedAnnotation {
                        file_name: file_name.to_string(),
                        annotation_id: ann_id.as_u64(),
                        reason: reasons.join(", "),
                    });
                }
            }
        }
    }

    counts.only_in_a += ids_a.difference(&ids_b).count();
    counts.only_in_b += ids_b.difference(&ids_a).count();
}

fn diff_annotations_by_iou(
    anns_a: &[&Annotation],
    anns_b: &[&Annotation],
    cat_names_a: &HashMap<CategoryId, String>,
    cat_names_b: &HashMap<CategoryId, String>,
    counts: &mut DiffAnnotationCounts,
    opts: &DiffOptions,
) {
    let mut grouped_a: HashMap<String, Vec<&Annotation>> = HashMap::new();
    let mut grouped_b: HashMap<String, Vec<&Annotation>> = HashMap::new();

    for ann in anns_a {
        let category = category_name(cat_names_a, ann.category_id);
        grouped_a.entry(category).or_default().push(*ann);
    }
    for ann in anns_b {
        let category = category_name(cat_names_b, ann.category_id);
        grouped_b.entry(category).or_default().push(*ann);
    }

    let categories: HashSet<String> = grouped_a.keys().chain(grouped_b.keys()).cloned().collect();

    for category in categories {
        let list_a = grouped_a.remove(&category).unwrap_or_default();
        let list_b = grouped_b.remove(&category).unwrap_or_default();

        let mut used_b = vec![false; list_b.len()];

        for ann_a in &list_a {
            let mut best_idx: Option<usize> = None;
            let mut best_iou = f64::MIN;

            for (idx, ann_b) in list_b.iter().enumerate() {
                if used_b[idx] {
                    continue;
                }

                let iou = ann_a.bbox.iou(&ann_b.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                if best_iou >= opts.iou_threshold {
                    used_b[idx] = true;
                    counts.shared += 1;
                } else {
                    counts.only_in_a += 1;
                }
            } else {
                counts.only_in_a += 1;
            }
        }

        counts.only_in_b += used_b.iter().filter(|matched| !**matched).count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Annotation, BBoxXYXY, Category, Image, Pixel};

    fn dataset_for_diff() -> Dataset {
        Dataset {
            images: vec![Image::new(1u64, "img.jpg", 100, 100)],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 10.0, 20.0, 20.0),
            )],
            ..Default::default()
        }
    }

    #[test]
    fn id_mode_identical_has_no_changes() {
        let a = dataset_for_diff();
        let b = dataset_for_diff();

        let report = diff_datasets(&a, &b, &DiffOptions::default());
        assert_eq!(report.images.shared, 1);
        assert_eq!(report.annotations.only_in_a, 0);
        assert_eq!(report.annotations.only_in_b, 0);
        assert_eq!(report.annotations.modified, 0);
        assert_eq!(report.annotations.shared, 1);
    }

    #[test]
    fn iou_mode_matches_different_ids() {
        let a = dataset_for_diff();
        let mut b = dataset_for_diff();
        b.annotations[0].id = 999u64.into();

        let opts = DiffOptions {
            match_by: MatchBy::Iou,
            ..Default::default()
        };

        let report = diff_datasets(&a, &b, &opts);
        assert_eq!(report.annotations.shared, 1);
        assert_eq!(report.annotations.only_in_a, 0);
        assert_eq!(report.annotations.only_in_b, 0);
    }
}
