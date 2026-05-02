use std::collections::BTreeMap;

use super::io_bbox_adapters_common::annotations_by_image;
use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, CategoryId, ImageId};

pub(crate) struct WriterDatasetView<'a> {
    image_by_id: BTreeMap<ImageId, &'a Image>,
    category_by_id: BTreeMap<CategoryId, &'a Category>,
    // Used by file-oriented writers once VOC/KITTI migrate in a later slice.
    #[allow(dead_code)]
    annotations_by_image: BTreeMap<ImageId, Vec<&'a Annotation>>,
    annotations_by_id: Vec<&'a Annotation>,
    annotations_in_dataset_order: Vec<&'a Annotation>,
    // Used by file-oriented writers once VOC/KITTI migrate in a later slice.
    #[allow(dead_code)]
    images_by_file_name: Vec<&'a Image>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnnotationValidationOrder {
    DatasetOrder,
    AnnotationIdOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MissingDatasetReference {
    Image {
        annotation_id: AnnotationId,
        image_id: ImageId,
    },
    Category {
        annotation_id: AnnotationId,
        category_id: CategoryId,
    },
}

impl<'a> WriterDatasetView<'a> {
    pub(crate) fn new(dataset: &'a Dataset) -> Self {
        let image_by_id = dataset.images.iter().map(|img| (img.id, img)).collect();
        let category_by_id = dataset.categories.iter().map(|cat| (cat.id, cat)).collect();
        let annotations_by_image = annotations_by_image(dataset);

        let annotations_in_dataset_order: Vec<&Annotation> = dataset.annotations.iter().collect();

        let mut annotations_by_id = annotations_in_dataset_order.clone();
        annotations_by_id.sort_by_key(|ann| ann.id);

        let mut images_by_file_name: Vec<&Image> = dataset.images.iter().collect();
        images_by_file_name.sort_by(|left, right| left.file_name.cmp(&right.file_name));

        Self {
            image_by_id,
            category_by_id,
            annotations_by_image,
            annotations_by_id,
            annotations_in_dataset_order,
            images_by_file_name,
        }
    }

    pub(crate) fn validate_references(
        &self,
        order: AnnotationValidationOrder,
    ) -> Result<(), MissingDatasetReference> {
        let annotations = match order {
            AnnotationValidationOrder::DatasetOrder => &self.annotations_in_dataset_order,
            AnnotationValidationOrder::AnnotationIdOrder => &self.annotations_by_id,
        };

        for ann in annotations {
            if !self.image_by_id.contains_key(&ann.image_id) {
                return Err(MissingDatasetReference::Image {
                    annotation_id: ann.id,
                    image_id: ann.image_id,
                });
            }
            if !self.category_by_id.contains_key(&ann.category_id) {
                return Err(MissingDatasetReference::Category {
                    annotation_id: ann.id,
                    category_id: ann.category_id,
                });
            }
        }

        Ok(())
    }

    pub(crate) fn image(&self, id: ImageId) -> Option<&'a Image> {
        self.image_by_id.get(&id).copied()
    }

    fn category(&self, id: CategoryId) -> Option<&'a Category> {
        self.category_by_id.get(&id).copied()
    }

    pub(crate) fn category_name(&self, id: CategoryId) -> Option<&'a str> {
        self.category(id).map(|category| category.name.as_str())
    }

    pub(crate) fn annotations_sorted_by_id(&self) -> impl Iterator<Item = &'a Annotation> + '_ {
        self.annotations_by_id.iter().copied()
    }

    #[allow(dead_code)]
    pub(crate) fn images_sorted_by_file_name(&self) -> impl Iterator<Item = &'a Image> + '_ {
        self.images_by_file_name.iter().copied()
    }

    #[allow(dead_code)]
    pub(crate) fn annotations_for_image_sorted_by_id(
        &self,
        image_id: ImageId,
    ) -> impl Iterator<Item = &'a Annotation> + '_ {
        self.annotations_by_image
            .get(&image_id)
            .into_iter()
            .flatten()
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BBoxXYXY, Pixel};

    fn ann(id: u64, image_id: u64, category_id: u64) -> Annotation {
        Annotation::new(
            id,
            image_id,
            category_id,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 1.0, 1.0),
        )
    }

    fn sample_dataset() -> Dataset {
        Dataset {
            images: vec![
                Image::new(2u64, "b.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![ann(3, 2, 1), ann(1, 1, 2), ann(2, 1, 1)],
            ..Default::default()
        }
    }

    #[test]
    fn annotations_are_sorted_by_id_without_changing_dataset_order() {
        let dataset = sample_dataset();
        let view = WriterDatasetView::new(&dataset);

        let ids: Vec<u64> = view
            .annotations_sorted_by_id()
            .map(|ann| ann.id.as_u64())
            .collect();

        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(dataset.annotations[0].id.as_u64(), 3);
    }

    #[test]
    fn images_are_sorted_by_file_name() {
        let dataset = sample_dataset();
        let view = WriterDatasetView::new(&dataset);

        let names: Vec<&str> = view
            .images_sorted_by_file_name()
            .map(|image| image.file_name.as_str())
            .collect();

        assert_eq!(names, vec!["a.jpg", "b.jpg"]);
    }

    #[test]
    fn annotations_for_image_are_grouped_and_sorted_by_id() {
        let dataset = sample_dataset();
        let view = WriterDatasetView::new(&dataset);

        let ids: Vec<u64> = view
            .annotations_for_image_sorted_by_id(ImageId::new(1))
            .map(|ann| ann.id.as_u64())
            .collect();

        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn validation_reports_missing_image_before_missing_category_for_same_annotation() {
        let dataset = Dataset {
            images: vec![],
            categories: vec![],
            annotations: vec![ann(1, 999, 888)],
            ..Default::default()
        };
        let view = WriterDatasetView::new(&dataset);

        let err = view
            .validate_references(AnnotationValidationOrder::DatasetOrder)
            .expect_err("expected missing reference");

        assert_eq!(
            err,
            MissingDatasetReference::Image {
                annotation_id: AnnotationId::new(1),
                image_id: ImageId::new(999),
            }
        );
    }

    #[test]
    fn validation_order_can_follow_dataset_or_annotation_id_order() {
        let dataset = Dataset {
            images: vec![],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![ann(20, 20, 1), ann(10, 10, 1)],
            ..Default::default()
        };
        let view = WriterDatasetView::new(&dataset);

        let dataset_order_err = view
            .validate_references(AnnotationValidationOrder::DatasetOrder)
            .expect_err("expected dataset-order missing reference");
        let id_order_err = view
            .validate_references(AnnotationValidationOrder::AnnotationIdOrder)
            .expect_err("expected id-order missing reference");

        assert_eq!(
            dataset_order_err,
            MissingDatasetReference::Image {
                annotation_id: AnnotationId::new(20),
                image_id: ImageId::new(20),
            }
        );
        assert_eq!(
            id_order_err,
            MissingDatasetReference::Image {
                annotation_id: AnnotationId::new(10),
                image_id: ImageId::new(10),
            }
        );
    }

    #[test]
    fn category_name_uses_category_lookup() {
        let dataset = sample_dataset();
        let view = WriterDatasetView::new(&dataset);

        assert_eq!(view.category_name(CategoryId::new(2)), Some("dog"));
        assert_eq!(view.category_name(CategoryId::new(999)), None);
    }
}
