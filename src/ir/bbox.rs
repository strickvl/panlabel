//! Bounding box types in canonical XYXY format.

use serde::{Deserialize, Serialize};

use super::coord::Coord;

/// An axis-aligned bounding box in XYXY format (xmin, ymin, xmax, ymax).
///
/// The `TSpace` parameter should be either [`Pixel`](super::Pixel) or
/// [`Normalized`](super::Normalized), ensuring type safety across
/// coordinate spaces.
///
/// Note: This type does NOT enforce that min < max in the constructor,
/// allowing "malformed" boxes to exist in the IR. This is intentional -
/// validation should catch and report these issues rather than preventing
/// them from being represented.
#[derive(Clone, Copy, PartialEq)]
pub struct BBoxXYXY<TSpace> {
    pub min: Coord<TSpace>,
    pub max: Coord<TSpace>,
}

impl<TSpace> BBoxXYXY<TSpace> {
    /// Creates a new bounding box from min and max coordinates.
    #[inline]
    pub fn new(min: Coord<TSpace>, max: Coord<TSpace>) -> Self {
        Self { min, max }
    }

    /// Creates a new bounding box from explicit coordinates.
    #[inline]
    pub fn from_xyxy(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Self {
        Self {
            min: Coord::new(xmin, ymin),
            max: Coord::new(xmax, ymax),
        }
    }

    /// Returns the minimum x coordinate.
    #[inline]
    pub fn xmin(&self) -> f64 {
        self.min.x
    }

    /// Returns the minimum y coordinate.
    #[inline]
    pub fn ymin(&self) -> f64 {
        self.min.y
    }

    /// Returns the maximum x coordinate.
    #[inline]
    pub fn xmax(&self) -> f64 {
        self.max.x
    }

    /// Returns the maximum y coordinate.
    #[inline]
    pub fn ymax(&self) -> f64 {
        self.max.y
    }

    /// Returns the width of the bounding box.
    ///
    /// May be negative if the box is malformed (xmax < xmin).
    #[inline]
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Returns the height of the bounding box.
    ///
    /// May be negative if the box is malformed (ymax < ymin).
    #[inline]
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Returns the area of the bounding box.
    ///
    /// May be negative if the box is malformed.
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Returns true if all coordinates are finite (not NaN or infinite).
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }

    /// Returns true if the box is properly ordered (min <= max for both axes).
    #[inline]
    pub fn is_ordered(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y
    }
}

impl<TSpace> std::fmt::Debug for BBoxXYXY<TSpace> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BBoxXYXY")
            .field("xmin", &self.min.x)
            .field("ymin", &self.min.y)
            .field("xmax", &self.max.x)
            .field("ymax", &self.max.y)
            .finish()
    }
}

impl<TSpace> Default for BBoxXYXY<TSpace> {
    fn default() -> Self {
        Self::from_xyxy(0.0, 0.0, 0.0, 0.0)
    }
}

// Custom serde implementation to avoid TSpace: Serialize/Deserialize bounds
impl<TSpace> Serialize for BBoxXYXY<TSpace> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("BBoxXYXY", 4)?;
        state.serialize_field("xmin", &self.min.x)?;
        state.serialize_field("ymin", &self.min.y)?;
        state.serialize_field("xmax", &self.max.x)?;
        state.serialize_field("ymax", &self.max.y)?;
        state.end()
    }
}

impl<'de, TSpace> Deserialize<'de> for BBoxXYXY<TSpace> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct BBoxData {
            xmin: f64,
            ymin: f64,
            xmax: f64,
            ymax: f64,
        }
        let data = BBoxData::deserialize(deserializer)?;
        Ok(BBoxXYXY::from_xyxy(
            data.xmin, data.ymin, data.xmax, data.ymax,
        ))
    }
}

/// Conversion utilities for different bbox formats.
impl<TSpace> BBoxXYXY<TSpace> {
    /// Converts from XYWH format (x, y, width, height) where (x, y) is the top-left corner.
    ///
    /// This is the format used by COCO annotations.
    #[inline]
    pub fn from_xywh(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self::from_xyxy(x, y, x + width, y + height)
    }

    /// Converts to XYWH format (x, y, width, height).
    #[inline]
    pub fn to_xywh(&self) -> (f64, f64, f64, f64) {
        (self.xmin(), self.ymin(), self.width(), self.height())
    }
}

/// Conversion between pixel and normalized coordinates.
use super::{Normalized, Pixel};

impl BBoxXYXY<Pixel> {
    /// Converts pixel coordinates to normalized coordinates.
    ///
    /// # Arguments
    /// * `image_width` - The width of the image in pixels
    /// * `image_height` - The height of the image in pixels
    pub fn to_normalized(&self, image_width: f64, image_height: f64) -> BBoxXYXY<Normalized> {
        BBoxXYXY::from_xyxy(
            self.min.x / image_width,
            self.min.y / image_height,
            self.max.x / image_width,
            self.max.y / image_height,
        )
    }
}

impl BBoxXYXY<Normalized> {
    /// Converts normalized coordinates to pixel coordinates.
    ///
    /// # Arguments
    /// * `image_width` - The width of the image in pixels
    /// * `image_height` - The height of the image in pixels
    pub fn to_pixel(&self, image_width: f64, image_height: f64) -> BBoxXYXY<Pixel> {
        BBoxXYXY::from_xyxy(
            self.min.x * image_width,
            self.min.y * image_height,
            self.max.x * image_width,
            self.max.y * image_height,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    #[test]
    fn test_bbox_from_xyxy() {
        let bbox: BBoxXYXY<Pixel> = BBoxXYXY::from_xyxy(10.0, 20.0, 100.0, 80.0);
        assert_eq!(bbox.xmin(), 10.0);
        assert_eq!(bbox.ymin(), 20.0);
        assert_eq!(bbox.xmax(), 100.0);
        assert_eq!(bbox.ymax(), 80.0);
    }

    #[test]
    fn test_bbox_from_xywh() {
        let bbox: BBoxXYXY<Pixel> = BBoxXYXY::from_xywh(10.0, 20.0, 90.0, 60.0);
        assert_eq!(bbox.xmin(), 10.0);
        assert_eq!(bbox.ymin(), 20.0);
        assert_eq!(bbox.xmax(), 100.0);
        assert_eq!(bbox.ymax(), 80.0);
    }

    #[test]
    fn test_bbox_dimensions() {
        let bbox: BBoxXYXY<Pixel> = BBoxXYXY::from_xyxy(10.0, 20.0, 100.0, 80.0);
        assert_eq!(bbox.width(), 90.0);
        assert_eq!(bbox.height(), 60.0);
        assert_eq!(bbox.area(), 5400.0);
    }

    #[test]
    fn test_bbox_ordering() {
        let ordered: BBoxXYXY<Pixel> = BBoxXYXY::from_xyxy(10.0, 20.0, 100.0, 80.0);
        assert!(ordered.is_ordered());

        let unordered: BBoxXYXY<Pixel> = BBoxXYXY::from_xyxy(100.0, 80.0, 10.0, 20.0);
        assert!(!unordered.is_ordered());
    }

    #[test]
    fn test_bbox_to_xywh_roundtrip() {
        let original: BBoxXYXY<Pixel> = BBoxXYXY::from_xywh(15.0, 25.0, 50.0, 30.0);
        let (x, y, w, h) = original.to_xywh();
        let restored: BBoxXYXY<Pixel> = BBoxXYXY::from_xywh(x, y, w, h);
        assert_eq!(original, restored);
    }
}
