//! Stats report types and terminal formatting.
//!
//! This module provides rich, structured dataset statistics that can be
//! rendered as text (Display), serialized as JSON, or used for HTML charts.

use serde::Serialize;
use std::fmt;

/// The result of computing dataset statistics.
#[derive(Clone, Debug, Serialize)]
pub struct StatsReport {
    /// Summary counts for the dataset.
    pub summary: SummarySection,
    /// Label distribution histogram.
    pub labels: LabelsSection,
    /// Bounding box statistics.
    pub bboxes: BBoxStats,
    /// Image resolution spread.
    pub image_resolutions: ImageResolutionStats,
    /// Annotation density across images.
    pub annotation_density: AnnotationDensityStats,
    /// Bounding box area distribution buckets.
    pub area_distribution: AreaDistribution,
    /// Bounding box aspect ratio distribution buckets.
    pub aspect_ratios: AspectRatioDistribution,
    /// Per-category bounding box area statistics.
    pub per_category_bbox: Vec<PerCategoryBBoxStats>,
    /// Top category co-occurrence pairs.
    pub cooccurrence_top_pairs: CooccurrenceTopPairs,
    /// Display-only option for histogram rendering width.
    #[serde(skip)]
    pub(crate) bar_width: usize,
}

/// Summary counts for the dataset.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SummarySection {
    /// Total number of images.
    pub images: usize,
    /// Total number of categories.
    pub categories: usize,
    /// Total number of annotations.
    pub annotations: usize,
    /// Total number of licenses.
    pub licenses: usize,
    /// Number of images that have at least one annotation.
    pub annotated_images: usize,
}

/// Label distribution section.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LabelsSection {
    /// How many top labels to show.
    pub top_n: usize,
    /// Total distinct categories in the dataset.
    pub total_distinct: usize,
    /// Total annotations counted.
    pub total_annotations: usize,
    /// Top label entries (sorted by count descending).
    pub entries: Vec<LabelCount>,
    /// Sum of counts for labels not in the top N.
    pub other_count: usize,
}

/// A single label with its annotation count.
#[derive(Clone, Debug, Serialize)]
pub struct LabelCount {
    /// The category/label name.
    pub label: String,
    /// Number of annotations with this label.
    pub count: usize,
}

/// Bounding box statistics.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BBoxStats {
    /// Total annotations analyzed.
    pub total: usize,
    /// Annotations with finite (non-NaN, non-Inf) coordinates.
    pub finite: usize,
    /// Annotations with properly ordered coordinates (min <= max).
    pub ordered: usize,
    /// Annotations where out-of-bounds check was possible.
    pub oob_checked: usize,
    /// Annotations that extend outside image bounds.
    pub out_of_bounds: usize,
    /// Annotations with zero or negative area.
    pub degenerate_area: usize,
    /// Annotations referencing non-existent images.
    pub missing_image_ref: usize,
    /// Minimum bbox width (pixels), if any valid boxes exist.
    pub min_width: Option<f64>,
    /// Maximum bbox width (pixels).
    pub max_width: Option<f64>,
    /// Minimum bbox height (pixels).
    pub min_height: Option<f64>,
    /// Maximum bbox height (pixels).
    pub max_height: Option<f64>,
}

/// Image resolution statistics.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ImageResolutionStats {
    pub min_w: u32,
    pub max_w: u32,
    pub mean_w: f64,
    pub min_h: u32,
    pub max_h: u32,
    pub mean_h: f64,
}

/// Annotation density statistics.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AnnotationDensityStats {
    pub min_per_image: usize,
    pub max_per_image: usize,
    pub mean_per_image: f64,
    pub zero_annotation_images: usize,
}

/// Bounding box area bucket counts.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AreaDistribution {
    pub small: usize,
    pub medium: usize,
    pub large: usize,
    pub invalid: usize,
}

/// A single aspect-ratio bucket.
#[derive(Clone, Debug, Serialize)]
pub struct AspectRatioBucket {
    pub name: String,
    pub count: usize,
}

/// Aspect-ratio bucket counts.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AspectRatioDistribution {
    pub buckets: Vec<AspectRatioBucket>,
    pub invalid: usize,
}

/// Per-category bbox area stats.
#[derive(Clone, Debug, Serialize)]
pub struct PerCategoryBBoxStats {
    pub category: String,
    pub annotations: usize,
    pub min_area: Option<f64>,
    pub max_area: Option<f64>,
    pub mean_area: Option<f64>,
}

/// A single co-occurrence pair.
#[derive(Clone, Debug, Serialize)]
pub struct CooccurrencePair {
    pub a: String,
    pub b: String,
    pub count: usize,
}

/// Top category co-occurrence pairs.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CooccurrenceTopPairs {
    pub top_n: usize,
    pub pairs: Vec<CooccurrencePair>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextReportStyle {
    Rich,
    Plain,
}

pub struct StatsReportDisplay<'a> {
    report: &'a StatsReport,
    style: TextReportStyle,
}

impl fmt::Display for StatsReportDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.style {
            TextReportStyle::Rich => self.report.fmt_rich(f),
            TextReportStyle::Plain => self.report.fmt_plain(f),
        }
    }
}

impl fmt::Display for StatsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.display(TextReportStyle::Rich), f)
    }
}

impl StatsReport {
    pub fn display(&self, style: TextReportStyle) -> StatsReportDisplay<'_> {
        StatsReportDisplay {
            report: self,
            style,
        }
    }

    fn fmt_rich(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(
            f,
            "╭─────────────────────────────────────────────────────────────╮"
        )?;
        writeln!(
            f,
            "│                📊  Dataset Stats Report                    │"
        )?;
        writeln!(
            f,
            "╰─────────────────────────────────────────────────────────────╯"
        )?;
        writeln!(f)?;

        self.fmt_summary(f)?;
        writeln!(f)?;
        self.fmt_labels(f)?;
        writeln!(f)?;
        self.fmt_bboxes(f)?;
        writeln!(f)?;
        self.fmt_image_resolutions(f)?;
        writeln!(f)?;
        self.fmt_annotation_density(f)?;
        writeln!(f)?;
        self.fmt_area_distribution(f)?;
        writeln!(f)?;
        self.fmt_aspect_ratios(f)?;
        writeln!(f)?;
        self.fmt_per_category_bbox(f)?;
        writeln!(f)?;
        self.fmt_cooccurrence(f)?;

        Ok(())
    }

    fn fmt_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dataset Stats Report")?;
        writeln!(f)?;
        self.fmt_summary_plain(f)?;
        writeln!(f)?;
        self.fmt_labels_plain(f)?;
        writeln!(f)?;
        self.fmt_bboxes_plain(f)?;
        writeln!(f)?;
        self.fmt_image_resolutions_plain(f)?;
        writeln!(f)?;
        self.fmt_annotation_density_plain(f)?;
        writeln!(f)?;
        self.fmt_area_distribution_plain(f)?;
        writeln!(f)?;
        self.fmt_aspect_ratios_plain(f)?;
        writeln!(f)?;
        self.fmt_per_category_bbox_plain(f)?;
        writeln!(f)?;
        self.fmt_cooccurrence_plain(f)?;
        Ok(())
    }

    fn fmt_summary_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.summary;
        fmt_plain_section_header(f, "Summary")?;
        writeln!(f, "Images: {}", format_number(s.images))?;
        writeln!(f, "Categories: {}", format_number(s.categories))?;
        writeln!(f, "Annotations: {}", format_number(s.annotations))?;
        if s.licenses > 0 {
            writeln!(f, "Licenses: {}", format_number(s.licenses))?;
        }
        let pct = if s.images > 0 {
            (s.annotated_images as f64 / s.images as f64) * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "Annotated images: {} of {} ({:.1}%)",
            format_number(s.annotated_images),
            format_number(s.images),
            pct
        )?;
        Ok(())
    }

    fn fmt_labels_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = &self.labels;
        let header = if l.total_distinct > l.top_n {
            format!("Labels (top {} of {})", l.top_n, l.total_distinct)
        } else {
            format!("Labels ({})", l.total_distinct)
        };
        fmt_plain_section_header(f, &header)?;

        if l.entries.is_empty() {
            writeln!(f, "No annotations found.")?;
            return Ok(());
        }

        let max_count = l.entries.iter().map(|entry| entry.count).max().unwrap_or(1);
        for entry in &l.entries {
            let pct = if l.total_annotations > 0 {
                (entry.count as f64 / l.total_annotations as f64) * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "{:<16} {:>7} {:>5.1}%  {}",
                truncate_label_ascii(&entry.label, 16),
                format_number(entry.count),
                pct,
                pad_bar(
                    &render_ascii_bar(entry.count, max_count, self.bar_width),
                    self.bar_width
                )
            )?;
        }

        if l.other_count > 0 {
            let pct = if l.total_annotations > 0 {
                (l.other_count as f64 / l.total_annotations as f64) * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "{:<16} {:>7} {:>5.1}%  {}",
                "(other)",
                format_number(l.other_count),
                pct,
                pad_bar(
                    &render_ascii_bar(l.other_count, max_count, self.bar_width),
                    self.bar_width
                )
            )?;
        }

        Ok(())
    }

    fn fmt_bboxes_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.bboxes;
        fmt_plain_section_header(f, "Bounding Boxes")?;

        if b.total == 0 {
            writeln!(f, "No bounding boxes found.")?;
            return Ok(());
        }

        if let (Some(min_w), Some(max_w), Some(min_h), Some(max_h)) =
            (b.min_width, b.max_width, b.min_height, b.max_height)
        {
            writeln!(f, "Width (px): min {min_w:>8.1} max {max_w:>8.1}")?;
            writeln!(f, "Height (px): min {min_h:>7.1} max {max_h:>8.1}")?;
        } else {
            writeln!(f, "Width/Height: No valid bounding boxes to measure")?;
        }

        writeln!(f)?;
        writeln!(f, "Quality metrics:")?;
        writeln!(
            f,
            "  OK   finite coords:    {:>7} / {:>7} ({:>5})",
            format_number(b.finite),
            format_number(b.total),
            fmt_percent(b.finite, b.total)
        )?;
        writeln!(
            f,
            "  OK   properly ordered: {:>7} / {:>7} ({:>5})",
            format_number(b.ordered),
            format_number(b.total),
            fmt_percent(b.ordered, b.total)
        )?;

        let has_issues = b.degenerate_area > 0
            || b.out_of_bounds > 0
            || b.missing_image_ref > 0
            || b.finite < b.total;

        if has_issues {
            writeln!(f)?;
            writeln!(f, "Issues found:")?;

            if b.degenerate_area > 0 {
                writeln!(
                    f,
                    "  WARN degenerate area:  {:>7} / {:>7} ({:>5})",
                    format_number(b.degenerate_area),
                    format_number(b.total),
                    fmt_percent(b.degenerate_area, b.total)
                )?;
            }

            if b.out_of_bounds > 0 {
                writeln!(
                    f,
                    "  WARN out of bounds:    {:>7} / {:>7} ({:>5})",
                    format_number(b.out_of_bounds),
                    format_number(b.oob_checked),
                    fmt_percent(b.out_of_bounds, b.oob_checked)
                )?;
            }

            if b.missing_image_ref > 0 {
                writeln!(
                    f,
                    "  ERR  missing image ref:{:>8} / {:>7} ({:>5})",
                    format_number(b.missing_image_ref),
                    format_number(b.total),
                    fmt_percent(b.missing_image_ref, b.total)
                )?;
            }

            if b.finite < b.total {
                let non_finite = b.total - b.finite;
                writeln!(
                    f,
                    "  ERR  non-finite coords:{:>8} / {:>7} ({:>5})",
                    format_number(non_finite),
                    format_number(b.total),
                    fmt_percent(non_finite, b.total)
                )?;
            }
        } else {
            writeln!(f)?;
            writeln!(f, "No issues detected.")?;
        }

        Ok(())
    }

    fn fmt_image_resolutions_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.image_resolutions;
        fmt_plain_section_header(f, "Image Resolutions")?;
        writeln!(
            f,
            "Width (px): min {:>6} mean {:>8.1} max {:>6}",
            s.min_w, s.mean_w, s.max_w
        )?;
        writeln!(
            f,
            "Height (px): min {:>5} mean {:>8.1} max {:>6}",
            s.min_h, s.mean_h, s.max_h
        )?;
        Ok(())
    }

    fn fmt_annotation_density_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.annotation_density;
        fmt_plain_section_header(f, "Annotation Density")?;
        writeln!(
            f,
            "Per image: min {:>6} mean {:>8.2} max {:>6}",
            s.min_per_image, s.mean_per_image, s.max_per_image
        )?;
        writeln!(
            f,
            "Images with 0 annotations: {}",
            format_number(s.zero_annotation_images)
        )?;
        Ok(())
    }

    fn fmt_area_distribution_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = &self.area_distribution;
        let max_count = [a.small, a.medium, a.large].into_iter().max().unwrap_or(1);
        fmt_plain_section_header(f, "Area Distribution (COCO: small<1024, medium<9216)")?;
        writeln!(
            f,
            "small   {:>7}  {}",
            format_number(a.small),
            pad_bar(
                &render_ascii_bar(a.small, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(
            f,
            "medium  {:>7}  {}",
            format_number(a.medium),
            pad_bar(
                &render_ascii_bar(a.medium, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(
            f,
            "large   {:>7}  {}",
            format_number(a.large),
            pad_bar(
                &render_ascii_bar(a.large, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(f, "invalid {:>7}", format_number(a.invalid))?;
        Ok(())
    }

    fn fmt_aspect_ratios_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = &self.aspect_ratios;
        let max_count = a
            .buckets
            .iter()
            .map(|bucket| bucket.count)
            .max()
            .unwrap_or(1);
        fmt_plain_section_header(f, "Aspect Ratios (w/h)")?;
        for bucket in &a.buckets {
            writeln!(
                f,
                "{:<8} {:>7}  {}",
                bucket.name,
                format_number(bucket.count),
                pad_bar(
                    &render_ascii_bar(bucket.count, max_count, self.bar_width),
                    self.bar_width
                )
            )?;
        }
        writeln!(f, "invalid  {:>7}", format_number(a.invalid))?;
        Ok(())
    }

    fn fmt_per_category_bbox_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_plain_section_header(f, "Per-category BBox Area (top)")?;
        if self.per_category_bbox.is_empty() {
            writeln!(f, "No per-category bbox stats available.")?;
            return Ok(());
        }

        for row in &self.per_category_bbox {
            let min = row
                .min_area
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            let mean = row
                .mean_area
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            let max = row
                .max_area
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            writeln!(
                f,
                "{:<14} n={:>5} min {:>8} mean {:>8} max {:>8}",
                truncate_label_ascii(&row.category, 14),
                row.annotations,
                min,
                mean,
                max
            )?;
        }
        Ok(())
    }

    fn fmt_cooccurrence_plain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cooccurrence_top_pairs;
        fmt_plain_section_header(f, "Co-occurrence Top Pairs")?;
        if c.pairs.is_empty() {
            writeln!(f, "No co-occurring category pairs found.")?;
            return Ok(());
        }

        for pair in &c.pairs {
            let label = format!("{} + {}", pair.a, pair.b);
            writeln!(
                f,
                "{:<38} {:>9}",
                truncate_label_ascii(&label, 38),
                format_number(pair.count)
            )?;
        }
        Ok(())
    }

    fn fmt_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.summary;

        writeln!(
            f,
            "┌─ Summary ─────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "│   Images:        {:>8}                                  │",
            format_number(s.images)
        )?;
        writeln!(
            f,
            "│   Categories:    {:>8}                                  │",
            format_number(s.categories)
        )?;
        writeln!(
            f,
            "│   Annotations:   {:>8}                                  │",
            format_number(s.annotations)
        )?;
        if s.licenses > 0 {
            writeln!(
                f,
                "│   Licenses:      {:>8}                                  │",
                format_number(s.licenses)
            )?;
        }
        writeln!(
            f,
            "│                                                           │"
        )?;

        let pct = if s.images > 0 {
            (s.annotated_images as f64 / s.images as f64) * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "│   Annotated:     {:>8} of {} ({:.1}%){}│",
            format_number(s.annotated_images),
            format_number(s.images),
            pct,
            " ".repeat(
                59usize
                    .saturating_sub(28)
                    .saturating_sub(format_number(s.annotated_images).len())
                    .saturating_sub(format_number(s.images).len())
                    .saturating_sub(format!("{:.1}", pct).len())
            )
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;

        Ok(())
    }

    fn fmt_labels(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = &self.labels;

        let header = if l.total_distinct > l.top_n {
            format!("Labels (top {} of {})", l.top_n, l.total_distinct)
        } else {
            format!("Labels ({})", l.total_distinct)
        };

        writeln!(
            f,
            "┌─ {} {}┐",
            header,
            "─".repeat(57usize.saturating_sub(header.len()))
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;

        if l.entries.is_empty() {
            writeln!(
                f,
                "│   No annotations found.                                   │"
            )?;
        } else {
            let max_count = l.entries.iter().map(|e| e.count).max().unwrap_or(1);

            for entry in &l.entries {
                let pct = if l.total_annotations > 0 {
                    (entry.count as f64 / l.total_annotations as f64) * 100.0
                } else {
                    0.0
                };

                let bar = render_bar(entry.count, max_count, self.bar_width);
                let label_display = truncate_label(&entry.label, 16);

                writeln!(
                    f,
                    "│   {:<16} {:>7} {:>5.1}%  {}│",
                    label_display,
                    format_number(entry.count),
                    pct,
                    pad_bar(&bar, self.bar_width)
                )?;
            }

            if l.other_count > 0 {
                let pct = if l.total_annotations > 0 {
                    (l.other_count as f64 / l.total_annotations as f64) * 100.0
                } else {
                    0.0
                };
                let bar = render_bar(l.other_count, max_count, self.bar_width);
                writeln!(
                    f,
                    "│   {:<16} {:>7} {:>5.1}%  {}│",
                    "(other)",
                    format_number(l.other_count),
                    pct,
                    pad_bar(&bar, self.bar_width)
                )?;
            }
        }

        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;

        Ok(())
    }

    fn fmt_bboxes(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.bboxes;

        writeln!(
            f,
            "┌─ Bounding Boxes ──────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;

        if b.total == 0 {
            writeln!(
                f,
                "│   No bounding boxes found.                                │"
            )?;
        } else {
            if let (Some(min_w), Some(max_w), Some(min_h), Some(max_h)) =
                (b.min_width, b.max_width, b.min_height, b.max_height)
            {
                writeln!(
                    f,
                    "│   Width  (px):    min {:>8.1}    max {:>8.1}            │",
                    min_w, max_w
                )?;
                writeln!(
                    f,
                    "│   Height (px):    min {:>8.1}    max {:>8.1}            │",
                    min_h, max_h
                )?;
            } else {
                writeln!(
                    f,
                    "│   Width/Height:   No valid bounding boxes to measure      │"
                )?;
            }

            writeln!(
                f,
                "│                                                           │"
            )?;
            writeln!(
                f,
                "│   Quality metrics:                                        │"
            )?;

            let finite_pct = fmt_percent(b.finite, b.total);
            writeln!(
                f,
                "│     ✓ Finite coords:     {:>7} / {:>7}  ({:>5})      │",
                format_number(b.finite),
                format_number(b.total),
                finite_pct
            )?;

            let ordered_pct = fmt_percent(b.ordered, b.total);
            writeln!(
                f,
                "│     ✓ Properly ordered:  {:>7} / {:>7}  ({:>5})      │",
                format_number(b.ordered),
                format_number(b.total),
                ordered_pct
            )?;

            writeln!(
                f,
                "│                                                           │"
            )?;

            let has_issues = b.degenerate_area > 0
                || b.out_of_bounds > 0
                || b.missing_image_ref > 0
                || b.finite < b.total;

            if has_issues {
                writeln!(
                    f,
                    "│   Issues found:                                           │"
                )?;

                if b.degenerate_area > 0 {
                    let pct = fmt_percent(b.degenerate_area, b.total);
                    writeln!(
                        f,
                        "│     ⚠ Degenerate area:   {:>7} / {:>7}  ({:>5})      │",
                        format_number(b.degenerate_area),
                        format_number(b.total),
                        pct
                    )?;
                }

                if b.out_of_bounds > 0 {
                    let pct = fmt_percent(b.out_of_bounds, b.oob_checked);
                    writeln!(
                        f,
                        "│     ⚠ Out of bounds:     {:>7} / {:>7}  ({:>5})      │",
                        format_number(b.out_of_bounds),
                        format_number(b.oob_checked),
                        pct
                    )?;
                }

                if b.missing_image_ref > 0 {
                    let pct = fmt_percent(b.missing_image_ref, b.total);
                    writeln!(
                        f,
                        "│     ✗ Missing image ref: {:>7} / {:>7}  ({:>5})      │",
                        format_number(b.missing_image_ref),
                        format_number(b.total),
                        pct
                    )?;
                }

                if b.finite < b.total {
                    let non_finite = b.total - b.finite;
                    let pct = fmt_percent(non_finite, b.total);
                    writeln!(
                        f,
                        "│     ✗ Non-finite coords: {:>7} / {:>7}  ({:>5})      │",
                        format_number(non_finite),
                        format_number(b.total),
                        pct
                    )?;
                }
            } else {
                writeln!(
                    f,
                    "│   ✓ No issues detected                                    │"
                )?;
            }
        }

        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;

        Ok(())
    }

    fn fmt_image_resolutions(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.image_resolutions;
        writeln!(
            f,
            "┌─ Image Resolutions ───────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "│   Width  (px): min {:>6}  mean {:>8.1}  max {:>6}        │",
            s.min_w, s.mean_w, s.max_w
        )?;
        writeln!(
            f,
            "│   Height (px): min {:>6}  mean {:>8.1}  max {:>6}        │",
            s.min_h, s.mean_h, s.max_h
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }

    fn fmt_annotation_density(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.annotation_density;
        writeln!(
            f,
            "┌─ Annotation Density ──────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "│   Per image: min {:>6}  mean {:>8.2}  max {:>6}         │",
            s.min_per_image, s.mean_per_image, s.max_per_image
        )?;
        writeln!(
            f,
            "│   Images with 0 annotations: {:>8}                        │",
            format_number(s.zero_annotation_images)
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }

    fn fmt_area_distribution(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = &self.area_distribution;
        let max_count = [a.small, a.medium, a.large].into_iter().max().unwrap_or(1);

        writeln!(
            f,
            "┌─ Area Distribution (COCO: small<1024, medium<9216) ─────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "│   small   {:>7}  {}│",
            format_number(a.small),
            pad_bar(
                &render_bar(a.small, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(
            f,
            "│   medium  {:>7}  {}│",
            format_number(a.medium),
            pad_bar(
                &render_bar(a.medium, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(
            f,
            "│   large   {:>7}  {}│",
            format_number(a.large),
            pad_bar(
                &render_bar(a.large, max_count, self.bar_width),
                self.bar_width
            )
        )?;
        writeln!(
            f,
            "│   invalid {:>7}                                         │",
            format_number(a.invalid)
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }

    fn fmt_aspect_ratios(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = &self.aspect_ratios;
        let max_count = a.buckets.iter().map(|b| b.count).max().unwrap_or(1);

        writeln!(
            f,
            "┌─ Aspect Ratios (w/h) ────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        for bucket in &a.buckets {
            writeln!(
                f,
                "│   {:<8} {:>7}  {}│",
                bucket.name,
                format_number(bucket.count),
                pad_bar(
                    &render_bar(bucket.count, max_count, self.bar_width),
                    self.bar_width
                )
            )?;
        }
        writeln!(
            f,
            "│   invalid  {:>7}                                         │",
            format_number(a.invalid)
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }

    fn fmt_per_category_bbox(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "┌─ Per-category BBox Area (top) ───────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;

        if self.per_category_bbox.is_empty() {
            writeln!(
                f,
                "│   No per-category bbox stats available.                   │"
            )?;
        } else {
            for row in &self.per_category_bbox {
                let min = row
                    .min_area
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "n/a".to_string());
                let mean = row
                    .mean_area
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "n/a".to_string());
                let max = row
                    .max_area
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "n/a".to_string());

                writeln!(
                    f,
                    "│   {:<14} n={:>5} min {:>8} mean {:>8} max {:>8}│",
                    truncate_label(&row.category, 14),
                    row.annotations,
                    min,
                    mean,
                    max
                )?;
            }
        }

        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }

    fn fmt_cooccurrence(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cooccurrence_top_pairs;
        writeln!(
            f,
            "┌─ Co-occurrence Top Pairs ─────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                                                           │"
        )?;

        if c.pairs.is_empty() {
            writeln!(
                f,
                "│   No co-occurring category pairs found.                   │"
            )?;
        } else {
            for pair in &c.pairs {
                let label = format!("{} + {}", pair.a, pair.b);
                writeln!(
                    f,
                    "│   {:<38} {:>9}                         │",
                    truncate_label(&label, 38),
                    format_number(pair.count)
                )?;
            }
        }

        writeln!(
            f,
            "│                                                           │"
        )?;
        writeln!(
            f,
            "└───────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }
}

/// Format a number with thousands separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format a percentage, handling zero denominators.
fn fmt_percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", (numerator as f64 / denominator as f64) * 100.0)
    }
}

/// Render a horizontal bar using Unicode block characters.
fn render_bar(count: usize, max_count: usize, width: usize) -> String {
    if max_count == 0 || width == 0 {
        return String::new();
    }

    let filled = (count * width) / max_count;
    let filled = filled.min(width);
    "█".repeat(filled) + &"░".repeat(width - filled)
}

/// Render a horizontal bar using plain ASCII characters.
fn render_ascii_bar(count: usize, max_count: usize, width: usize) -> String {
    if max_count == 0 || width == 0 {
        return String::new();
    }

    let filled = (count * width) / max_count;
    let filled = filled.min(width);
    "#".repeat(filled) + &"-".repeat(width - filled)
}

/// Pad a bar string to ensure consistent column alignment.
fn pad_bar(bar: &str, width: usize) -> String {
    let visual_len = bar.chars().count();
    let padding = (width + 2).saturating_sub(visual_len);
    format!("{}{}", bar, " ".repeat(padding))
}

fn fmt_plain_section_header(f: &mut fmt::Formatter<'_>, title: &str) -> fmt::Result {
    writeln!(f, "{title}")?;
    writeln!(f, "{}", "-".repeat(title.len()))
}

/// Truncate a label to fit in the display column.
fn truncate_label(label: &str, max_len: usize) -> String {
    if label.len() <= max_len {
        label.to_string()
    } else {
        format!("{}…", &label[..max_len - 1])
    }
}

/// Truncate a label for plain ASCII output.
fn truncate_label_ascii(label: &str, max_len: usize) -> String {
    if label.chars().count() <= max_len {
        label.to_string()
    } else {
        let keep = max_len.saturating_sub(3);
        let prefix: String = label.chars().take(keep).collect();
        format!("{prefix}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_fmt_percent() {
        assert_eq!(fmt_percent(0, 0), "n/a");
        assert_eq!(fmt_percent(1, 2), "50.0%");
        assert_eq!(fmt_percent(1, 3), "33.3%");
    }

    #[test]
    fn test_render_bar() {
        assert_eq!(render_bar(5, 10, 10), "█████░░░░░");
        assert_eq!(render_bar(10, 10, 10), "██████████");
        assert_eq!(render_bar(0, 10, 10), "░░░░░░░░░░");
    }

    #[test]
    fn test_render_ascii_bar() {
        assert_eq!(render_ascii_bar(5, 10, 10), "#####-----");
        assert_eq!(render_ascii_bar(10, 10, 10), "##########");
        assert_eq!(render_ascii_bar(0, 10, 10), "----------");
    }

    #[test]
    fn test_truncate_label() {
        assert_eq!(truncate_label("short", 10), "short");
        assert_eq!(truncate_label("verylonglabel", 10), "verylongl…");
    }

    #[test]
    fn test_truncate_label_ascii() {
        assert_eq!(truncate_label_ascii("short", 10), "short");
        assert_eq!(truncate_label_ascii("verylonglabel", 10), "verylon...");
    }

    #[test]
    fn test_truncate_label_ascii_handles_unicode_safely() {
        assert_eq!(truncate_label_ascii("überraschung", 8), "überr...");
    }

    #[test]
    fn test_plain_display_uses_ascii_only_markers() {
        let report = StatsReport {
            summary: SummarySection {
                images: 3,
                categories: 2,
                annotations: 4,
                licenses: 0,
                annotated_images: 2,
            },
            labels: LabelsSection {
                top_n: 5,
                total_distinct: 2,
                total_annotations: 4,
                entries: vec![
                    LabelCount {
                        label: "person".to_string(),
                        count: 3,
                    },
                    LabelCount {
                        label: "car".to_string(),
                        count: 1,
                    },
                ],
                other_count: 0,
            },
            bboxes: BBoxStats {
                total: 4,
                finite: 4,
                ordered: 4,
                oob_checked: 4,
                out_of_bounds: 0,
                degenerate_area: 0,
                missing_image_ref: 0,
                min_width: Some(10.0),
                max_width: Some(20.0),
                min_height: Some(5.0),
                max_height: Some(15.0),
            },
            image_resolutions: ImageResolutionStats {
                min_w: 640,
                max_w: 1280,
                mean_w: 960.0,
                min_h: 480,
                max_h: 720,
                mean_h: 600.0,
            },
            annotation_density: AnnotationDensityStats {
                min_per_image: 0,
                max_per_image: 2,
                mean_per_image: 1.3,
                zero_annotation_images: 1,
            },
            area_distribution: AreaDistribution {
                small: 1,
                medium: 2,
                large: 1,
                invalid: 0,
            },
            aspect_ratios: AspectRatioDistribution {
                buckets: vec![AspectRatioBucket {
                    name: "square".to_string(),
                    count: 4,
                }],
                invalid: 0,
            },
            per_category_bbox: vec![PerCategoryBBoxStats {
                category: "person".to_string(),
                annotations: 3,
                min_area: Some(10.0),
                max_area: Some(20.0),
                mean_area: Some(15.0),
            }],
            cooccurrence_top_pairs: CooccurrenceTopPairs {
                top_n: 5,
                pairs: vec![CooccurrencePair {
                    a: "car".to_string(),
                    b: "person".to_string(),
                    count: 1,
                }],
            },
            bar_width: 10,
        };

        let output = format!("{}", report.display(TextReportStyle::Plain));
        assert!(output.contains("Dataset Stats Report"));
        assert!(output.contains("Summary"));
        assert!(output.contains("Bounding Boxes"));
        assert!(output.contains("#####"));
        assert!(!output.contains('📊'));
        assert!(!output.contains('╭'));
        assert!(!output.contains('█'));
        assert!(!output.contains('…'));
    }
}
