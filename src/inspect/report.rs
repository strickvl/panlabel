//! Inspect report types and terminal formatting.
//!
//! This module provides rich, structured inspection results that are
//! displayed beautifully in the terminal.

use std::fmt;

/// The result of inspecting a dataset.
#[derive(Clone, Debug)]
pub struct InspectReport {
    /// Summary counts for the dataset.
    pub summary: SummarySection,
    /// Label distribution histogram.
    pub labels: LabelsSection,
    /// Bounding box statistics.
    pub bboxes: BBoxStats,
    /// Display options for formatting.
    pub(crate) bar_width: usize,
}

/// Summary counts for the dataset.
#[derive(Clone, Debug, Default)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct LabelCount {
    /// The category/label name.
    pub label: String,
    /// Number of annotations with this label.
    pub count: usize,
}

/// Bounding box statistics.
#[derive(Clone, Debug, Default)]
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

impl fmt::Display for InspectReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header
        writeln!(f)?;
        writeln!(f, "╭─────────────────────────────────────────────────────────────╮")?;
        writeln!(f, "│              📊  Dataset Inspection Report                  │")?;
        writeln!(f, "╰─────────────────────────────────────────────────────────────╯")?;
        writeln!(f)?;

        // Summary section
        self.fmt_summary(f)?;
        writeln!(f)?;

        // Labels section
        self.fmt_labels(f)?;
        writeln!(f)?;

        // BBox section
        self.fmt_bboxes(f)?;

        Ok(())
    }
}

impl InspectReport {
    fn fmt_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.summary;

        writeln!(f, "┌─ Summary ─────────────────────────────────────────────────┐")?;
        writeln!(f, "│                                                           │")?;
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
        writeln!(f, "│                                                           │")?;

        // Show annotated vs total images
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
            " ".repeat(59 - 28 - format_number(s.annotated_images).len() - format_number(s.images).len() - format!("{:.1}", pct).len())
        )?;
        writeln!(f, "│                                                           │")?;
        writeln!(f, "└───────────────────────────────────────────────────────────┘")?;

        Ok(())
    }

    fn fmt_labels(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = &self.labels;

        let header = if l.total_distinct > l.top_n {
            format!("Labels (top {} of {})", l.top_n, l.total_distinct)
        } else {
            format!("Labels ({})", l.total_distinct)
        };

        writeln!(f, "┌─ {} {}┐", header, "─".repeat(57 - header.len()))?;
        writeln!(f, "│                                                           │")?;

        if l.entries.is_empty() {
            writeln!(f, "│   No annotations found.                                   │")?;
        } else {
            // Find max count for bar scaling
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

            // Show "Other" bucket if there are more categories
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

        writeln!(f, "│                                                           │")?;
        writeln!(f, "└───────────────────────────────────────────────────────────┘")?;

        Ok(())
    }

    fn fmt_bboxes(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.bboxes;

        writeln!(f, "┌─ Bounding Boxes ──────────────────────────────────────────┐")?;
        writeln!(f, "│                                                           │")?;

        if b.total == 0 {
            writeln!(f, "│   No bounding boxes found.                                │")?;
        } else {
            // Dimensions
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
                writeln!(f, "│   Width/Height:   No valid bounding boxes to measure      │")?;
            }

            writeln!(f, "│                                                           │")?;

            // Quality metrics
            writeln!(f, "│   Quality metrics:                                        │")?;

            // Finite coordinates
            let finite_pct = fmt_percent(b.finite, b.total);
            writeln!(
                f,
                "│     ✓ Finite coords:     {:>7} / {:>7}  ({:>5})      │",
                format_number(b.finite),
                format_number(b.total),
                finite_pct
            )?;

            // Properly ordered
            let ordered_pct = fmt_percent(b.ordered, b.total);
            writeln!(
                f,
                "│     ✓ Properly ordered:  {:>7} / {:>7}  ({:>5})      │",
                format_number(b.ordered),
                format_number(b.total),
                ordered_pct
            )?;

            writeln!(f, "│                                                           │")?;

            // Issues (if any)
            let has_issues = b.degenerate_area > 0
                || b.out_of_bounds > 0
                || b.missing_image_ref > 0
                || b.finite < b.total;

            if has_issues {
                writeln!(f, "│   Issues found:                                           │")?;

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
                writeln!(f, "│   ✓ No issues detected                                    │")?;
            }
        }

        writeln!(f, "│                                                           │")?;
        writeln!(f, "└───────────────────────────────────────────────────────────┘")?;

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
    let filled = filled.min(width); // Clamp to width

    // Use Unicode blocks for a nicer look
    "█".repeat(filled) + &"░".repeat(width - filled)
}

/// Pad a bar string to ensure consistent column alignment.
fn pad_bar(bar: &str, width: usize) -> String {
    // Each Unicode char is 1 char, but we want consistent visual width
    let visual_len = bar.chars().count();
    let padding = (width + 2).saturating_sub(visual_len);
    format!("{}{}", bar, " ".repeat(padding))
}

/// Truncate a label to fit in the display column.
fn truncate_label(label: &str, max_len: usize) -> String {
    if label.len() <= max_len {
        label.to_string()
    } else {
        format!("{}…", &label[..max_len - 1])
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
    fn test_truncate_label() {
        assert_eq!(truncate_label("short", 10), "short");
        assert_eq!(truncate_label("verylonglabel", 10), "verylongl…");
    }
}
