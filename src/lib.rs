//! Panlabel: The universal annotation converter.
//!
//! Panlabel converts between different object detection annotation formats,
//! similar to how Pandoc converts between document formats. It uses an
//! intermediate representation (IR) to enable N×M format conversions with
//! only 2N converters.
//!
//! # Modules
//!
//! - [`ir`]: Intermediate representation types (Dataset, Image, Annotation, etc.)
//! - [`validation`]: Dataset validation and error reporting
//! - [`conversion`]: Conversion reporting and lossiness tracking
//! - [`error`]: Error types for panlabel operations

pub mod conversion;
pub mod diff;
pub mod error;
#[cfg(feature = "hf-remote")]
pub mod hf;
pub mod ir;
pub mod sample;
pub mod stats;
pub mod validation;

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

pub use error::PanlabelError;

/// The panlabel CLI application.
#[derive(Parser)]
#[command(name = "panlabel")]
#[command(version, author, about)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Validate a dataset for errors and warnings.
    Validate(ValidateArgs),
    /// Convert a dataset between formats.
    Convert(ConvertArgs),
    /// Show rich dataset statistics.
    Stats(StatsArgs),
    /// Compare two datasets semantically.
    Diff(DiffArgs),
    /// Sample a subset dataset.
    Sample(SampleArgs),
    /// List supported formats and their capabilities.
    ListFormats(ListFormatsArgs),
}

/// Supported formats for conversion.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ConvertFormat {
    /// Panlabel's intermediate representation (JSON).
    #[value(name = "ir-json")]
    IrJson,
    /// COCO object detection format (JSON).
    #[value(name = "coco", alias = "coco-json")]
    Coco,
    /// CVAT for images task export (XML).
    #[value(name = "cvat", alias = "cvat-xml")]
    Cvat,
    /// Label Studio task export (JSON).
    #[value(name = "label-studio", alias = "label-studio-json", alias = "ls")]
    LabelStudio,
    /// TensorFlow Object Detection format (CSV).
    #[value(name = "tfod", alias = "tfod-csv")]
    Tfod,
    /// Ultralytics-style YOLO object detection format (directory-based).
    #[value(
        name = "yolo",
        alias = "ultralytics",
        alias = "yolov8",
        alias = "yolov5"
    )]
    Yolo,
    /// Pascal VOC XML format (directory-based).
    #[value(name = "voc", alias = "pascal-voc", alias = "voc-xml")]
    Voc,
    /// Hugging Face ImageFolder metadata format (directory-based).
    #[value(name = "hf", alias = "hf-imagefolder", alias = "huggingface")]
    HfImagefolder,
}

impl ConvertFormat {
    /// Convert CLI format to conversion module format.
    fn to_conversion_format(self) -> conversion::Format {
        match self {
            ConvertFormat::IrJson => conversion::Format::IrJson,
            ConvertFormat::Coco => conversion::Format::Coco,
            ConvertFormat::Cvat => conversion::Format::Cvat,
            ConvertFormat::LabelStudio => conversion::Format::LabelStudio,
            ConvertFormat::Tfod => conversion::Format::Tfod,
            ConvertFormat::Yolo => conversion::Format::Yolo,
            ConvertFormat::Voc => conversion::Format::Voc,
            ConvertFormat::HfImagefolder => conversion::Format::HfImagefolder,
        }
    }
}

/// Source format for conversion (allows 'auto' for detection).
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ConvertFromFormat {
    /// Auto-detect format from input path.
    #[value(name = "auto")]
    Auto,
    /// Panlabel's intermediate representation (JSON).
    #[value(name = "ir-json")]
    IrJson,
    /// COCO object detection format (JSON).
    #[value(name = "coco", alias = "coco-json")]
    Coco,
    /// CVAT for images task export (XML).
    #[value(name = "cvat", alias = "cvat-xml")]
    Cvat,
    /// Label Studio task export (JSON).
    #[value(name = "label-studio", alias = "label-studio-json", alias = "ls")]
    LabelStudio,
    /// TensorFlow Object Detection format (CSV).
    #[value(name = "tfod", alias = "tfod-csv")]
    Tfod,
    /// Ultralytics-style YOLO object detection format (directory-based).
    #[value(
        name = "yolo",
        alias = "ultralytics",
        alias = "yolov8",
        alias = "yolov5"
    )]
    Yolo,
    /// Pascal VOC XML format (directory-based).
    #[value(name = "voc", alias = "pascal-voc", alias = "voc-xml")]
    Voc,
    /// Hugging Face ImageFolder metadata format (directory-based).
    #[value(name = "hf", alias = "hf-imagefolder", alias = "huggingface")]
    HfImagefolder,
}

impl ConvertFromFormat {
    /// Convert to a concrete format, returning None for Auto.
    fn as_concrete(self) -> Option<ConvertFormat> {
        match self {
            ConvertFromFormat::Auto => None,
            ConvertFromFormat::IrJson => Some(ConvertFormat::IrJson),
            ConvertFromFormat::Coco => Some(ConvertFormat::Coco),
            ConvertFromFormat::Cvat => Some(ConvertFormat::Cvat),
            ConvertFromFormat::LabelStudio => Some(ConvertFormat::LabelStudio),
            ConvertFromFormat::Tfod => Some(ConvertFormat::Tfod),
            ConvertFromFormat::Yolo => Some(ConvertFormat::Yolo),
            ConvertFromFormat::Voc => Some(ConvertFormat::Voc),
            ConvertFromFormat::HfImagefolder => Some(ConvertFormat::HfImagefolder),
        }
    }
}

/// Output format for conversion reports.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum ReportFormat {
    /// Human-readable text output.
    #[default]
    #[value(name = "text")]
    Text,
    /// Machine-readable JSON output.
    #[value(name = "json")]
    Json,
}

/// Output format for stats reports.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum StatsOutputFormat {
    /// Human-readable text output.
    #[default]
    #[value(name = "text")]
    Text,
    /// Machine-readable JSON output.
    #[value(name = "json")]
    Json,
    /// Self-contained HTML report.
    #[value(name = "html")]
    Html,
}

/// Annotation matching strategy for dataset diff.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum DiffMatchBy {
    /// Match annotations by ID.
    #[default]
    #[value(name = "id")]
    Id,
    /// Match annotations by IoU.
    #[value(name = "iou")]
    Iou,
}

/// Image sampling strategy.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum SampleStrategyArg {
    /// Uniform random sampling.
    #[default]
    #[value(name = "random")]
    Random,
    /// Category-aware stratified sampling.
    #[value(name = "stratified")]
    Stratified,
}

/// Category filter mode.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum CategoryModeArg {
    /// Keep whole images that contain at least one selected category.
    #[default]
    #[value(name = "images")]
    Images,
    /// Keep only selected-category annotations.
    #[value(name = "annotations")]
    Annotations,
}

/// HF bbox format interpretation.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum HfBboxFormatArg {
    /// `[x, y, width, height]`
    #[default]
    #[value(name = "xywh")]
    Xywh,
    /// `[x1, y1, x2, y2]`
    #[value(name = "xyxy")]
    Xyxy,
}

impl HfBboxFormatArg {
    fn to_hf_bbox_format(self) -> ir::io_hf_imagefolder::HfBboxFormat {
        match self {
            HfBboxFormatArg::Xywh => ir::io_hf_imagefolder::HfBboxFormat::Xywh,
            HfBboxFormatArg::Xyxy => ir::io_hf_imagefolder::HfBboxFormat::Xyxy,
        }
    }
}

/// Arguments for the validate subcommand.
#[derive(clap::Args)]
struct ValidateArgs {
    /// Input path to validate.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', 'cvat', 'label-studio', 'tfod', 'yolo', 'voc', or 'hf').
    #[arg(long, default_value = "ir-json")]
    format: String,

    /// Treat warnings as errors (exit non-zero if any warnings).
    #[arg(long)]
    strict: bool,

    /// Output format for the report ('text' or 'json').
    #[arg(long, default_value = "text")]
    output: String,
}

/// Arguments for the stats subcommand.
#[derive(clap::Args)]
struct StatsArgs {
    /// Input path to analyze.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', 'cvat', 'label-studio', 'tfod', 'yolo', 'voc', or 'hf').
    ///
    /// If omitted, panlabel auto-detects the format. If detection fails for a JSON
    /// file, stats falls back to reading as ir-json.
    #[arg(long, value_enum)]
    format: Option<ConvertFormat>,

    /// Number of top labels / pairs to show.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Tolerance in pixels for out-of-bounds checks.
    #[arg(long, default_value_t = 0.5)]
    tolerance: f64,

    /// Output format for the stats report.
    #[arg(long, value_enum, default_value = "text")]
    output: StatsOutputFormat,
}

/// Arguments for the diff subcommand.
#[derive(clap::Args)]
struct DiffArgs {
    /// First dataset path.
    input_a: PathBuf,

    /// Second dataset path.
    input_b: PathBuf,

    /// Format for the first input (or auto-detect).
    #[arg(long = "format-a", value_enum, default_value = "auto")]
    format_a: ConvertFromFormat,

    /// Format for the second input (or auto-detect).
    #[arg(long = "format-b", value_enum, default_value = "auto")]
    format_b: ConvertFromFormat,

    /// Annotation matching strategy.
    #[arg(long, value_enum, default_value = "id")]
    match_by: DiffMatchBy,

    /// IoU threshold used with --match-by iou.
    #[arg(long, default_value_t = 0.5)]
    iou_threshold: f64,

    /// Include item-level detail in output.
    #[arg(long)]
    detail: bool,

    /// Output format for diff report.
    #[arg(long, value_enum, default_value = "text")]
    output: ReportFormat,
}

/// Arguments for the sample subcommand.
#[derive(clap::Args)]
struct SampleArgs {
    /// Input path.
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output path.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Source format (or auto-detect).
    #[arg(long = "from", value_enum, default_value = "auto")]
    from: ConvertFromFormat,

    /// Target format.
    #[arg(long = "to", value_enum)]
    to: Option<ConvertFormat>,

    /// Number of images to sample.
    #[arg(short = 'n', long = "n")]
    n: Option<usize>,

    /// Fraction of images to sample.
    #[arg(long = "fraction")]
    fraction: Option<f64>,

    /// Optional random seed for deterministic sampling.
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Sampling strategy.
    #[arg(long, value_enum, default_value = "random")]
    strategy: SampleStrategyArg,

    /// Comma-separated category names to filter on.
    #[arg(long = "categories")]
    categories: Option<String>,

    /// Category filter mode.
    #[arg(long = "category-mode", value_enum, default_value = "images")]
    category_mode: CategoryModeArg,

    /// Allow lossy output format conversions.
    #[arg(long = "allow-lossy")]
    allow_lossy: bool,
}

/// Arguments for the convert subcommand.
#[derive(clap::Args)]
struct ConvertArgs {
    /// Source format (use 'auto' for automatic detection).
    #[arg(short = 'f', long = "from", value_enum)]
    from: ConvertFromFormat,

    /// Target format.
    #[arg(short = 't', long = "to", value_enum)]
    to: ConvertFormat,

    /// Input path (required for local inputs; optional with --hf-repo when --from hf).
    #[arg(short = 'i', long = "input")]
    input: Option<PathBuf>,

    /// Output path.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Treat validation warnings as errors.
    #[arg(long)]
    strict: bool,

    /// Skip input validation entirely.
    #[arg(long = "no-validate")]
    no_validate: bool,

    /// Allow conversions that drop information (e.g., metadata, images without annotations).
    #[arg(long = "allow-lossy")]
    allow_lossy: bool,

    /// Output format for the conversion report ('text' or 'json').
    #[arg(long, value_enum, default_value = "text")]
    report: ReportFormat,

    /// HF bbox format for --from hf / --to hf (xywh or xyxy).
    #[arg(long = "hf-bbox-format", value_enum, default_value = "xywh")]
    hf_bbox_format: HfBboxFormatArg,

    /// Override the object container column in HF metadata (e.g. annotations).
    #[arg(long = "hf-objects-column")]
    hf_objects_column: Option<String>,

    /// JSON file mapping integer category IDs to names for HF import.
    #[arg(long = "hf-category-map")]
    hf_category_map: Option<PathBuf>,

    /// HF dataset repo ID or dataset page URL for remote import.
    #[arg(long = "hf-repo")]
    hf_repo: Option<String>,

    /// HF split name (e.g. train/validation/test).
    #[arg(long = "split")]
    split: Option<String>,

    /// HF revision (branch, tag, or commit SHA).
    #[arg(long = "revision")]
    revision: Option<String>,

    /// HF config/subset.
    #[arg(long = "config")]
    config: Option<String>,

    /// HF auth token (also supports HF_TOKEN env var).
    #[arg(long = "token", env = "HF_TOKEN")]
    token: Option<String>,
}

/// Arguments for the list-formats subcommand.
#[derive(clap::Args)]
struct ListFormatsArgs {}

const SUPPORTED_VALIDATE_FORMATS: &str = "ir-json, coco, cvat, label-studio, tfod, yolo, voc, hf";

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Validate(args)) => run_validate(args),
        Some(Commands::Convert(args)) => run_convert(args),
        Some(Commands::Stats(args)) => run_stats(args),
        Some(Commands::Diff(args)) => run_diff(args),
        Some(Commands::Sample(args)) => run_sample(args),
        Some(Commands::ListFormats(args)) => run_list_formats(args),
        None => {
            // No subcommand: just print help hint and exit successfully
            // This keeps backward compatibility with the existing test
            println!("panlabel {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("The universal annotation converter.");
            println!();
            println!("Run 'panlabel --help' for usage information.");
            Ok(())
        }
    }
}

/// Execute the stats subcommand.
fn run_stats(args: StatsArgs) -> Result<(), PanlabelError> {
    let format = resolve_stats_format(args.format, &args.input)?;
    let dataset = read_dataset(format, &args.input)?;

    let opts = stats::StatsOptions {
        top_labels: args.top,
        top_pairs: args.top,
        oob_tolerance_px: args.tolerance,
        bar_width: 20,
    };

    let report = stats::stats_dataset(&dataset, &opts);

    match args.output {
        StatsOutputFormat::Text => print!("{}", report),
        StatsOutputFormat::Json => {
            serde_json::to_writer_pretty(std::io::stdout(), &report)
                .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
            println!();
        }
        StatsOutputFormat::Html => {
            let html = stats::html::render_html(&report)?;
            print!("{html}");
        }
    }

    Ok(())
}

/// Execute the diff subcommand.
fn run_diff(args: DiffArgs) -> Result<(), PanlabelError> {
    if matches!(args.match_by, DiffMatchBy::Iou)
        && !(0.0 < args.iou_threshold && args.iou_threshold <= 1.0)
    {
        return Err(PanlabelError::DiffFailed {
            message: "--iou-threshold must be in the interval (0.0, 1.0] when --match-by iou"
                .to_string(),
        });
    }

    let format_a = resolve_from_format(args.format_a, &args.input_a)?;
    let format_b = resolve_from_format(args.format_b, &args.input_b)?;

    let dataset_a = read_dataset(format_a, &args.input_a)?;
    let dataset_b = read_dataset(format_b, &args.input_b)?;

    ensure_unique_image_file_names(&dataset_a, "A")?;
    ensure_unique_image_file_names(&dataset_b, "B")?;

    let match_by = match args.match_by {
        DiffMatchBy::Id => diff::MatchBy::Id,
        DiffMatchBy::Iou => diff::MatchBy::Iou,
    };

    let opts = diff::DiffOptions {
        match_by,
        iou_threshold: args.iou_threshold,
        detail: args.detail,
        max_items: 20,
        bbox_eps: 1e-6,
    };

    let report = diff::diff_datasets(&dataset_a, &dataset_b, &opts);

    match args.output {
        ReportFormat::Text => {
            println!(
                "Dataset Diff: {} vs {}",
                args.input_a.display(),
                args.input_b.display()
            );
            println!();
            print!("{}", report);
        }
        ReportFormat::Json => {
            serde_json::to_writer_pretty(std::io::stdout(), &report)
                .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
            println!();
        }
    }

    Ok(())
}

/// Execute the sample subcommand.
fn run_sample(args: SampleArgs) -> Result<(), PanlabelError> {
    let from_format = resolve_from_format(args.from, &args.input)?;
    let to_format = match args.to {
        Some(target) => target,
        None => args.from.as_concrete().unwrap_or(ConvertFormat::IrJson),
    };

    let dataset = read_dataset(from_format, &args.input)?;

    let strategy = match args.strategy {
        SampleStrategyArg::Random => sample::SampleStrategy::Random,
        SampleStrategyArg::Stratified => sample::SampleStrategy::Stratified,
    };
    let category_mode = match args.category_mode {
        CategoryModeArg::Images => sample::CategoryMode::Images,
        CategoryModeArg::Annotations => sample::CategoryMode::Annotations,
    };

    let sample_opts = sample::SampleOptions {
        n: args.n,
        fraction: args.fraction,
        seed: args.seed,
        strategy,
        categories: parse_categories_arg(args.categories),
        category_mode,
    };

    let sampled_dataset = sample::sample_dataset(&dataset, &sample_opts)?;

    let conv_report = conversion::build_conversion_report(
        &sampled_dataset,
        from_format.to_conversion_format(),
        to_format.to_conversion_format(),
    );

    if conv_report.is_lossy() && !args.allow_lossy {
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(from_format).to_string(),
            to: format_name(to_format).to_string(),
            report: Box::new(conv_report),
        });
    }

    write_dataset(to_format, &args.output, &sampled_dataset)?;

    println!(
        "Sampled {} images -> {} images: {} ({}) -> {} ({})",
        dataset.images.len(),
        sampled_dataset.images.len(),
        args.input.display(),
        format_name(from_format),
        args.output.display(),
        format_name(to_format)
    );
    print!("{}", conv_report);

    Ok(())
}

/// Execute the validate subcommand.
fn run_validate(args: ValidateArgs) -> Result<(), PanlabelError> {
    // Load the dataset based on format
    let dataset = match args.format.as_str() {
        "ir-json" => ir::io_json::read_ir_json(&args.input)?,
        "coco" | "coco-json" => ir::io_coco_json::read_coco_json(&args.input)?,
        "cvat" | "cvat-xml" => ir::io_cvat_xml::read_cvat_xml(&args.input)?,
        "label-studio" | "label-studio-json" | "ls" => {
            ir::io_label_studio_json::read_label_studio_json(&args.input)?
        }
        "tfod" | "tfod-csv" => ir::io_tfod_csv::read_tfod_csv(&args.input)?,
        "yolo" | "ultralytics" | "yolov8" | "yolov5" => ir::io_yolo::read_yolo_dir(&args.input)?,
        "voc" | "pascal-voc" | "voc-xml" => ir::io_voc_xml::read_voc_dir(&args.input)?,
        "hf" | "hf-imagefolder" | "huggingface" => {
            read_hf_dataset_with_default_options(&args.input)?
        }
        other => {
            return Err(PanlabelError::UnsupportedFormat(format!(
                "'{}' (supported: {})",
                other, SUPPORTED_VALIDATE_FORMATS
            )));
        }
    };

    // Validate
    let opts = validation::ValidateOptions {
        strict: args.strict,
    };
    let report = validation::validate_dataset(&dataset, &opts);

    // Output results
    match args.output.as_str() {
        "json" => {
            // JSON output for programmatic use (using serde for proper escaping)
            serde_json::to_writer_pretty(std::io::stdout(), &report.as_json())
                .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
            println!(); // trailing newline
        }
        _ => {
            // Default text output
            print!("{}", report);
        }
    }

    // Determine exit status
    let has_errors = report.error_count() > 0;
    let has_warnings = report.warning_count() > 0;

    if has_errors || (args.strict && has_warnings) {
        Err(PanlabelError::ValidationFailed {
            error_count: report.error_count(),
            warning_count: report.warning_count(),
            report,
        })
    } else {
        Ok(())
    }
}

/// Execute the convert subcommand.
fn run_convert(args: ConvertArgs) -> Result<(), PanlabelError> {
    let from_format = match args.from.as_concrete() {
        Some(format) => format,
        None => {
            let input = args.input.as_ref().ok_or_else(|| {
                PanlabelError::UnsupportedFormat("--from auto requires --input <path>".to_string())
            })?;
            detect_format(input)?
        }
    };

    validate_hf_flag_usage(&args, from_format)?;

    #[allow(unused_mut)]
    let mut hf_read_options = ir::io_hf_imagefolder::HfReadOptions {
        bbox_format: args.hf_bbox_format.to_hf_bbox_format(),
        objects_column: args.hf_objects_column.clone(),
        split: args.split.clone(),
        category_map: load_hf_category_map(args.hf_category_map.as_deref())?,
        provenance: Default::default(),
    };
    let hf_write_options = ir::io_hf_imagefolder::HfWriteOptions {
        bbox_format: args.hf_bbox_format.to_hf_bbox_format(),
    };
    #[cfg(feature = "hf-remote")]
    let mut remote_hf_provenance: Option<std::collections::BTreeMap<String, String>> = None;
    #[cfg(not(feature = "hf-remote"))]
    let remote_hf_provenance: Option<std::collections::BTreeMap<String, String>> = None;

    let (effective_input, source_display, effective_from_format) =
        if from_format == ConvertFormat::HfImagefolder && args.hf_repo.is_some() {
            #[cfg(feature = "hf-remote")]
            {
                let repo_input = args.hf_repo.as_deref().expect("checked is_some");
                let repo_ref = hf::resolve::parse_hf_input(
                    repo_input,
                    args.revision.as_deref(),
                    args.config.as_deref(),
                    args.split.as_deref(),
                )?;

                let preflight = hf::preflight::run_preflight(&repo_ref, args.token.as_deref());
                if preflight.is_none() {
                    eprintln!("Note: HF viewer API unavailable; proceeding with direct download.");
                }

                if let Some(preflight_data) = preflight.as_ref() {
                    if hf_read_options.objects_column.is_none() {
                        hf_read_options.objects_column =
                            preflight_data.detected_objects_column.clone();
                    }

                    if hf_read_options.category_map.is_empty() {
                        if let Some(labels) = preflight_data.category_labels.as_ref() {
                            for (idx, label) in labels.iter().enumerate() {
                                hf_read_options
                                    .category_map
                                    .insert(idx as i64, label.clone());
                            }
                        }
                    }

                    if let Some(license) = preflight_data.license.as_ref() {
                        hf_read_options
                            .provenance
                            .insert("hf_license".to_string(), license.clone());
                    }
                    if let Some(description) = preflight_data.description.as_ref() {
                        hf_read_options
                            .provenance
                            .insert("hf_description".to_string(), description.clone());
                    }

                    if hf_read_options.split.is_none() {
                        hf_read_options.split = preflight_data.selected_split.clone();
                    }
                }

                let acquired =
                    hf::acquire::acquire(&repo_ref, preflight.as_ref(), args.token.as_deref())?;
                let revision = repo_ref
                    .revision
                    .clone()
                    .unwrap_or_else(|| "main".to_string());
                hf_read_options
                    .provenance
                    .insert("hf_repo_id".to_string(), repo_ref.repo_id.clone());
                hf_read_options
                    .provenance
                    .insert("hf_revision".to_string(), revision);
                hf_read_options.provenance.insert(
                    "hf_bbox_format".to_string(),
                    args.hf_bbox_format.to_hf_bbox_format().as_str().to_string(),
                );
                if let Some(split_name) = acquired
                    .split_name
                    .clone()
                    .or_else(|| repo_ref.split.clone())
                {
                    hf_read_options
                        .provenance
                        .insert("hf_split".to_string(), split_name);
                }
                remote_hf_provenance = Some(hf_read_options.provenance.clone());

                if acquired.payload_format == hf::acquire::HfAcquirePayloadFormat::HfImagefolder
                    && hf_read_options.split.is_some()
                    && (acquired.payload_path.join("metadata.jsonl").is_file()
                        || acquired.payload_path.join("metadata.parquet").is_file())
                {
                    hf_read_options.split = None;
                }

                (
                    acquired.payload_path,
                    args.hf_repo.clone().expect("checked is_some"),
                    remote_payload_to_convert_format(acquired.payload_format),
                )
            }
            #[cfg(not(feature = "hf-remote"))]
            {
                return Err(PanlabelError::UnsupportedFormat(
                    "remote HF import requires the 'hf-remote' feature".to_string(),
                ));
            }
        } else {
            let input = args.input.clone().ok_or_else(|| {
                PanlabelError::UnsupportedFormat("missing required --input <path>".to_string())
            })?;
            let display = input.display().to_string();
            (input, display, from_format)
        };

    // Step 1: Read the dataset
    let mut dataset = if effective_from_format == ConvertFormat::HfImagefolder {
        read_dataset_with_options(effective_from_format, &effective_input, &hf_read_options)?
    } else {
        read_dataset(effective_from_format, &effective_input)?
    };
    if let Some(provenance) = remote_hf_provenance {
        dataset.info.attributes.extend(provenance);
    }

    // Step 2: Optionally validate the input
    if !args.no_validate {
        let opts = validation::ValidateOptions {
            strict: args.strict,
        };
        let validation_report = validation::validate_dataset(&dataset, &opts);

        let has_errors = validation_report.error_count() > 0;
        let has_warnings = validation_report.warning_count() > 0;

        if has_errors || has_warnings {
            eprintln!("{}", validation_report);
        }

        if has_errors || (args.strict && has_warnings) {
            return Err(PanlabelError::ValidationFailed {
                error_count: validation_report.error_count(),
                warning_count: validation_report.warning_count(),
                report: validation_report,
            });
        }
    }

    // Step 3: Build conversion report and check for lossiness
    let conv_report = conversion::build_conversion_report(
        &dataset,
        effective_from_format.to_conversion_format(),
        args.to.to_conversion_format(),
    );

    if conv_report.is_lossy() && !args.allow_lossy {
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(effective_from_format).to_string(),
            to: format_name(args.to).to_string(),
            report: Box::new(conv_report),
        });
    }

    // Step 4: Write the dataset
    write_dataset_with_options(args.to, &args.output, &dataset, &hf_write_options)?;

    // Step 5: Output the report
    match args.report {
        ReportFormat::Text => {
            println!(
                "Converted {} ({}) -> {} ({})",
                source_display,
                format_name(effective_from_format),
                args.output.display(),
                format_name(args.to)
            );
            print!("{}", conv_report);
        }
        ReportFormat::Json => {
            serde_json::to_writer_pretty(std::io::stdout(), &conv_report)
                .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
            println!();
        }
    }

    Ok(())
}

#[cfg(feature = "hf-remote")]
fn remote_payload_to_convert_format(payload: hf::acquire::HfAcquirePayloadFormat) -> ConvertFormat {
    match payload {
        hf::acquire::HfAcquirePayloadFormat::HfImagefolder => ConvertFormat::HfImagefolder,
        hf::acquire::HfAcquirePayloadFormat::Yolo => ConvertFormat::Yolo,
        hf::acquire::HfAcquirePayloadFormat::Voc => ConvertFormat::Voc,
        hf::acquire::HfAcquirePayloadFormat::Coco => ConvertFormat::Coco,
    }
}

fn resolve_from_format(
    from: ConvertFromFormat,
    path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    match from.as_concrete() {
        Some(format) => Ok(format),
        None => detect_format(path),
    }
}

fn resolve_stats_format(
    format: Option<ConvertFormat>,
    path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    if let Some(format) = format {
        return Ok(format);
    }

    match detect_format(path) {
        Ok(format) => Ok(format),
        Err(error) => {
            let is_json_file = path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("json"))
                    .unwrap_or(false);

            if is_json_file {
                Ok(ConvertFormat::IrJson)
            } else {
                Err(error)
            }
        }
    }
}

fn parse_categories_arg(raw: Option<String>) -> Vec<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn load_hf_category_map(
    path: Option<&Path>,
) -> Result<std::collections::BTreeMap<i64, String>, PanlabelError> {
    let Some(path) = path else {
        return Ok(Default::default());
    };

    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::HfLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("invalid JSON in category map: {source}"),
        })?;

    let mut map = std::collections::BTreeMap::new();
    match value {
        serde_json::Value::Object(obj) => {
            for (raw_key, raw_value) in obj {
                let key = raw_key
                    .parse::<i64>()
                    .map_err(|_| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!("category-map key '{}' is not a valid integer", raw_key),
                    })?;
                let label = raw_value
                    .as_str()
                    .ok_or_else(|| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!(
                            "category-map value for key '{}' must be a string",
                            raw_key
                        ),
                    })?;
                map.insert(key, label.to_string());
            }
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.into_iter().enumerate() {
                let label = item
                    .as_str()
                    .ok_or_else(|| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!("category-map array entry {} must be a string", idx),
                    })?;
                map.insert(idx as i64, label.to_string());
            }
        }
        _ => {
            return Err(PanlabelError::HfLayoutInvalid {
                path: path.to_path_buf(),
                message:
                    "category map must be either a JSON object {\"0\":\"person\"} or string array"
                        .to_string(),
            });
        }
    }

    Ok(map)
}

fn validate_hf_flag_usage(
    args: &ConvertArgs,
    from_format: ConvertFormat,
) -> Result<(), PanlabelError> {
    let hf_involved =
        from_format == ConvertFormat::HfImagefolder || args.to == ConvertFormat::HfImagefolder;

    let hf_specific_flags_used = args.hf_repo.is_some()
        || args.hf_objects_column.is_some()
        || args.hf_category_map.is_some()
        || args.split.is_some()
        || args.revision.is_some()
        || args.config.is_some()
        || !matches!(args.hf_bbox_format, HfBboxFormatArg::Xywh);

    if hf_specific_flags_used && !hf_involved {
        return Err(PanlabelError::UnsupportedFormat(
            "HF-specific flags (--hf-*) can only be used with --from hf or --to hf".to_string(),
        ));
    }

    if args.hf_repo.is_some() && from_format != ConvertFormat::HfImagefolder {
        return Err(PanlabelError::UnsupportedFormat(
            "--hf-repo can only be used with --from hf".to_string(),
        ));
    }

    if args.hf_repo.is_none() && (args.revision.is_some() || args.config.is_some()) {
        return Err(PanlabelError::UnsupportedFormat(
            "--revision/--config require --hf-repo".to_string(),
        ));
    }

    if from_format == ConvertFormat::HfImagefolder && args.hf_repo.is_none() && args.input.is_none()
    {
        return Err(PanlabelError::UnsupportedFormat(
            "--from hf requires either --input <path> or --hf-repo <namespace/dataset>".to_string(),
        ));
    }

    Ok(())
}

fn ensure_unique_image_file_names(dataset: &ir::Dataset, side: &str) -> Result<(), PanlabelError> {
    let mut seen = std::collections::HashSet::new();
    for image in &dataset.images {
        if !seen.insert(image.file_name.clone()) {
            return Err(PanlabelError::DiffFailed {
                message: format!(
                    "duplicate image file_name '{}' found in dataset {}. Use unique image names for reliable diffing.",
                    image.file_name, side
                ),
            });
        }
    }
    Ok(())
}

/// Read a dataset from a file in the specified format.
fn read_dataset(format: ConvertFormat, path: &Path) -> Result<ir::Dataset, PanlabelError> {
    read_dataset_with_options(
        format,
        path,
        &ir::io_hf_imagefolder::HfReadOptions::default(),
    )
}

fn read_dataset_with_options(
    format: ConvertFormat,
    path: &Path,
    hf_options: &ir::io_hf_imagefolder::HfReadOptions,
) -> Result<ir::Dataset, PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::read_ir_json(path),
        ConvertFormat::Coco => ir::io_coco_json::read_coco_json(path),
        ConvertFormat::Cvat => ir::io_cvat_xml::read_cvat_xml(path),
        ConvertFormat::LabelStudio => ir::io_label_studio_json::read_label_studio_json(path),
        ConvertFormat::Tfod => ir::io_tfod_csv::read_tfod_csv(path),
        ConvertFormat::Yolo => ir::io_yolo::read_yolo_dir(path),
        ConvertFormat::Voc => ir::io_voc_xml::read_voc_dir(path),
        ConvertFormat::HfImagefolder => read_hf_dataset_with_options(path, hf_options),
    }
}

/// Write a dataset to a file in the specified format.
fn write_dataset(
    format: ConvertFormat,
    path: &Path,
    dataset: &ir::Dataset,
) -> Result<(), PanlabelError> {
    write_dataset_with_options(
        format,
        path,
        dataset,
        &ir::io_hf_imagefolder::HfWriteOptions::default(),
    )
}

fn write_dataset_with_options(
    format: ConvertFormat,
    path: &Path,
    dataset: &ir::Dataset,
    hf_options: &ir::io_hf_imagefolder::HfWriteOptions,
) -> Result<(), PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::write_ir_json(path, dataset),
        ConvertFormat::Coco => ir::io_coco_json::write_coco_json(path, dataset),
        ConvertFormat::Cvat => ir::io_cvat_xml::write_cvat_xml(path, dataset),
        ConvertFormat::LabelStudio => {
            ir::io_label_studio_json::write_label_studio_json(path, dataset)
        }
        ConvertFormat::Tfod => ir::io_tfod_csv::write_tfod_csv(path, dataset),
        ConvertFormat::Yolo => ir::io_yolo::write_yolo_dir(path, dataset),
        ConvertFormat::Voc => ir::io_voc_xml::write_voc_dir(path, dataset),
        ConvertFormat::HfImagefolder => {
            ir::io_hf_imagefolder::write_hf_imagefolder_with_options(path, dataset, hf_options)
        }
    }
}

fn read_hf_dataset_with_default_options(path: &Path) -> Result<ir::Dataset, PanlabelError> {
    read_hf_dataset_with_options(path, &ir::io_hf_imagefolder::HfReadOptions::default())
}

fn read_hf_dataset_with_options(
    path: &Path,
    options: &ir::io_hf_imagefolder::HfReadOptions,
) -> Result<ir::Dataset, PanlabelError> {
    #[cfg(feature = "hf-parquet")]
    {
        if should_read_hf_parquet(path, options.split.as_deref())? {
            return ir::io_hf_parquet::read_hf_parquet_with_options(path, options);
        }
    }

    ir::io_hf_imagefolder::read_hf_imagefolder_with_options(path, options)
}

#[cfg(feature = "hf-parquet")]
fn should_read_hf_parquet(path: &Path, split: Option<&str>) -> Result<bool, PanlabelError> {
    let has_jsonl = hf_has_metadata(path, split, "metadata.jsonl")?;
    let has_parquet_layout =
        hf_has_metadata(path, split, "metadata.parquet")? || hf_has_any_parquet_file(path, split)?;
    Ok(has_parquet_layout && !has_jsonl)
}

#[cfg(feature = "hf-parquet")]
fn hf_has_metadata(
    path: &Path,
    split: Option<&str>,
    metadata_file_name: &str,
) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    if path.join(metadata_file_name).is_file() {
        return Ok(true);
    }

    if let Some(split_name) = split {
        let normalized = normalize_split_hint(split_name);
        return Ok(path.join(&normalized).join(metadata_file_name).is_file());
    }

    for entry in std::fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if entry_path.is_dir() && entry_path.join(metadata_file_name).is_file() {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(feature = "hf-parquet")]
fn hf_has_any_parquet_file(path: &Path, split: Option<&str>) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    let normalized_split = split.map(normalize_split_hint);

    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::HfLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("failed while scanning parquet files: {source}"),
        })?;
        if !entry.file_type().is_file() {
            continue;
        }

        let entry_path = entry.path();
        let is_parquet = entry_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("parquet"))
            .unwrap_or(false);
        if !is_parquet {
            continue;
        }

        if entry_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("metadata.parquet"))
            .unwrap_or(false)
        {
            return Ok(true);
        }

        if let Some(split_name) = normalized_split.as_deref() {
            if parquet_path_matches_split(entry_path, split_name) {
                return Ok(true);
            }
            continue;
        }

        return Ok(true);
    }

    Ok(false)
}

#[cfg(feature = "hf-parquet")]
fn parquet_path_matches_split(path: &Path, split: &str) -> bool {
    let split = normalize_split_hint(split);

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_default();

    if file_name.starts_with(&format!("{split}-")) {
        return true;
    }

    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .map(|value| normalize_split_hint(value) == split)
            .unwrap_or(false)
    })
}

#[cfg(feature = "hf-parquet")]
fn normalize_split_hint(value: &str) -> String {
    match value.to_ascii_lowercase().as_str() {
        "val" | "valid" => "validation".to_string(),
        "validation" => "validation".to_string(),
        "train" => "train".to_string(),
        "test" => "test".to_string(),
        "dev" => "dev".to_string(),
        _ => value.to_ascii_lowercase(),
    }
}

/// Get a human-readable name for a format.
fn format_name(format: ConvertFormat) -> &'static str {
    match format {
        ConvertFormat::IrJson => "ir-json",
        ConvertFormat::Coco => "coco",
        ConvertFormat::Cvat => "cvat",
        ConvertFormat::LabelStudio => "label-studio",
        ConvertFormat::Tfod => "tfod",
        ConvertFormat::Yolo => "yolo",
        ConvertFormat::Voc => "voc",
        ConvertFormat::HfImagefolder => "hf",
    }
}

/// Execute the list-formats subcommand.
fn run_list_formats(_args: ListFormatsArgs) -> Result<(), PanlabelError> {
    use conversion::IrLossiness;

    println!("Supported formats:");
    println!();
    println!(
        "  {:<12} {:<6} {:<6} {:<12} DESCRIPTION",
        "FORMAT", "READ", "WRITE", "LOSSINESS"
    );
    println!(
        "  {:<12} {:<6} {:<6} {:<12} -----------",
        "------", "----", "-----", "---------"
    );

    // Define format info
    let formats = [
        (
            ConvertFormat::IrJson,
            "Panlabel's intermediate representation (JSON)",
        ),
        (ConvertFormat::Coco, "COCO object detection format (JSON)"),
        (ConvertFormat::Cvat, "CVAT for images XML annotation export"),
        (
            ConvertFormat::LabelStudio,
            "Label Studio task export (JSON)",
        ),
        (
            ConvertFormat::Tfod,
            "TensorFlow Object Detection format (CSV)",
        ),
        (
            ConvertFormat::Yolo,
            "Ultralytics YOLO .txt (directory-based)",
        ),
        (ConvertFormat::Voc, "Pascal VOC XML (directory-based)"),
        (
            ConvertFormat::HfImagefolder,
            "Hugging Face ImageFolder metadata (metadata.jsonl/parquet)",
        ),
    ];

    for (fmt, description) in formats {
        let lossiness = fmt.to_conversion_format().lossiness_relative_to_ir();
        let lossiness_str = match lossiness {
            IrLossiness::Lossless => "lossless",
            IrLossiness::Conditional => "conditional",
            IrLossiness::Lossy => "lossy",
        };

        println!(
            "  {:<12} {:<6} {:<6} {:<12} {}",
            format_name(fmt),
            "yes",
            "yes",
            lossiness_str,
            description
        );
    }

    println!();
    println!("Lossiness key:");
    println!("  lossless    - Format preserves all IR information");
    println!("  conditional - Format may lose info depending on dataset content");
    println!("  lossy       - Format always loses some IR information");
    println!();
    println!("Tip: Use '--from auto' with 'convert' for automatic format detection.");

    Ok(())
}

/// Detect the format of an input path based on extension/content (files)
/// or structure (directories).
fn detect_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if path.is_dir() {
        return detect_dir_format(path);
    }

    // First try extension-based detection
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "csv" => return Ok(ConvertFormat::Tfod),
            "json" => return detect_json_format(path),
            "xml" => return detect_xml_format(path),
            _ => {}
        }
    }

    // Keep message stable (existing CLI tests assert this substring).
    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized file extension (expected .json, .csv, or .xml). Use --from to specify format explicitly.".to_string(),
    })
}

fn detect_dir_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let labels_subdir = path.join("labels");
    let is_yolo = (labels_subdir.is_dir() && dir_contains_txt_files(&labels_subdir)?)
        || (is_labels_dir(path) && dir_contains_txt_files(path)?);

    let annotations_subdir = path.join("Annotations");
    let images_subdir = path.join("JPEGImages");
    let is_voc = (annotations_subdir.is_dir()
        && images_subdir.is_dir()
        && dir_contains_xml_files(&annotations_subdir)?)
        || (is_annotations_dir(path)
            && path
                .parent()
                .map(|parent| parent.join("JPEGImages").is_dir())
                .unwrap_or(false)
            && dir_contains_xml_files(path)?);

    let is_cvat = path.join("annotations.xml").is_file();
    let is_hf = dir_contains_hf_metadata(path)?;

    if is_yolo && is_voc {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both YOLO and VOC layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_yolo && is_cvat {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both YOLO and CVAT layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_voc && is_cvat {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both VOC and CVAT layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_hf && is_yolo {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both HF and YOLO layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_hf && is_voc {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both HF and VOC layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_hf && is_cvat {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both HF and CVAT layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_yolo {
        return Ok(ConvertFormat::Yolo);
    }

    if is_voc {
        return Ok(ConvertFormat::Voc);
    }

    if is_cvat {
        return Ok(ConvertFormat::Cvat);
    }

    if is_hf {
        return Ok(ConvertFormat::HfImagefolder);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized directory layout (expected YOLO labels/ with .txt files, VOC Annotations/ with .xml files plus JPEGImages/, CVAT directory export with annotations.xml at root, or HF metadata.jsonl/metadata.parquet layout). Use --from to specify format explicitly.".to_string(),
    })
}

fn dir_contains_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "txt")
}

fn dir_contains_xml_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "xml")
}

fn dir_contains_hf_metadata(path: &Path) -> Result<bool, PanlabelError> {
    if path.join("metadata.jsonl").is_file() || path.join("metadata.parquet").is_file() {
        return Ok(true);
    }

    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir()
            && (entry_path.join("metadata.jsonl").is_file()
                || entry_path.join("metadata.parquet").is_file())
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn dir_contains_extension_files(path: &Path, extension: &str) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;

        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn is_labels_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("labels"))
        .unwrap_or(false)
}

fn is_annotations_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("annotations"))
        .unwrap_or(false)
}

/// Detect whether a JSON file is Label Studio, COCO, or IR JSON format.
///
/// Heuristics:
/// - Array-root JSON: Label Studio task export
/// - Object-root JSON: inspect `annotations[0].bbox`
///   - array of 4 numbers -> COCO
///   - object min/max or xmin/ymin/xmax/ymax -> IR JSON
fn detect_json_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value = serde_json::from_reader(reader).map_err(|source| {
        PanlabelError::FormatDetectionJsonParse {
            path: path.to_path_buf(),
            source,
        }
    })?;

    if let Some(tasks) = value.as_array() {
        if tasks.is_empty() {
            return Ok(ConvertFormat::LabelStudio);
        }

        if is_likely_label_studio_task(&tasks[0]) {
            return Ok(ConvertFormat::LabelStudio);
        }

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "array-root JSON not recognized (expected Label Studio task export array). Use --from to specify format explicitly.".to_string(),
        });
    }

    // Object-root detection remains the same COCO-vs-IR heuristic.

    // Get annotations array
    let annotations = value.get("annotations").and_then(|v| v.as_array());

    let Some(annotations) = annotations else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "missing or invalid 'annotations' array. Cannot determine format.".to_string(),
        });
    };

    if annotations.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "empty 'annotations' array. Cannot determine format from empty dataset. Use --from to specify format explicitly.".to_string(),
        });
    }

    // Inspect the first annotation's bbox
    let first_ann = &annotations[0];
    let bbox = first_ann.get("bbox");

    let Some(bbox) = bbox else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "first annotation has no 'bbox' field. Cannot determine format.".to_string(),
        });
    };

    // Check if bbox is an array (COCO) or object (IR JSON)
    if let Some(arr) = bbox.as_array() {
        // COCO uses [x, y, width, height] - array of 4 numbers
        if arr.len() == 4 && arr.iter().all(|v| v.is_number()) {
            return Ok(ConvertFormat::Coco);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!(
                "bbox is an array but not [x,y,w,h] format (found {} elements). Cannot determine format.",
                arr.len()
            ),
        });
    }

    if let Some(obj) = bbox.as_object() {
        // IR JSON uses {min: {x, y}, max: {x, y}} or {xmin, ymin, xmax, ymax}
        // Check for the serialized format from our bbox.rs
        if obj.contains_key("min") && obj.contains_key("max") {
            return Ok(ConvertFormat::IrJson);
        }
        // Alternative flat format
        if obj.contains_key("xmin")
            && obj.contains_key("ymin")
            && obj.contains_key("xmax")
            && obj.contains_key("ymax")
        {
            return Ok(ConvertFormat::IrJson);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "bbox is an object but doesn't match IR JSON format (expected min/max or xmin/ymin/xmax/ymax). Cannot determine format.".to_string(),
        });
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "bbox has unexpected type (expected array or object). Cannot determine format."
            .to_string(),
    })
}

/// Detect whether an XML file is CVAT XML.
///
/// Heuristic:
/// - root `<annotations>` => CVAT
/// - root `<annotation>` => looks like a single VOC XML (not auto-detected)
fn detect_xml_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let xml = std::fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let doc = roxmltree::Document::parse(&xml).map_err(|source| {
        PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed to parse XML while detecting format: {source}"),
        }
    })?;

    match doc.root_element().tag_name().name() {
        "annotations" => Ok(ConvertFormat::Cvat),
        "annotation" => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "XML root is <annotation> (looks like a single VOC file). Panlabel expects VOC as a directory layout; use --from voc with a VOC dataset directory.".to_string(),
        }),
        other => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("unrecognized XML root element <{other}>; cannot determine format. Use --from to specify format explicitly."),
        }),
    }
}

fn is_likely_label_studio_task(value: &serde_json::Value) -> bool {
    let Some(task_obj) = value.as_object() else {
        return false;
    };

    let Some(data_obj) = task_obj.get("data").and_then(|v| v.as_object()) else {
        return false;
    };

    data_obj
        .get("image")
        .map(|value| value.is_string())
        .unwrap_or(false)
}
