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
pub mod error;
pub mod inspect;
pub mod ir;
pub mod validation;

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
    /// Inspect a dataset and display summary statistics.
    Inspect(InspectArgs),
    /// List supported formats and their capabilities.
    ListFormats(ListFormatsArgs),
}

/// Supported formats for conversion.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ConvertFormat {
    /// Panlabel's intermediate representation (JSON).
    #[value(name = "ir-json")]
    IrJson,
    /// COCO object detection format (JSON).
    #[value(name = "coco", alias = "coco-json")]
    Coco,
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
}

impl ConvertFormat {
    /// Convert CLI format to conversion module format.
    fn to_conversion_format(self) -> conversion::Format {
        match self {
            ConvertFormat::IrJson => conversion::Format::IrJson,
            ConvertFormat::Coco => conversion::Format::Coco,
            ConvertFormat::LabelStudio => conversion::Format::LabelStudio,
            ConvertFormat::Tfod => conversion::Format::Tfod,
            ConvertFormat::Yolo => conversion::Format::Yolo,
            ConvertFormat::Voc => conversion::Format::Voc,
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
}

impl ConvertFromFormat {
    /// Convert to a concrete format, returning None for Auto.
    fn as_concrete(self) -> Option<ConvertFormat> {
        match self {
            ConvertFromFormat::Auto => None,
            ConvertFromFormat::IrJson => Some(ConvertFormat::IrJson),
            ConvertFromFormat::Coco => Some(ConvertFormat::Coco),
            ConvertFromFormat::LabelStudio => Some(ConvertFormat::LabelStudio),
            ConvertFromFormat::Tfod => Some(ConvertFormat::Tfod),
            ConvertFromFormat::Yolo => Some(ConvertFormat::Yolo),
            ConvertFromFormat::Voc => Some(ConvertFormat::Voc),
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

/// Arguments for the validate subcommand.
#[derive(clap::Args)]
struct ValidateArgs {
    /// Input path to validate.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', 'label-studio', 'tfod', 'yolo', or 'voc').
    #[arg(long, default_value = "ir-json")]
    format: String,

    /// Treat warnings as errors (exit non-zero if any warnings).
    #[arg(long)]
    strict: bool,

    /// Output format for the report ('text' or 'json').
    #[arg(long, default_value = "text")]
    output: String,
}

/// Arguments for the inspect subcommand.
#[derive(clap::Args)]
struct InspectArgs {
    /// Input path to inspect.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', 'label-studio', 'tfod', 'yolo', or 'voc').
    #[arg(long, value_enum, default_value = "ir-json")]
    format: ConvertFormat,

    /// Number of top labels to show in the histogram.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Tolerance in pixels for out-of-bounds checks.
    #[arg(long, default_value_t = 0.5)]
    tolerance: f64,
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

    /// Input path.
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

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
}

/// Arguments for the list-formats subcommand.
#[derive(clap::Args)]
struct ListFormatsArgs {}

const SUPPORTED_VALIDATE_FORMATS: &str = "ir-json, coco, label-studio, tfod, yolo, voc";

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Validate(args)) => run_validate(args),
        Some(Commands::Convert(args)) => run_convert(args),
        Some(Commands::Inspect(args)) => run_inspect(args),
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

/// Execute the inspect subcommand.
fn run_inspect(args: InspectArgs) -> Result<(), PanlabelError> {
    let dataset = read_dataset(args.format, &args.input)?;

    let opts = inspect::InspectOptions {
        top_labels: args.top,
        oob_tolerance_px: args.tolerance,
        bar_width: 20,
    };

    let report = inspect::inspect_dataset(&dataset, &opts);

    print!("{}", report);
    Ok(())
}

/// Execute the validate subcommand.
fn run_validate(args: ValidateArgs) -> Result<(), PanlabelError> {
    // Load the dataset based on format
    let dataset = match args.format.as_str() {
        "ir-json" => ir::io_json::read_ir_json(&args.input)?,
        "coco" | "coco-json" => ir::io_coco_json::read_coco_json(&args.input)?,
        "label-studio" | "label-studio-json" | "ls" => {
            ir::io_label_studio_json::read_label_studio_json(&args.input)?
        }
        "tfod" | "tfod-csv" => ir::io_tfod_csv::read_tfod_csv(&args.input)?,
        "yolo" | "ultralytics" | "yolov8" | "yolov5" => ir::io_yolo::read_yolo_dir(&args.input)?,
        "voc" | "pascal-voc" | "voc-xml" => ir::io_voc_xml::read_voc_dir(&args.input)?,
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
    // Step 0: Resolve auto-detection if needed
    let from_format: ConvertFormat = match args.from.as_concrete() {
        Some(f) => f,
        None => detect_format(&args.input)?,
    };

    // Step 1: Read the dataset
    let dataset = read_dataset(from_format, &args.input)?;

    // Step 2: Optionally validate the input
    if !args.no_validate {
        let opts = validation::ValidateOptions {
            strict: args.strict,
        };
        let validation_report = validation::validate_dataset(&dataset, &opts);

        let has_errors = validation_report.error_count() > 0;
        let has_warnings = validation_report.warning_count() > 0;

        // Print validation issues if any
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
        from_format.to_conversion_format(),
        args.to.to_conversion_format(),
    );

    if conv_report.is_lossy() && !args.allow_lossy {
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(from_format).to_string(),
            to: format_name(args.to).to_string(),
            report: Box::new(conv_report),
        });
    }

    // Step 4: Write the dataset
    write_dataset(args.to, &args.output, &dataset)?;

    // Step 5: Output the report
    match args.report {
        ReportFormat::Text => {
            // Success message with conversion report
            println!(
                "Converted {} ({}) -> {} ({})",
                args.input.display(),
                format_name(from_format),
                args.output.display(),
                format_name(args.to)
            );
            print!("{}", conv_report);
        }
        ReportFormat::Json => {
            // JSON-only output for machine consumption
            serde_json::to_writer_pretty(std::io::stdout(), &conv_report)
                .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
            println!(); // trailing newline
        }
    }

    Ok(())
}

/// Read a dataset from a file in the specified format.
fn read_dataset(format: ConvertFormat, path: &Path) -> Result<ir::Dataset, PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::read_ir_json(path),
        ConvertFormat::Coco => ir::io_coco_json::read_coco_json(path),
        ConvertFormat::LabelStudio => ir::io_label_studio_json::read_label_studio_json(path),
        ConvertFormat::Tfod => ir::io_tfod_csv::read_tfod_csv(path),
        ConvertFormat::Yolo => ir::io_yolo::read_yolo_dir(path),
        ConvertFormat::Voc => ir::io_voc_xml::read_voc_dir(path),
    }
}

/// Write a dataset to a file in the specified format.
fn write_dataset(
    format: ConvertFormat,
    path: &Path,
    dataset: &ir::Dataset,
) -> Result<(), PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::write_ir_json(path, dataset),
        ConvertFormat::Coco => ir::io_coco_json::write_coco_json(path, dataset),
        ConvertFormat::LabelStudio => {
            ir::io_label_studio_json::write_label_studio_json(path, dataset)
        }
        ConvertFormat::Tfod => ir::io_tfod_csv::write_tfod_csv(path, dataset),
        ConvertFormat::Yolo => ir::io_yolo::write_yolo_dir(path, dataset),
        ConvertFormat::Voc => ir::io_voc_xml::write_voc_dir(path, dataset),
    }
}

/// Get a human-readable name for a format.
fn format_name(format: ConvertFormat) -> &'static str {
    match format {
        ConvertFormat::IrJson => "ir-json",
        ConvertFormat::Coco => "coco",
        ConvertFormat::LabelStudio => "label-studio",
        ConvertFormat::Tfod => "tfod",
        ConvertFormat::Yolo => "yolo",
        ConvertFormat::Voc => "voc",
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
            _ => {}
        }
    }

    // Keep message stable (existing CLI tests assert this substring).
    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized file extension (expected .json or .csv). Use --from to specify format explicitly.".to_string(),
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

    if is_yolo && is_voc {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "directory matches both YOLO and VOC layouts. Use --from to specify format explicitly."
                .to_string(),
        });
    }

    if is_yolo {
        return Ok(ConvertFormat::Yolo);
    }

    if is_voc {
        return Ok(ConvertFormat::Voc);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized directory layout (expected YOLO labels/ with .txt files or VOC Annotations/ with .xml files plus JPEGImages/). Use --from to specify format explicitly.".to_string(),
    })
}

fn dir_contains_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "txt")
}

fn dir_contains_xml_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "xml")
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
