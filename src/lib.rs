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
//! - [`error`]: Error types for panlabel operations

pub mod error;
pub mod ir;
pub mod validation;

use std::collections::HashSet;
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
    /// TensorFlow Object Detection format (CSV).
    #[value(name = "tfod", alias = "tfod-csv")]
    Tfod,
}

/// Arguments for the validate subcommand.
#[derive(clap::Args)]
struct ValidateArgs {
    /// Input file to validate.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', or 'tfod').
    #[arg(long, default_value = "ir-json")]
    format: String,

    /// Treat warnings as errors (exit non-zero if any warnings).
    #[arg(long)]
    strict: bool,

    /// Output format for the report ('text' or 'json').
    #[arg(long, default_value = "text")]
    output: String,
}

/// Arguments for the convert subcommand.
#[derive(clap::Args)]
struct ConvertArgs {
    /// Source format.
    #[arg(short = 'f', long = "from", value_enum)]
    from: ConvertFormat,

    /// Target format.
    #[arg(short = 't', long = "to", value_enum)]
    to: ConvertFormat,

    /// Input file path.
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output file path.
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
}

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Validate(args)) => run_validate(args),
        Some(Commands::Convert(args)) => run_convert(args),
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

/// Execute the validate subcommand.
fn run_validate(args: ValidateArgs) -> Result<(), PanlabelError> {
    // Load the dataset based on format
    let dataset = match args.format.as_str() {
        "ir-json" => ir::io_json::read_ir_json(&args.input)?,
        "coco" | "coco-json" => ir::io_coco_json::read_coco_json(&args.input)?,
        "tfod" | "tfod-csv" => ir::io_tfod_csv::read_tfod_csv(&args.input)?,
        other => {
            return Err(PanlabelError::UnsupportedFormat(format!(
                "'{}' (supported: ir-json, coco, tfod)",
                other
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
            // Simple JSON output for programmatic use
            println!("{{");
            println!("  \"error_count\": {},", report.error_count());
            println!("  \"warning_count\": {},", report.warning_count());
            println!("  \"issues\": [");
            for (i, issue) in report.issues.iter().enumerate() {
                let comma = if i < report.issues.len() - 1 { "," } else { "" };
                println!("    {{");
                println!("      \"severity\": \"{:?}\",", issue.severity);
                println!("      \"code\": \"{:?}\",", issue.code);
                println!(
                    "      \"message\": \"{}\",",
                    issue.message.replace('"', "\\\"")
                );
                println!("      \"context\": \"{}\"", issue.context);
                println!("    }}{}", comma);
            }
            println!("  ]");
            println!("}}");
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
    // Step 1: Read the dataset
    let dataset = read_dataset(args.from, &args.input)?;

    // Step 2: Optionally validate the input
    if !args.no_validate {
        let opts = validation::ValidateOptions {
            strict: args.strict,
        };
        let report = validation::validate_dataset(&dataset, &opts);

        let has_errors = report.error_count() > 0;
        let has_warnings = report.warning_count() > 0;

        // Print validation issues if any
        if has_errors || has_warnings {
            eprintln!("{}", report);
        }

        if has_errors || (args.strict && has_warnings) {
            return Err(PanlabelError::ValidationFailed {
                error_count: report.error_count(),
                warning_count: report.warning_count(),
                report,
            });
        }
    }

    // Step 3: Check for lossiness
    let reasons = lossy_reasons(&dataset, args.to);
    if !reasons.is_empty() && !args.allow_lossy {
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(args.from).to_string(),
            to: format_name(args.to).to_string(),
            reasons,
        });
    }

    // Print warnings if lossy but allowed
    if !reasons.is_empty() {
        eprintln!(
            "Warning: Lossy conversion from {} to {}:",
            format_name(args.from),
            format_name(args.to)
        );
        for reason in &reasons {
            eprintln!("  - {}", reason);
        }
    }

    // Step 4: Write the dataset
    write_dataset(args.to, &args.output, &dataset)?;

    // Success message
    println!(
        "Converted {} ({}) -> {} ({})",
        args.input.display(),
        format_name(args.from),
        args.output.display(),
        format_name(args.to)
    );
    println!(
        "  {} images, {} categories, {} annotations",
        dataset.images.len(),
        dataset.categories.len(),
        dataset.annotations.len()
    );

    Ok(())
}

/// Read a dataset from a file in the specified format.
fn read_dataset(format: ConvertFormat, path: &Path) -> Result<ir::Dataset, PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::read_ir_json(path),
        ConvertFormat::Coco => ir::io_coco_json::read_coco_json(path),
        ConvertFormat::Tfod => ir::io_tfod_csv::read_tfod_csv(path),
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
        ConvertFormat::Tfod => ir::io_tfod_csv::write_tfod_csv(path, dataset),
    }
}

/// Get a human-readable name for a format.
fn format_name(format: ConvertFormat) -> &'static str {
    match format {
        ConvertFormat::IrJson => "ir-json",
        ConvertFormat::Coco => "coco",
        ConvertFormat::Tfod => "tfod",
    }
}

/// Compute reasons why a conversion would be lossy for the given target format.
fn lossy_reasons(dataset: &ir::Dataset, to: ConvertFormat) -> Vec<String> {
    let mut reasons = Vec::new();

    match to {
        ConvertFormat::Tfod => {
            // TFOD CSV cannot represent:
            // - Dataset info/metadata
            if !dataset.info.is_empty() {
                reasons.push("dataset info/metadata will be dropped".to_string());
            }
            // - Licenses
            if !dataset.licenses.is_empty() {
                reasons.push(format!(
                    "{} license(s) will be dropped",
                    dataset.licenses.len()
                ));
            }
            // - Image license_id and date_captured
            let images_with_metadata = dataset
                .images
                .iter()
                .filter(|img| img.license_id.is_some() || img.date_captured.is_some())
                .count();
            if images_with_metadata > 0 {
                reasons.push(format!(
                    "{} image(s) have license_id/date_captured that will be dropped",
                    images_with_metadata
                ));
            }
            // - Category supercategory
            let cats_with_supercategory = dataset
                .categories
                .iter()
                .filter(|cat| cat.supercategory.is_some())
                .count();
            if cats_with_supercategory > 0 {
                reasons.push(format!(
                    "{} category(s) have supercategory that will be dropped",
                    cats_with_supercategory
                ));
            }
            // - Annotation confidence
            let anns_with_confidence = dataset
                .annotations
                .iter()
                .filter(|ann| ann.confidence.is_some())
                .count();
            if anns_with_confidence > 0 {
                reasons.push(format!(
                    "{} annotation(s) have confidence scores that will be dropped",
                    anns_with_confidence
                ));
            }
            // - Annotation attributes
            let anns_with_attributes = dataset
                .annotations
                .iter()
                .filter(|ann| !ann.attributes.is_empty())
                .count();
            if anns_with_attributes > 0 {
                reasons.push(format!(
                    "{} annotation(s) have attributes that will be dropped",
                    anns_with_attributes
                ));
            }
            // - Images with no annotations (will not appear in output)
            let image_ids_with_annotations: HashSet<_> =
                dataset.annotations.iter().map(|a| a.image_id).collect();
            let images_without_annotations = dataset
                .images
                .iter()
                .filter(|img| !image_ids_with_annotations.contains(&img.id))
                .count();
            if images_without_annotations > 0 {
                reasons.push(format!(
                    "{} image(s) have no annotations and will not appear in output",
                    images_without_annotations
                ));
            }
        }
        ConvertFormat::Coco => {
            // COCO doesn't have a dataset name field (only info.description, version, etc.)
            if dataset.info.name.is_some() {
                reasons.push("dataset info.name has no COCO equivalent".to_string());
            }
            // COCO round-trips area/iscrowd via attributes, but other attributes may be lost
            // depending on COCO tools consuming the output
            let anns_with_other_attributes = dataset
                .annotations
                .iter()
                .filter(|ann| ann.attributes.keys().any(|k| k != "area" && k != "iscrowd"))
                .count();
            if anns_with_other_attributes > 0 {
                reasons.push(format!(
                    "{} annotation(s) have attributes (other than area/iscrowd) that may not be preserved by COCO tools",
                    anns_with_other_attributes
                ));
            }
        }
        ConvertFormat::IrJson => {
            // IR JSON is the canonical format - no lossiness
        }
    }

    reasons
}
