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

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Validate(args)) => run_validate(args),
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
