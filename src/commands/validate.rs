use crate::{
    read_dataset, validation, write_json_stdout, OutputContext, PanlabelError, ReportFormat,
    ValidateArgs,
};

/// Execute the validate subcommand.
pub(crate) fn run(args: ValidateArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let dataset = read_dataset(args.format, &args.input)?;

    let opts = validation::ValidateOptions {
        strict: args.strict,
    };
    let report = validation::validate_dataset(&dataset, &opts);

    match args.output_format {
        ReportFormat::Json => write_json_stdout(&report.as_json(), output)?,
        ReportFormat::Text => print!("{}", report),
    }

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
