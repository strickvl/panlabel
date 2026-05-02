use crate::{
    list_format_entries, write_json_stdout, ListFormatsArgs, OutputContext, PanlabelError,
    ReportFormat,
};

/// Execute the list-formats subcommand.
pub(crate) fn run(args: ListFormatsArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let entries = list_format_entries();

    match args.output_format {
        ReportFormat::Text => {
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

            for entry in &entries {
                println!(
                    "  {:<12} {:<6} {:<6} {:<12} {}",
                    entry.name,
                    if entry.read { "yes" } else { "no" },
                    if entry.write { "yes" } else { "no" },
                    entry.lossiness,
                    entry.description
                );
            }

            println!();
            println!("Lossiness key:");
            println!("  lossless    - Format preserves all IR information");
            println!("  conditional - Format may lose info depending on dataset content");
            println!("  lossy       - Format always loses some IR information");
            println!();
            println!("Tip: Use '--from auto' with 'convert' for automatic format detection.");
        }
        ReportFormat::Json => write_json_stdout(&entries, output)?,
    }

    Ok(())
}
