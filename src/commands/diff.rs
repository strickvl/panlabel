use crate::{
    ensure_unique_image_file_names, read_dataset, resolve_from_format, write_json_stdout, DiffArgs,
    DiffMatchBy, OutputContext, PanlabelError, ReportFormat,
};

/// Execute the diff subcommand.
pub(crate) fn run(args: DiffArgs, output: OutputContext) -> Result<(), PanlabelError> {
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
        DiffMatchBy::Id => crate::diff::MatchBy::Id,
        DiffMatchBy::Iou => crate::diff::MatchBy::Iou,
    };

    let opts = crate::diff::DiffOptions {
        match_by,
        iou_threshold: args.iou_threshold,
        detail: args.detail,
        max_items: 20,
        bbox_eps: 1e-6,
    };

    let report = crate::diff::diff_datasets(&dataset_a, &dataset_b, &opts);

    match args.output_format {
        ReportFormat::Text => {
            println!(
                "Dataset Diff: {} vs {}",
                args.input_a.display(),
                args.input_b.display()
            );
            println!();
            print!("{}", report);
        }
        ReportFormat::Json => write_json_stdout(&report, output)?,
    }

    Ok(())
}
