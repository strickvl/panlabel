use crate::{
    read_dataset, resolve_stats_format, write_json_stdout, OutputContext, PanlabelError, StatsArgs,
    StatsOutputFormat,
};

/// Execute the stats subcommand.
pub(crate) fn run(args: StatsArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let format = resolve_stats_format(args.format, &args.input)?;
    let dataset = read_dataset(format, &args.input)?;

    let opts = crate::stats::StatsOptions {
        top_labels: args.top,
        top_pairs: args.top,
        oob_tolerance_px: args.tolerance,
        bar_width: 20,
    };

    let report = crate::stats::stats_dataset(&dataset, &opts);

    match args.output_format {
        StatsOutputFormat::Text => print!("{}", report.display(output.stats_text_style())),
        StatsOutputFormat::Json => write_json_stdout(&report, output)?,
        StatsOutputFormat::Html => {
            let html = crate::stats::html::render_html(&report)?;
            print!("{html}");
        }
    }

    Ok(())
}
