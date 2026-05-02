use crate::{
    conversion, emit_conversion_report, format_name, parse_categories_arg, read_dataset,
    resolve_from_format, sample as sample_engine, write_dataset, CategoryModeArg, ConvertFormat,
    OutputContext, PanlabelError, ReportFormat, SampleArgs, SampleStrategyArg,
};

/// Execute the sample subcommand.
pub(crate) fn run(args: SampleArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let from_format = resolve_from_format(args.from, &args.input)?;
    let to_format = match args.to {
        Some(target) => target,
        None => args.from.as_concrete().unwrap_or(ConvertFormat::IrJson),
    };

    let dataset = read_dataset(from_format, &args.input)?;

    let strategy = match args.strategy {
        SampleStrategyArg::Random => sample_engine::SampleStrategy::Random,
        SampleStrategyArg::Stratified => sample_engine::SampleStrategy::Stratified,
    };
    let category_mode = match args.category_mode {
        CategoryModeArg::Images => sample_engine::CategoryMode::Images,
        CategoryModeArg::Annotations => sample_engine::CategoryMode::Annotations,
    };

    let sample_opts = sample_engine::SampleOptions {
        n: args.n,
        fraction: args.fraction,
        seed: args.seed,
        strategy,
        categories: parse_categories_arg(args.categories),
        category_mode,
    };

    let sampled_dataset = sample_engine::sample_dataset(&dataset, &sample_opts)?;

    let conv_report = conversion::build_conversion_report(
        &sampled_dataset,
        from_format.to_conversion_format(),
        to_format.to_conversion_format(),
    );

    if conv_report.is_lossy() && !args.allow_lossy {
        emit_conversion_report(&conv_report, args.output_format, output)?;
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(from_format).to_string(),
            to: format_name(to_format).to_string(),
            report: Box::new(conv_report),
        });
    }

    if !args.dry_run {
        write_dataset(to_format, &args.output, &sampled_dataset)?;
    }

    match args.output_format {
        ReportFormat::Text => {
            println!(
                "{} {} images -> {} images: {} ({}) -> {} ({})",
                if args.dry_run {
                    "Dry run: would sample"
                } else {
                    "Sampled"
                },
                dataset.images.len(),
                sampled_dataset.images.len(),
                args.input.display(),
                format_name(from_format),
                args.output.display(),
                format_name(to_format)
            );
            emit_conversion_report(&conv_report, ReportFormat::Text, output)?;
        }
        ReportFormat::Json => {
            emit_conversion_report(&conv_report, ReportFormat::Json, output)?;
        }
    }

    Ok(())
}
