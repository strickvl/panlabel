use crate::{
    conversion, emit_conversion_report, format_detection, format_name, ir, load_hf_category_map,
    read_dataset, read_dataset_with_options, validate_hf_flag_usage, validation,
    write_dataset_with_options, ConvertArgs, ConvertFormat, OutputContext, PanlabelError,
    ReportFormat,
};

#[cfg(feature = "hf-remote")]
use crate::{
    hf::{
        acquire::{self, HfAcquirePayloadFormat},
        preflight, resolve,
    },
    remote_payload_to_convert_format,
};

/// Execute the convert subcommand.
pub(crate) fn run(args: ConvertArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let from_format = match args.from.as_concrete() {
        Some(format) => format,
        None => {
            let input = args.input.as_ref().ok_or_else(|| {
                PanlabelError::UnsupportedFormat("--from auto requires --input <path>".to_string())
            })?;
            format_detection::detect_format(input)?
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

    let (effective_input, source_display, effective_from_format) = if from_format
        == ConvertFormat::HfImagefolder
        && args.hf_repo.is_some()
    {
        #[cfg(feature = "hf-remote")]
        {
            let repo_input = args.hf_repo.as_deref().expect("checked is_some");
            let repo_ref = resolve::parse_hf_input(
                repo_input,
                args.revision.as_deref(),
                args.config.as_deref(),
                args.split.as_deref(),
            )?;

            let preflight = preflight::run_preflight(&repo_ref, args.token.as_deref());
            if preflight.is_none() {
                eprintln!("Note: HF viewer API unavailable; proceeding with direct download.");
            }

            if let Some(preflight_data) = preflight.as_ref() {
                if hf_read_options.objects_column.is_none() {
                    hf_read_options.objects_column = preflight_data.detected_objects_column.clone();
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

            let acquired = acquire::acquire(&repo_ref, preflight.as_ref(), args.token.as_deref())?;
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

            if acquired.payload_format == HfAcquirePayloadFormat::HfImagefolder
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

    let yolo_read_options = ir::io_yolo::YoloReadOptions {
        split: args.split.clone(),
    };
    let mut dataset = if effective_from_format == ConvertFormat::HfImagefolder
        || effective_from_format == ConvertFormat::Yolo
    {
        read_dataset_with_options(
            effective_from_format,
            &effective_input,
            &hf_read_options,
            &yolo_read_options,
        )?
    } else {
        read_dataset(effective_from_format, &effective_input)?
    };
    if let Some(provenance) = remote_hf_provenance {
        dataset.info.attributes.extend(provenance);
    }

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

    let conv_report = conversion::build_conversion_report(
        &dataset,
        effective_from_format.to_conversion_format(),
        args.to.to_conversion_format(),
    );

    if conv_report.is_lossy() && !args.allow_lossy {
        emit_conversion_report(&conv_report, args.output_format, output)?;
        return Err(PanlabelError::LossyConversionBlocked {
            from: format_name(effective_from_format).to_string(),
            to: format_name(args.to).to_string(),
            report: Box::new(conv_report),
        });
    }

    if !args.dry_run {
        write_dataset_with_options(args.to, &args.output, &dataset, &hf_write_options)?;
    }

    match args.output_format {
        ReportFormat::Text => {
            println!(
                "{} {} ({}) -> {} ({})",
                if args.dry_run {
                    "Dry run: would convert"
                } else {
                    "Converted"
                },
                source_display,
                format_name(effective_from_format),
                args.output.display(),
                format_name(args.to)
            );
            emit_conversion_report(&conv_report, ReportFormat::Text, output)?;
        }
        ReportFormat::Json => {
            emit_conversion_report(&conv_report, ReportFormat::Json, output)?;
        }
    }

    Ok(())
}
