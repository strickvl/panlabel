use crate::{
    conversion_text::{build_text_conversion_report, TaskConversionIssue, TaskConversionIssueCode},
    ir_text::{
        self, ArtifactManifest, ArtifactOutputLayout, HarborWriteOptions, RlvrHfReadOptions,
        TextDataset, TextDatasetBundle, VerifiersTasksetReadOptions,
    },
    list_text_format_entries,
    text_format_catalog::TextFormat,
    text_format_detection,
    validation_text::{self, TextValidateOptions},
    write_json_stdout, OutputContext, PanlabelError, ReportFormat, TextArtifactPolicyArg,
    TextCommands, TextConvertArgs, TextListFormatsArgs, TextValidateArgs,
};

pub(crate) fn run(command: TextCommands, output: OutputContext) -> Result<(), PanlabelError> {
    match command {
        TextCommands::Convert(args) => convert(args, output),
        TextCommands::Validate(args) => validate(args, output),
        TextCommands::ListFormats(args) => list_formats(args, output),
    }
}

fn convert(args: TextConvertArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let from = match args.from.as_concrete() {
        Some(format) => format,
        None => text_format_detection::detect_text_format(&args.input)?,
    };
    let to = args.to.to_text_format();
    let mut bundle = read_text_bundle(
        from,
        &args.input,
        &TextReadOptions {
            split: args.split.clone(),
            prompt_column: args.prompt_column.clone(),
            answer_column: args.answer_column.clone(),
            id_column: args.id_column.clone(),
            split_column: args.split_column.clone(),
        },
    )?;

    let original_artifact_count = ArtifactManifest::from_dataset(&bundle.dataset)
        .entries
        .len();
    if matches!(args.artifact_policy, TextArtifactPolicyArg::Drop) {
        drop_artifacts(&mut bundle.dataset);
    }

    if !args.no_validate {
        let report = validation_text::validate_text_dataset(
            &bundle.dataset,
            Some(&bundle.source_root),
            &TextValidateOptions {
                strict: args.strict,
            },
        );
        let has_errors = report.error_count() > 0;
        let has_warnings = report.warning_count() > 0;
        if has_errors || has_warnings {
            eprintln!("{}", report);
        }
        if has_errors || (args.strict && has_warnings) {
            return Err(PanlabelError::TextValidationFailed {
                error_count: report.error_count(),
                warning_count: report.warning_count(),
                report,
            });
        }
    }

    let compatibility = validation_text::validate_text_target_compatibility(&bundle.dataset, to);
    if !compatibility.is_ok() {
        eprintln!("{}", compatibility);
        return Err(PanlabelError::TextValidationFailed {
            error_count: compatibility.error_count(),
            warning_count: compatibility.warning_count(),
            report: compatibility,
        });
    }

    if to == TextFormat::Harbor
        && !args.scaffold
        && !ir_text::harbor_can_write_without_scaffold(&bundle.dataset)
    {
        return Err(PanlabelError::UnsupportedFormat(
            "writing Harbor requires an executable verifier artifact; pass --scaffold to write a fail-fast placeholder".to_string(),
        ));
    }

    let artifact_manifest = ArtifactManifest::from_dataset(&bundle.dataset);
    let artifact_plan = match (args.artifact_policy, to) {
        (TextArtifactPolicyArg::Copy, TextFormat::Harbor) => None,
        (TextArtifactPolicyArg::Copy, _) => Some(ir_text::build_sidecar_artifact_write_plan(
            &artifact_manifest,
            &bundle.source_root,
            &args.output,
            if to == TextFormat::TextIrJsonl && args.output.extension().is_none() {
                ArtifactOutputLayout::TextIrDirectory
            } else {
                ArtifactOutputLayout::File
            },
        )?),
        (TextArtifactPolicyArg::Reference | TextArtifactPolicyArg::Drop, _) => None,
    };
    let mut output_dataset = bundle.dataset.clone();
    if let Some(plan) = &artifact_plan {
        ir_text::apply_artifact_write_plan_to_dataset(&mut output_dataset, plan);
    }

    let mut conv_report = build_text_conversion_report(
        &output_dataset,
        &ArtifactManifest::from_dataset(&output_dataset),
        from,
        to,
    );
    if matches!(args.artifact_policy, TextArtifactPolicyArg::Drop) && original_artifact_count > 0 {
        conv_report.add(TaskConversionIssue::warning(
            TaskConversionIssueCode::TaskRewardDropped,
            format!(
                "{} artifact reference(s) were dropped because --artifact-policy drop was used",
                original_artifact_count
            ),
        ));
    }
    if to == TextFormat::Harbor
        && args.scaffold
        && !ir_text::harbor_can_write_without_scaffold(&bundle.dataset)
    {
        conv_report.add(TaskConversionIssue::warning(
            TaskConversionIssueCode::TaskRewardStub,
            "Harbor output will include a fail-fast scaffold verifier that must be replaced"
                .to_string(),
        ));
    }
    if let Some(plan) = &artifact_plan {
        let destinations = plan
            .entries
            .iter()
            .filter_map(|entry| entry.record_ref.as_deref())
            .collect::<Vec<_>>();
        if !destinations.is_empty() {
            conv_report.add(TaskConversionIssue::writer_info(
                TaskConversionIssueCode::TaskArtifactPreserved,
                format!("artifact destinations: {}", destinations.join(", ")),
            ));
        }
    }
    if from == TextFormat::SweBench {
        conv_report.add(TaskConversionIssue::writer_info(
            TaskConversionIssueCode::TaskSwebBenchPatchPreserved,
            "SWE-bench patch and test_patch fields are preserved as patch artifacts, not prompt text".to_string(),
        ));
    }

    if conv_report.is_lossy() && !args.allow_lossy {
        emit_text_conversion_report(&conv_report, args.output_format, output)?;
        return Err(PanlabelError::TextLossyConversionBlocked {
            from: from.name().to_string(),
            to: to.name().to_string(),
            report: Box::new(conv_report),
        });
    }

    if !args.dry_run {
        write_text_dataset(
            to,
            &args.output,
            &output_dataset,
            args.scaffold,
            Some(&bundle.source_root),
        )?;
        if to != TextFormat::Harbor {
            if let Some(plan) = &artifact_plan {
                ir_text::copy_artifacts(plan)?;
            }
        }
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
                args.input.display(),
                from.name(),
                args.output.display(),
                to.name()
            );
            emit_text_conversion_report(&conv_report, ReportFormat::Text, output)?;
        }
        ReportFormat::Json => {
            emit_text_conversion_report(&conv_report, ReportFormat::Json, output)?
        }
    }

    Ok(())
}

fn validate(args: TextValidateArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let format = match args.format.as_concrete() {
        Some(format) => format,
        None => text_format_detection::detect_text_format(&args.input)?,
    };
    let bundle = read_text_bundle(format, &args.input, &TextReadOptions::default())?;
    let report = validation_text::validate_text_dataset(
        &bundle.dataset,
        Some(&bundle.source_root),
        &TextValidateOptions {
            strict: args.strict,
        },
    );
    match args.output_format {
        ReportFormat::Text => print!("{}", report),
        ReportFormat::Json => write_json_stdout(&report.as_json(), output)?,
    }
    if report.error_count() > 0 || (args.strict && report.warning_count() > 0) {
        return Err(PanlabelError::TextValidationFailed {
            error_count: report.error_count(),
            warning_count: report.warning_count(),
            report,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct TextReadOptions {
    split: Option<String>,
    prompt_column: Option<String>,
    answer_column: Option<String>,
    id_column: Option<String>,
    split_column: Option<String>,
}

fn read_text_bundle(
    format: TextFormat,
    path: &std::path::Path,
    options: &TextReadOptions,
) -> Result<TextDatasetBundle, PanlabelError> {
    let dataset = match format {
        TextFormat::TextIrJsonl => ir_text::read_text_ir_jsonl(path)?,
        TextFormat::RlvrHf => ir_text::read_rlvr_hf(
            path,
            &RlvrHfReadOptions {
                prompt_column: options.prompt_column.clone(),
                answer_column: options.answer_column.clone(),
                id_column: options.id_column.clone(),
                split_column: options.split_column.clone(),
                split: options.split.clone(),
            },
        )?,
        TextFormat::VerifiersTaskset => ir_text::read_verifiers_taskset(
            path,
            &VerifiersTasksetReadOptions {
                split: options.split.clone(),
            },
        )?,
        TextFormat::Harbor => ir_text::read_harbor(path)?,
        TextFormat::SweBench => ir_text::read_swe_bench(path)?,
    };
    let source_root = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    };
    Ok(TextDatasetBundle::new(dataset, source_root))
}

fn write_text_dataset(
    format: TextFormat,
    path: &std::path::Path,
    dataset: &TextDataset,
    scaffold: bool,
    source_root: Option<&std::path::Path>,
) -> Result<(), PanlabelError> {
    match format {
        TextFormat::TextIrJsonl => ir_text::write_text_ir_jsonl(path, dataset),
        TextFormat::RlvrHf => ir_text::write_rlvr_hf(path, dataset),
        TextFormat::VerifiersTaskset => ir_text::write_verifiers_taskset(path, dataset),
        TextFormat::Harbor => ir_text::write_harbor(
            path,
            dataset,
            &HarborWriteOptions {
                scaffold,
                source_root: source_root.map(std::path::Path::to_path_buf),
            },
        ),
        TextFormat::SweBench => Err(PanlabelError::UnsupportedFormat(
            "swe-bench is read-only in this implementation pass".to_string(),
        )),
    }
}

fn drop_artifacts(dataset: &mut TextDataset) {
    for example in &mut dataset.examples {
        example.data.artifacts.clear();
        if let Some(reward) = &mut example.data.reward {
            reward.artifacts.clear();
        }
        if let Some(runtime) = &mut example.data.runtime {
            runtime.artifacts.clear();
        }
        if let Some(harness) = &mut example.data.harness {
            harness.artifacts.clear();
        }
        if let Some(solution) = &mut example.data.solution {
            solution.artifacts.clear();
        }
    }
}

fn emit_text_conversion_report(
    report: &crate::conversion_text::TextConversionReport,
    format: ReportFormat,
    output: OutputContext,
) -> Result<(), PanlabelError> {
    match format {
        ReportFormat::Text => print!("{}", report),
        ReportFormat::Json => write_json_stdout(report, output)?,
    }
    Ok(())
}

fn list_formats(args: TextListFormatsArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let entries = list_text_format_entries();

    match args.output_format {
        ReportFormat::Text => {
            println!("Supported text/task formats:");
            println!();
            println!(
                "  {:<18} {:<6} {:<6} {:<12} DESCRIPTION",
                "FORMAT", "READ", "WRITE", "LOSSINESS"
            );
            println!(
                "  {:<18} {:<6} {:<6} {:<12} -----------",
                "------", "----", "-----", "---------"
            );

            for entry in &entries {
                println!(
                    "  {:<18} {:<6} {:<6} {:<12} {}",
                    entry.name,
                    if entry.read { "yes" } else { "no" },
                    if entry.write { "yes" } else { "no" },
                    entry.lossiness,
                    entry.description
                );
            }

            println!();
            println!("Task formats live under 'panlabel text ...'.");
        }
        ReportFormat::Json => write_json_stdout(&entries, output)?,
    }

    Ok(())
}
