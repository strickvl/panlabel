#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextFormat {
    TextIrJsonl,
    RlvrHf,
    VerifiersTaskset,
    Harbor,
    SweBench,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextIrLossiness {
    Lossless,
    Conditional,
    Lossy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextFormatCapabilities {
    pub text_input: bool,
    pub message_input: bool,
    pub structured_gold_answer: bool,
    pub answer_match_reward: bool,
    pub executable_reward_artifacts: bool,
    pub runtime_image_workspace: bool,
    pub network_resources_timeouts: bool,
    pub tools_mcp_metadata: bool,
    pub multi_turn: bool,
    pub multi_step: bool,
    pub solution_artifacts: bool,
    pub arbitrary_example_metadata: bool,
    pub dataset_metadata: bool,
    pub external_harness: bool,
}

impl TextFormat {
    pub fn name(self) -> &'static str {
        match self {
            TextFormat::TextIrJsonl => "text-ir-jsonl",
            TextFormat::RlvrHf => "rlvr-hf",
            TextFormat::VerifiersTaskset => "verifiers-taskset",
            TextFormat::Harbor => "harbor",
            TextFormat::SweBench => "swe-bench",
        }
    }

    pub fn lossiness_relative_to_text_ir(self) -> TextIrLossiness {
        match self {
            TextFormat::TextIrJsonl => TextIrLossiness::Lossless,
            TextFormat::RlvrHf => TextIrLossiness::Conditional,
            TextFormat::VerifiersTaskset => TextIrLossiness::Conditional,
            TextFormat::Harbor => TextIrLossiness::Conditional,
            TextFormat::SweBench => TextIrLossiness::Conditional,
        }
    }

    pub fn capabilities(self) -> TextFormatCapabilities {
        match self {
            TextFormat::TextIrJsonl => TextFormatCapabilities {
                text_input: true,
                message_input: true,
                structured_gold_answer: true,
                answer_match_reward: true,
                executable_reward_artifacts: true,
                runtime_image_workspace: true,
                network_resources_timeouts: true,
                tools_mcp_metadata: true,
                multi_turn: true,
                multi_step: true,
                solution_artifacts: true,
                arbitrary_example_metadata: true,
                dataset_metadata: true,
                external_harness: true,
            },
            TextFormat::RlvrHf => TextFormatCapabilities {
                text_input: true,
                structured_gold_answer: true,
                answer_match_reward: true,
                arbitrary_example_metadata: true,
                dataset_metadata: true,
                ..TextFormatCapabilities::default()
            },
            TextFormat::VerifiersTaskset => TextFormatCapabilities {
                text_input: true,
                message_input: true,
                structured_gold_answer: true,
                answer_match_reward: true,
                executable_reward_artifacts: true,
                runtime_image_workspace: true,
                tools_mcp_metadata: true,
                multi_turn: true,
                arbitrary_example_metadata: true,
                dataset_metadata: true,
                external_harness: true,
                ..TextFormatCapabilities::default()
            },
            TextFormat::Harbor => TextFormatCapabilities {
                text_input: true,
                executable_reward_artifacts: true,
                runtime_image_workspace: true,
                network_resources_timeouts: true,
                multi_step: true,
                solution_artifacts: true,
                arbitrary_example_metadata: true,
                dataset_metadata: true,
                external_harness: true,
                ..TextFormatCapabilities::default()
            },
            TextFormat::SweBench => TextFormatCapabilities {
                text_input: true,
                executable_reward_artifacts: true,
                runtime_image_workspace: true,
                solution_artifacts: true,
                arbitrary_example_metadata: true,
                dataset_metadata: true,
                ..TextFormatCapabilities::default()
            },
        }
    }
}

pub fn text_lossiness_name(lossiness: TextIrLossiness) -> &'static str {
    match lossiness {
        TextIrLossiness::Lossless => "lossless",
        TextIrLossiness::Conditional => "conditional",
        TextIrLossiness::Lossy => "lossy",
    }
}

pub struct TextFormatCatalogEntry {
    pub format: TextFormat,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub read: bool,
    pub write: bool,
    pub file_based: bool,
    pub directory_based: bool,
}

pub const TEXT_FORMAT_CATALOG: &[TextFormatCatalogEntry] = &[
    TextFormatCatalogEntry {
        format: TextFormat::TextIrJsonl,
        aliases: &[],
        description: "Canonical Panlabel text/task IR JSONL with metadata sidecar",
        read: true,
        write: true,
        file_based: true,
        directory_based: true,
    },
    TextFormatCatalogEntry {
        format: TextFormat::RlvrHf,
        aliases: &["rlvr", "hf-rlvr"],
        description: "Generic Hugging Face-style JSONL task rows for RLVR datasets",
        read: true,
        write: true,
        file_based: true,
        directory_based: true,
    },
    TextFormatCatalogEntry {
        format: TextFormat::VerifiersTaskset,
        aliases: &["verifiers"],
        description:
            "Materialized Verifiers task rows with reward/environment artifacts by reference",
        read: true,
        write: true,
        file_based: true,
        directory_based: true,
    },
    TextFormatCatalogEntry {
        format: TextFormat::Harbor,
        aliases: &[],
        description:
            "Harbor task directories with instructions, task.toml, tests, and environment files",
        read: true,
        write: true,
        file_based: false,
        directory_based: true,
    },
    TextFormatCatalogEntry {
        format: TextFormat::SweBench,
        aliases: &["swebench"],
        description: "SWE-bench task rows imported as read-only benchmark instances",
        read: true,
        write: false,
        file_based: true,
        directory_based: true,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swe_bench_placeholder_is_not_writable() {
        let swe_bench = TEXT_FORMAT_CATALOG
            .iter()
            .find(|entry| entry.format == TextFormat::SweBench)
            .expect("swe-bench catalog entry");
        assert!(swe_bench.read);
        assert!(!swe_bench.write);
    }

    #[test]
    fn verifiers_alias_is_present() {
        let verifiers = TEXT_FORMAT_CATALOG
            .iter()
            .find(|entry| entry.format == TextFormat::VerifiersTaskset)
            .expect("verifiers catalog entry");
        assert!(verifiers.aliases.contains(&"verifiers"));
    }
}
