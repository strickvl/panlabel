pub mod artifact;
pub(crate) mod io_common;
pub mod io_harbor;
pub mod io_rlvr_hf;
pub mod io_swe_bench;
pub mod io_text_ir_jsonl;
pub mod io_verifiers_taskset;
pub mod model;

pub use artifact::{
    apply_artifact_write_plan_to_dataset, artifact_refs_for_example,
    build_sidecar_artifact_write_plan, copy_artifacts, materialize_artifact_to_target,
    safe_example_ids, sanitize_example_id, validate_relative_artifact_path, ArtifactDisposition,
    ArtifactManifest, ArtifactManifestEntry, ArtifactOutputLayout, ArtifactWritePlan,
    ArtifactWritePlanEntry, TextDatasetBundle,
};
pub use io_harbor::{
    harbor_can_write_without_scaffold, read_harbor, write_harbor, HarborWriteOptions,
};
pub use io_rlvr_hf::{read_rlvr_hf, write_rlvr_hf, RlvrHfReadOptions};
pub use io_swe_bench::read_swe_bench;
pub use io_text_ir_jsonl::{read_text_ir_jsonl, write_text_ir_jsonl, TextIrJsonlWriteOptions};
pub use io_verifiers_taskset::{
    read_verifiers_taskset, write_verifiers_taskset, VerifiersTasksetReadOptions,
};
pub use model::{
    ArtifactKind, ArtifactRef, Block, HarnessDescriptor, Message, Metadata, RewardDescriptor,
    RewardKind, Role, RuntimeDescriptor, SolutionDescriptor, TaskData, TaskInput, TaskStep,
    TextDataset, TextDatasetInfo, TextExample, TextExampleKind,
};
