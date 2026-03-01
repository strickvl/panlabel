//! Hugging Face Hub orchestration helpers.
//!
//! This module owns remote-specific concerns (repo resolution, preflight, and
//! acquisition). Pure file parsing stays in `crate::ir::io_hf_*`.

pub mod acquire;
pub mod preflight;
pub mod resolve;

/// Canonical reference to a Hugging Face dataset repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfRepoRef {
    pub repo_id: String,
    pub revision: Option<String>,
    pub config: Option<String>,
    pub split: Option<String>,
}
