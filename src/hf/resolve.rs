use crate::error::PanlabelError;

use super::HfRepoRef;

/// Parse a user-supplied HF dataset reference (repo ID or dataset URL).
pub fn parse_hf_input(
    input: &str,
    revision: Option<&str>,
    config: Option<&str>,
    split: Option<&str>,
) -> Result<HfRepoRef, PanlabelError> {
    let (repo_id, revision_from_url) =
        if input.starts_with("http://") || input.starts_with("https://") {
            parse_repo_id_from_url(input)?
        } else {
            (validate_repo_id(input)?, None)
        };

    let merged_revision = match (revision, revision_from_url) {
        (Some(arg), Some(url_rev)) if arg != url_rev => {
            return Err(PanlabelError::HfResolveError {
                input: input.to_string(),
                message: format!(
                    "conflicting revisions: --revision='{}' but URL encodes revision='{}'",
                    arg, url_rev
                ),
            });
        }
        (Some(arg), _) => Some(arg.to_string()),
        (None, Some(url_rev)) => Some(url_rev.to_string()),
        (None, None) => None,
    };

    Ok(HfRepoRef {
        repo_id,
        revision: merged_revision,
        config: config.map(str::to_string),
        split: split.map(str::to_string),
    })
}

fn parse_repo_id_from_url(input: &str) -> Result<(String, Option<String>), PanlabelError> {
    let url = url::Url::parse(input).map_err(|source| PanlabelError::HfResolveError {
        input: input.to_string(),
        message: format!("invalid URL: {source}"),
    })?;

    let host = url
        .host_str()
        .ok_or_else(|| PanlabelError::HfResolveError {
            input: input.to_string(),
            message: "URL is missing a host".to_string(),
        })?
        .to_ascii_lowercase();

    if host != "huggingface.co" {
        return Err(PanlabelError::HfResolveError {
            input: input.to_string(),
            message: format!("expected host 'huggingface.co', found '{}'", host),
        });
    }

    let segments: Vec<&str> = url
        .path_segments()
        .map(|iter| iter.filter(|seg| !seg.is_empty()).collect())
        .unwrap_or_default();

    if segments.len() < 3 || segments[0] != "datasets" {
        return Err(PanlabelError::HfResolveError {
            input: input.to_string(),
            message:
                "expected dataset URL like https://huggingface.co/datasets/<namespace>/<dataset>"
                    .to_string(),
        });
    }

    let namespace = segments[1];
    let dataset = segments[2];
    let repo_id = validate_repo_id(&format!("{namespace}/{dataset}"))?;

    let revision = if segments.get(3) == Some(&"tree") {
        segments.get(4).map(|value| (*value).to_string())
    } else {
        None
    };

    Ok((repo_id, revision))
}

fn validate_repo_id(repo_id: &str) -> Result<String, PanlabelError> {
    let trimmed = repo_id.trim();
    let mut parts = trimmed.split('/');
    let namespace = parts.next().unwrap_or_default();
    let dataset = parts.next().unwrap_or_default();
    let extra = parts.next();

    if namespace.is_empty() || dataset.is_empty() || extra.is_some() {
        return Err(PanlabelError::HfResolveError {
            input: repo_id.to_string(),
            message: "expected repo id in '<namespace>/<dataset>' form".to_string(),
        });
    }

    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_repo_id_input() {
        let parsed = parse_hf_input("org/dataset", None, None, Some("train")).expect("parse");
        assert_eq!(parsed.repo_id, "org/dataset");
        assert_eq!(parsed.split.as_deref(), Some("train"));
        assert_eq!(parsed.revision, None);
    }

    #[test]
    fn parse_dataset_url_input() {
        let parsed = parse_hf_input(
            "https://huggingface.co/datasets/org/dataset",
            None,
            None,
            None,
        )
        .expect("parse");
        assert_eq!(parsed.repo_id, "org/dataset");
    }

    #[test]
    fn parse_dataset_url_tree_revision() {
        let parsed = parse_hf_input(
            "https://huggingface.co/datasets/org/dataset/tree/release-v1",
            None,
            None,
            None,
        )
        .expect("parse");
        assert_eq!(parsed.repo_id, "org/dataset");
        assert_eq!(parsed.revision.as_deref(), Some("release-v1"));
    }

    #[test]
    fn revision_conflict_is_error() {
        let err = parse_hf_input(
            "https://huggingface.co/datasets/org/dataset/tree/main",
            Some("dev"),
            None,
            None,
        )
        .expect_err("should fail");

        match err {
            PanlabelError::HfResolveError { message, .. } => {
                assert!(message.contains("conflicting revisions"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
