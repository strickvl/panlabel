use std::collections::BTreeSet;
use std::time::Duration;

use serde_json::Value;

use super::HfRepoRef;

/// Best-effort metadata collected from HF dataset viewer APIs.
#[derive(Clone, Debug, Default)]
pub struct HfPreflight {
    pub splits: Vec<String>,
    pub configs: Vec<String>,
    pub features: Option<Value>,
    pub detected_objects_column: Option<String>,
    pub category_labels: Option<Vec<String>>,
    pub license: Option<String>,
    pub description: Option<String>,
    pub selected_split: Option<String>,
}

/// Best-effort preflight. Returns `Ok(None)` when the viewer API is unavailable.
pub fn run_preflight(repo: &HfRepoRef, token: Option<&str>) -> Option<HfPreflight> {
    let info_json = fetch_viewer_json("info", repo, None, token).ok()?;
    let splits_json = fetch_viewer_json("splits", repo, None, token).ok();

    let features = extract_features(&info_json);
    let detected_objects_column = features
        .as_ref()
        .and_then(detect_objects_column_from_features);
    let category_labels = features
        .as_ref()
        .and_then(extract_classlabel_names_from_features);

    let splits = splits_json
        .as_ref()
        .map(extract_split_names)
        .unwrap_or_default();
    let configs = splits_json
        .as_ref()
        .map(extract_config_names)
        .unwrap_or_default();

    let selected_split = if let Some(split) = &repo.split {
        Some(split.clone())
    } else if splits.iter().any(|split| split == "train") {
        Some("train".to_string())
    } else {
        splits.first().cloned()
    };

    let first_rows_json = selected_split
        .as_deref()
        .and_then(|split| fetch_viewer_json("first-rows", repo, Some(split), token).ok());
    let detected_objects_column = detected_objects_column.or_else(|| {
        first_rows_json
            .as_ref()
            .and_then(detect_objects_column_from_first_rows)
    });

    Some(HfPreflight {
        splits,
        configs,
        features,
        detected_objects_column,
        category_labels,
        license: extract_license(&info_json),
        description: extract_description(&info_json),
        selected_split,
    })
}

fn fetch_viewer_json(
    endpoint: &str,
    repo: &HfRepoRef,
    split: Option<&str>,
    token: Option<&str>,
) -> Result<Value, String> {
    let mut url = url::Url::parse(&format!(
        "https://datasets-server.huggingface.co/{}",
        endpoint
    ))
    .map_err(|source| source.to_string())?;

    {
        let mut query = url.query_pairs_mut();
        query.append_pair("dataset", &repo.repo_id);
        if let Some(config) = repo.config.as_deref() {
            query.append_pair("config", config);
        }
        if let Some(split) = split {
            query.append_pair("split", split);
        }
    }

    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .build();
    let agent: ureq::Agent = config.into();

    let mut request = agent.get(url.as_str());
    if let Some(token) = token {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }

    let mut response = request.call().map_err(|source| source.to_string())?;
    response
        .body_mut()
        .read_json::<Value>()
        .map_err(|source| source.to_string())
}

fn extract_features(info_json: &Value) -> Option<Value> {
    info_json
        .get("dataset_info")
        .and_then(|value| value.get("features"))
        .cloned()
        .or_else(|| info_json.get("features").cloned())
}

fn extract_split_names(splits_json: &Value) -> Vec<String> {
    let mut names = BTreeSet::new();
    if let Some(items) = splits_json.get("splits").and_then(Value::as_array) {
        for item in items {
            if let Some(name) = item.get("split").and_then(Value::as_str) {
                names.insert(name.to_string());
            }
        }
    }
    names.into_iter().collect()
}

fn extract_config_names(splits_json: &Value) -> Vec<String> {
    let mut configs = BTreeSet::new();
    if let Some(items) = splits_json.get("splits").and_then(Value::as_array) {
        for item in items {
            if let Some(config) = item.get("config").and_then(Value::as_str) {
                configs.insert(config.to_string());
            }
        }
    }
    configs.into_iter().collect()
}

fn detect_objects_column_from_features(features: &Value) -> Option<String> {
    let object = features.as_object()?;
    if object.contains_key("objects") {
        return Some("objects".to_string());
    }
    if object.contains_key("faces") {
        return Some("faces".to_string());
    }
    None
}

fn detect_objects_column_from_first_rows(first_rows_json: &Value) -> Option<String> {
    let rows = first_rows_json.get("rows")?.as_array()?;
    let first = rows.first()?;
    let row_obj = first.get("row")?.as_object()?;

    if row_obj.contains_key("objects") {
        return Some("objects".to_string());
    }
    if row_obj.contains_key("faces") {
        return Some("faces".to_string());
    }

    None
}

fn extract_classlabel_names_from_features(features: &Value) -> Option<Vec<String>> {
    if let Some(obj) = features.as_object() {
        // First try common object-detection nesting (objects/faces container).
        for container in ["objects", "faces"] {
            if let Some(names) = obj
                .get(container)
                .and_then(find_classlabel_names_in_value)
                .filter(|names| !names.is_empty())
            {
                return Some(names);
            }
        }

        // Then search all top-level feature entries.
        for value in obj.values() {
            if let Some(names) = find_classlabel_names_in_value(value).filter(|v| !v.is_empty()) {
                return Some(names);
            }
        }
    }

    find_classlabel_names_in_value(features)
}

fn find_classlabel_names_in_value(value: &Value) -> Option<Vec<String>> {
    let obj = value.as_object()?;

    if obj
        .get("_type")
        .and_then(Value::as_str)
        .map(|t| t.eq_ignore_ascii_case("ClassLabel"))
        .unwrap_or(false)
    {
        let names = obj
            .get("names")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !names.is_empty() {
            return Some(names);
        }
    }

    for child in obj.values() {
        if let Some(found) = find_classlabel_names_in_value(child).filter(|v| !v.is_empty()) {
            return Some(found);
        }
    }

    if let Some(sequence) = obj.get("feature") {
        return find_classlabel_names_in_value(sequence);
    }

    None
}

fn extract_license(info_json: &Value) -> Option<String> {
    info_json
        .get("dataset_info")
        .and_then(|value| value.get("license"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            info_json
                .get("license")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn extract_description(info_json: &Value) -> Option<String> {
    info_json
        .get("dataset_info")
        .and_then(|value| value.get("description"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            info_json
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classlabel_names_are_detected() {
        let features = serde_json::json!({
            "objects": {
                "bbox": {"feature": {"dtype": "float64"}},
                "categories": {
                    "feature": {
                        "_type": "ClassLabel",
                        "names": ["person", "car"]
                    }
                }
            }
        });

        let labels = extract_classlabel_names_from_features(&features).expect("labels");
        assert_eq!(labels, vec!["person", "car"]);
    }

    #[test]
    fn object_column_detection_prefers_objects() {
        let features = serde_json::json!({"objects": {}, "faces": {}});
        assert_eq!(
            detect_objects_column_from_features(&features).as_deref(),
            Some("objects")
        );
    }

    #[test]
    fn split_names_extracted_deterministically() {
        let response = serde_json::json!({
            "splits": [
                {"split": "validation", "config": "default"},
                {"split": "train", "config": "default"},
                {"split": "test", "config": "other"}
            ]
        });

        assert_eq!(
            extract_split_names(&response),
            vec![
                "test".to_string(),
                "train".to_string(),
                "validation".to_string()
            ]
        );
        assert_eq!(
            extract_config_names(&response),
            vec!["default".to_string(), "other".to_string()]
        );
    }
}
