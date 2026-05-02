//! Panlabel: The universal annotation converter.
//!
//! Panlabel converts between different object detection annotation formats,
//! similar to how Pandoc converts between document formats. It uses an
//! intermediate representation (IR) to enable N×M format conversions with
//! only 2N converters.
//!
//! # Modules
//!
//! - [`ir`]: Intermediate representation types (Dataset, Image, Annotation, etc.)
//! - [`validation`]: Dataset validation and error reporting
//! - [`conversion`]: Conversion reporting and lossiness tracking
//! - [`error`]: Error types for panlabel operations

mod commands;

pub mod conversion;
pub mod diff;
pub mod error;
pub mod format_catalog;
pub(crate) mod format_detection;
#[cfg(feature = "hf-remote")]
pub mod hf;
pub mod ir;
pub mod sample;
pub mod stats;
pub mod validation;

use std::fs::File;
use std::io::{BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

pub use error::PanlabelError;

/// The panlabel CLI application.
#[derive(Parser)]
#[command(name = "panlabel")]
#[command(version, author, about)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Validate a dataset for errors and warnings.
    Validate(ValidateArgs),
    /// Convert a dataset between formats.
    Convert(ConvertArgs),
    /// Show rich dataset statistics.
    Stats(StatsArgs),
    /// Compare two datasets semantically.
    Diff(DiffArgs),
    /// Sample a subset dataset.
    Sample(SampleArgs),
    /// List supported formats and their capabilities.
    ListFormats(ListFormatsArgs),
}

/// Supported formats for conversion.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum ConvertFormat {
    /// Panlabel's intermediate representation (JSON).
    #[value(name = "ir-json")]
    IrJson,
    /// COCO object detection format (JSON).
    #[value(name = "coco", alias = "coco-json")]
    Coco,
    /// IBM Cloud Annotations localization JSON (`_annotations.json`).
    #[value(
        name = "ibm-cloud-annotations",
        alias = "cloud-annotations",
        alias = "cloud-annotations-json",
        alias = "ibm-cloud-annotations-json"
    )]
    IbmCloudAnnotations,
    /// CVAT for images task export (XML).
    #[value(name = "cvat", alias = "cvat-xml")]
    Cvat,
    /// Label Studio task export (JSON).
    #[value(name = "label-studio", alias = "label-studio-json", alias = "ls")]
    LabelStudio,
    /// Labelbox current export rows (JSON/NDJSON).
    #[value(name = "labelbox", alias = "labelbox-json", alias = "labelbox-ndjson")]
    Labelbox,
    /// Scale AI image annotation task/response JSON.
    #[value(name = "scale-ai", alias = "scale", alias = "scale-ai-json")]
    ScaleAi,
    /// Unity Perception / SOLO JSON dataset.
    #[value(
        name = "unity-perception",
        alias = "unity",
        alias = "unity-perception-json",
        alias = "solo"
    )]
    UnityPerception,
    /// TensorFlow Object Detection format (CSV).
    #[value(name = "tfod", alias = "tfod-csv")]
    Tfod,
    /// TensorFlow Object Detection API TFRecord Examples.
    #[value(
        name = "tfrecord",
        alias = "tfrecords",
        alias = "tf-record",
        alias = "tfod-tfrecord",
        // Intentional typo-tolerant alias for a common doubled-"re" input mistake.
        alias = "tfod-tfrerecord"
    )]
    Tfrecord,
    /// Microsoft VoTT CSV export.
    #[value(name = "vott-csv", alias = "vott")]
    VottCsv,
    /// Microsoft VoTT JSON export.
    #[value(name = "vott-json", alias = "vott-json-export")]
    VottJson,
    /// Ultralytics-style YOLO object detection format (directory-based).
    #[value(
        name = "yolo",
        alias = "ultralytics",
        alias = "yolov8",
        alias = "yolov5",
        alias = "scaled-yolov4",
        alias = "scaled-yolov4-txt"
    )]
    Yolo,
    /// YOLO Keras absolute-coordinate TXT format.
    #[value(name = "yolo-keras", alias = "yolo-keras-txt", alias = "keras-yolo")]
    YoloKeras,
    /// YOLOv4 PyTorch absolute-coordinate TXT format.
    #[value(
        name = "yolov4-pytorch",
        alias = "yolov4-pytorch-txt",
        alias = "pytorch-yolov4"
    )]
    YoloV4Pytorch,
    /// Pascal VOC XML format (directory-based).
    #[value(name = "voc", alias = "pascal-voc", alias = "voc-xml")]
    Voc,
    /// Hugging Face ImageFolder metadata format (directory-based).
    #[value(name = "hf", alias = "hf-imagefolder", alias = "huggingface")]
    HfImagefolder,
    /// AWS SageMaker Ground Truth object-detection manifest (JSON Lines).
    #[value(
        name = "sagemaker",
        alias = "sagemaker-manifest",
        alias = "sagemaker-ground-truth",
        alias = "ground-truth",
        alias = "groundtruth",
        alias = "aws-sagemaker"
    )]
    SageMaker,
    /// LabelMe annotation format (per-image JSON, directory-based).
    #[value(name = "labelme", alias = "labelme-json")]
    LabelMe,
    /// SuperAnnotate JSON annotation format.
    #[value(name = "superannotate", alias = "superannotate-json", alias = "sa")]
    SuperAnnotate,
    /// Supervisely JSON annotation/project format.
    #[value(name = "supervisely", alias = "supervisely-json", alias = "sly")]
    Supervisely,
    /// Cityscapes polygon JSON annotation format.
    #[value(name = "cityscapes", alias = "cityscapes-json")]
    Cityscapes,
    /// Marmot XML document-layout annotation format.
    #[value(name = "marmot", alias = "marmot-xml")]
    Marmot,
    /// Apple CreateML annotation format (JSON).
    #[value(name = "create-ml", alias = "createml", alias = "create-ml-json")]
    CreateMl,
    /// KITTI object detection label files (directory-based).
    #[value(name = "kitti", alias = "kitti-txt")]
    Kitti,
    /// VGG Image Annotator JSON format.
    #[value(name = "via", alias = "via-json", alias = "vgg-via")]
    Via,
    /// keras-retinanet CSV format.
    #[value(name = "retinanet", alias = "retinanet-csv", alias = "keras-retinanet")]
    Retinanet,
    /// Google OpenImages CSV annotation format.
    #[value(name = "openimages", alias = "openimages-csv", alias = "open-images")]
    OpenImages,
    /// Datumaro JSON annotation format.
    #[value(name = "datumaro", alias = "datumaro-json", alias = "datumaro-dataset")]
    Datumaro,
    /// WIDER Face aggregate TXT annotation format.
    #[value(name = "wider-face", alias = "widerface", alias = "wider-face-txt")]
    WiderFace,
    /// OIDv4 Toolkit TXT label format.
    #[value(
        name = "oidv4",
        alias = "oidv4-txt",
        alias = "openimages-v4-txt",
        alias = "oid"
    )]
    Oidv4,
    /// BDD100K / Scalabel JSON detection format.
    #[value(
        name = "bdd100k",
        alias = "bdd100k-json",
        alias = "scalabel",
        alias = "scalabel-json"
    )]
    Bdd100k,
    /// V7 Darwin JSON annotation format.
    #[value(
        name = "v7-darwin",
        alias = "darwin",
        alias = "darwin-json",
        alias = "v7"
    )]
    V7Darwin,
    /// Edge Impulse bounding_boxes.labels format.
    #[value(
        name = "edge-impulse",
        alias = "edge-impulse-labels",
        alias = "edge-impulse-bounding-boxes",
        alias = "bounding-boxes-labels"
    )]
    EdgeImpulse,
    /// ASAM OpenLABEL JSON 2D bbox subset.
    #[value(
        name = "openlabel",
        alias = "asam-openlabel",
        alias = "openlabel-json",
        alias = "asam-openlabel-json"
    )]
    OpenLabel,
    /// VGG Image Annotator CSV format.
    #[value(name = "via-csv", alias = "vgg-via-csv")]
    ViaCsv,
    /// Kaggle Global Wheat Detection CSV format.
    #[value(name = "kaggle-wheat", alias = "kaggle-wheat-csv")]
    KaggleWheat,
    /// Google Cloud AutoML Vision CSV format.
    #[value(
        name = "automl-vision",
        alias = "automl-vision-csv",
        alias = "google-cloud-automl"
    )]
    AutoMlVision,
    /// Udacity Self-Driving Car Dataset CSV format.
    #[value(name = "udacity", alias = "udacity-csv", alias = "self-driving-car")]
    Udacity,
}

impl ConvertFormat {
    /// Convert CLI format to conversion module format.
    fn to_conversion_format(self) -> conversion::Format {
        match self {
            ConvertFormat::IrJson => conversion::Format::IrJson,
            ConvertFormat::Coco => conversion::Format::Coco,
            ConvertFormat::IbmCloudAnnotations => conversion::Format::IbmCloudAnnotations,
            ConvertFormat::Cvat => conversion::Format::Cvat,
            ConvertFormat::LabelStudio => conversion::Format::LabelStudio,
            ConvertFormat::Labelbox => conversion::Format::Labelbox,
            ConvertFormat::ScaleAi => conversion::Format::ScaleAi,
            ConvertFormat::UnityPerception => conversion::Format::UnityPerception,
            ConvertFormat::Tfod => conversion::Format::Tfod,
            ConvertFormat::Tfrecord => conversion::Format::Tfrecord,
            ConvertFormat::VottCsv => conversion::Format::VottCsv,
            ConvertFormat::VottJson => conversion::Format::VottJson,
            ConvertFormat::Yolo => conversion::Format::Yolo,
            ConvertFormat::YoloKeras => conversion::Format::YoloKeras,
            ConvertFormat::YoloV4Pytorch => conversion::Format::YoloV4Pytorch,
            ConvertFormat::Voc => conversion::Format::Voc,
            ConvertFormat::HfImagefolder => conversion::Format::HfImagefolder,
            ConvertFormat::SageMaker => conversion::Format::SageMaker,
            ConvertFormat::LabelMe => conversion::Format::LabelMe,
            ConvertFormat::SuperAnnotate => conversion::Format::SuperAnnotate,
            ConvertFormat::Supervisely => conversion::Format::Supervisely,
            ConvertFormat::Cityscapes => conversion::Format::Cityscapes,
            ConvertFormat::Marmot => conversion::Format::Marmot,
            ConvertFormat::CreateMl => conversion::Format::CreateMl,
            ConvertFormat::Kitti => conversion::Format::Kitti,
            ConvertFormat::Via => conversion::Format::Via,
            ConvertFormat::Retinanet => conversion::Format::Retinanet,
            ConvertFormat::OpenImages => conversion::Format::OpenImages,
            ConvertFormat::Datumaro => conversion::Format::Datumaro,
            ConvertFormat::WiderFace => conversion::Format::WiderFace,
            ConvertFormat::Oidv4 => conversion::Format::Oidv4,
            ConvertFormat::Bdd100k => conversion::Format::Bdd100k,
            ConvertFormat::V7Darwin => conversion::Format::V7Darwin,
            ConvertFormat::EdgeImpulse => conversion::Format::EdgeImpulse,
            ConvertFormat::OpenLabel => conversion::Format::OpenLabel,
            ConvertFormat::ViaCsv => conversion::Format::ViaCsv,
            ConvertFormat::KaggleWheat => conversion::Format::KaggleWheat,
            ConvertFormat::AutoMlVision => conversion::Format::AutoMlVision,
            ConvertFormat::Udacity => conversion::Format::Udacity,
        }
    }
}

/// Source format for conversion (allows 'auto' for detection).
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ConvertFromFormat {
    /// Auto-detect format from input path.
    #[value(name = "auto")]
    Auto,
    /// Panlabel's intermediate representation (JSON).
    #[value(name = "ir-json")]
    IrJson,
    /// COCO object detection format (JSON).
    #[value(name = "coco", alias = "coco-json")]
    Coco,
    /// IBM Cloud Annotations localization JSON (`_annotations.json`).
    #[value(
        name = "ibm-cloud-annotations",
        alias = "cloud-annotations",
        alias = "cloud-annotations-json",
        alias = "ibm-cloud-annotations-json"
    )]
    IbmCloudAnnotations,
    /// CVAT for images task export (XML).
    #[value(name = "cvat", alias = "cvat-xml")]
    Cvat,
    /// Label Studio task export (JSON).
    #[value(name = "label-studio", alias = "label-studio-json", alias = "ls")]
    LabelStudio,
    /// Labelbox current export rows (JSON/NDJSON).
    #[value(name = "labelbox", alias = "labelbox-json", alias = "labelbox-ndjson")]
    Labelbox,
    /// Scale AI image annotation task/response JSON.
    #[value(name = "scale-ai", alias = "scale", alias = "scale-ai-json")]
    ScaleAi,
    /// Unity Perception / SOLO JSON dataset.
    #[value(
        name = "unity-perception",
        alias = "unity",
        alias = "unity-perception-json",
        alias = "solo"
    )]
    UnityPerception,
    /// TensorFlow Object Detection format (CSV).
    #[value(name = "tfod", alias = "tfod-csv")]
    Tfod,
    /// TensorFlow Object Detection API TFRecord Examples.
    #[value(
        name = "tfrecord",
        alias = "tfrecords",
        alias = "tf-record",
        alias = "tfod-tfrecord",
        // Intentional typo-tolerant alias for a common doubled-"re" input mistake.
        alias = "tfod-tfrerecord"
    )]
    Tfrecord,
    /// Microsoft VoTT CSV export.
    #[value(name = "vott-csv", alias = "vott")]
    VottCsv,
    /// Microsoft VoTT JSON export.
    #[value(name = "vott-json", alias = "vott-json-export")]
    VottJson,
    /// Ultralytics-style YOLO object detection format (directory-based).
    #[value(
        name = "yolo",
        alias = "ultralytics",
        alias = "yolov8",
        alias = "yolov5",
        alias = "scaled-yolov4",
        alias = "scaled-yolov4-txt"
    )]
    Yolo,
    /// YOLO Keras absolute-coordinate TXT format.
    #[value(name = "yolo-keras", alias = "yolo-keras-txt", alias = "keras-yolo")]
    YoloKeras,
    /// YOLOv4 PyTorch absolute-coordinate TXT format.
    #[value(
        name = "yolov4-pytorch",
        alias = "yolov4-pytorch-txt",
        alias = "pytorch-yolov4"
    )]
    YoloV4Pytorch,
    /// Pascal VOC XML format (directory-based).
    #[value(name = "voc", alias = "pascal-voc", alias = "voc-xml")]
    Voc,
    /// Hugging Face ImageFolder metadata format (directory-based).
    #[value(name = "hf", alias = "hf-imagefolder", alias = "huggingface")]
    HfImagefolder,
    /// AWS SageMaker Ground Truth object-detection manifest (JSON Lines).
    #[value(
        name = "sagemaker",
        alias = "sagemaker-manifest",
        alias = "sagemaker-ground-truth",
        alias = "ground-truth",
        alias = "groundtruth",
        alias = "aws-sagemaker"
    )]
    SageMaker,
    /// LabelMe annotation format (per-image JSON, directory-based).
    #[value(name = "labelme", alias = "labelme-json")]
    LabelMe,
    /// SuperAnnotate JSON annotation format.
    #[value(name = "superannotate", alias = "superannotate-json", alias = "sa")]
    SuperAnnotate,
    /// Supervisely JSON annotation/project format.
    #[value(name = "supervisely", alias = "supervisely-json", alias = "sly")]
    Supervisely,
    /// Cityscapes polygon JSON annotation format.
    #[value(name = "cityscapes", alias = "cityscapes-json")]
    Cityscapes,
    /// Marmot XML document-layout annotation format.
    #[value(name = "marmot", alias = "marmot-xml")]
    Marmot,
    /// Apple CreateML annotation format (JSON).
    #[value(name = "create-ml", alias = "createml", alias = "create-ml-json")]
    CreateMl,
    /// KITTI object detection label files (directory-based).
    #[value(name = "kitti", alias = "kitti-txt")]
    Kitti,
    /// VGG Image Annotator JSON format.
    #[value(name = "via", alias = "via-json", alias = "vgg-via")]
    Via,
    /// keras-retinanet CSV format.
    #[value(name = "retinanet", alias = "retinanet-csv", alias = "keras-retinanet")]
    Retinanet,
    /// Google OpenImages CSV annotation format.
    #[value(name = "openimages", alias = "openimages-csv", alias = "open-images")]
    OpenImages,
    /// Datumaro JSON annotation format.
    #[value(name = "datumaro", alias = "datumaro-json", alias = "datumaro-dataset")]
    Datumaro,
    /// WIDER Face aggregate TXT annotation format.
    #[value(name = "wider-face", alias = "widerface", alias = "wider-face-txt")]
    WiderFace,
    /// OIDv4 Toolkit TXT label format.
    #[value(
        name = "oidv4",
        alias = "oidv4-txt",
        alias = "openimages-v4-txt",
        alias = "oid"
    )]
    Oidv4,
    /// BDD100K / Scalabel JSON detection format.
    #[value(
        name = "bdd100k",
        alias = "bdd100k-json",
        alias = "scalabel",
        alias = "scalabel-json"
    )]
    Bdd100k,
    /// V7 Darwin JSON annotation format.
    #[value(
        name = "v7-darwin",
        alias = "darwin",
        alias = "darwin-json",
        alias = "v7"
    )]
    V7Darwin,
    /// Edge Impulse bounding_boxes.labels format.
    #[value(
        name = "edge-impulse",
        alias = "edge-impulse-labels",
        alias = "edge-impulse-bounding-boxes",
        alias = "bounding-boxes-labels"
    )]
    EdgeImpulse,
    /// ASAM OpenLABEL JSON 2D bbox subset.
    #[value(
        name = "openlabel",
        alias = "asam-openlabel",
        alias = "openlabel-json",
        alias = "asam-openlabel-json"
    )]
    OpenLabel,
    /// VGG Image Annotator CSV format.
    #[value(name = "via-csv", alias = "vgg-via-csv")]
    ViaCsv,
    /// Kaggle Global Wheat Detection CSV format.
    #[value(name = "kaggle-wheat", alias = "kaggle-wheat-csv")]
    KaggleWheat,
    /// Google Cloud AutoML Vision CSV format.
    #[value(
        name = "automl-vision",
        alias = "automl-vision-csv",
        alias = "google-cloud-automl"
    )]
    AutoMlVision,
    /// Udacity Self-Driving Car Dataset CSV format.
    #[value(name = "udacity", alias = "udacity-csv", alias = "self-driving-car")]
    Udacity,
}

impl ConvertFromFormat {
    /// Convert to a concrete format, returning None for Auto.
    fn as_concrete(self) -> Option<ConvertFormat> {
        match self {
            ConvertFromFormat::Auto => None,
            ConvertFromFormat::IrJson => Some(ConvertFormat::IrJson),
            ConvertFromFormat::Coco => Some(ConvertFormat::Coco),
            ConvertFromFormat::IbmCloudAnnotations => Some(ConvertFormat::IbmCloudAnnotations),
            ConvertFromFormat::Cvat => Some(ConvertFormat::Cvat),
            ConvertFromFormat::LabelStudio => Some(ConvertFormat::LabelStudio),
            ConvertFromFormat::Labelbox => Some(ConvertFormat::Labelbox),
            ConvertFromFormat::ScaleAi => Some(ConvertFormat::ScaleAi),
            ConvertFromFormat::UnityPerception => Some(ConvertFormat::UnityPerception),
            ConvertFromFormat::Tfod => Some(ConvertFormat::Tfod),
            ConvertFromFormat::Tfrecord => Some(ConvertFormat::Tfrecord),
            ConvertFromFormat::VottCsv => Some(ConvertFormat::VottCsv),
            ConvertFromFormat::VottJson => Some(ConvertFormat::VottJson),
            ConvertFromFormat::Yolo => Some(ConvertFormat::Yolo),
            ConvertFromFormat::YoloKeras => Some(ConvertFormat::YoloKeras),
            ConvertFromFormat::YoloV4Pytorch => Some(ConvertFormat::YoloV4Pytorch),
            ConvertFromFormat::Voc => Some(ConvertFormat::Voc),
            ConvertFromFormat::HfImagefolder => Some(ConvertFormat::HfImagefolder),
            ConvertFromFormat::SageMaker => Some(ConvertFormat::SageMaker),
            ConvertFromFormat::LabelMe => Some(ConvertFormat::LabelMe),
            ConvertFromFormat::SuperAnnotate => Some(ConvertFormat::SuperAnnotate),
            ConvertFromFormat::Supervisely => Some(ConvertFormat::Supervisely),
            ConvertFromFormat::Cityscapes => Some(ConvertFormat::Cityscapes),
            ConvertFromFormat::Marmot => Some(ConvertFormat::Marmot),
            ConvertFromFormat::CreateMl => Some(ConvertFormat::CreateMl),
            ConvertFromFormat::Kitti => Some(ConvertFormat::Kitti),
            ConvertFromFormat::Via => Some(ConvertFormat::Via),
            ConvertFromFormat::Retinanet => Some(ConvertFormat::Retinanet),
            ConvertFromFormat::OpenImages => Some(ConvertFormat::OpenImages),
            ConvertFromFormat::Datumaro => Some(ConvertFormat::Datumaro),
            ConvertFromFormat::WiderFace => Some(ConvertFormat::WiderFace),
            ConvertFromFormat::Oidv4 => Some(ConvertFormat::Oidv4),
            ConvertFromFormat::Bdd100k => Some(ConvertFormat::Bdd100k),
            ConvertFromFormat::V7Darwin => Some(ConvertFormat::V7Darwin),
            ConvertFromFormat::EdgeImpulse => Some(ConvertFormat::EdgeImpulse),
            ConvertFromFormat::OpenLabel => Some(ConvertFormat::OpenLabel),
            ConvertFromFormat::ViaCsv => Some(ConvertFormat::ViaCsv),
            ConvertFromFormat::KaggleWheat => Some(ConvertFormat::KaggleWheat),
            ConvertFromFormat::AutoMlVision => Some(ConvertFormat::AutoMlVision),
            ConvertFromFormat::Udacity => Some(ConvertFormat::Udacity),
        }
    }
}

/// Output format for conversion reports.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum ReportFormat {
    /// Human-readable text output.
    #[default]
    #[value(name = "text")]
    Text,
    /// Machine-readable JSON output.
    #[value(name = "json")]
    Json,
}

/// Output format for stats reports.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum StatsOutputFormat {
    /// Human-readable text output.
    #[default]
    #[value(name = "text")]
    Text,
    /// Machine-readable JSON output.
    #[value(name = "json")]
    Json,
    /// Self-contained HTML report.
    #[value(name = "html")]
    Html,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum JsonStyle {
    Pretty,
    Compact,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct OutputContext {
    stdout_is_terminal: bool,
}

impl OutputContext {
    fn detect() -> Self {
        Self {
            stdout_is_terminal: std::io::stdout().is_terminal(),
        }
    }

    fn json_style(self) -> JsonStyle {
        if self.stdout_is_terminal {
            JsonStyle::Pretty
        } else {
            JsonStyle::Compact
        }
    }

    fn stats_text_style(self) -> stats::TextReportStyle {
        if self.stdout_is_terminal {
            stats::TextReportStyle::Rich
        } else {
            stats::TextReportStyle::Plain
        }
    }
}

/// Annotation matching strategy for dataset diff.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum DiffMatchBy {
    /// Match annotations by ID.
    #[default]
    #[value(name = "id")]
    Id,
    /// Match annotations by IoU.
    #[value(name = "iou")]
    Iou,
}

/// Image sampling strategy.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum SampleStrategyArg {
    /// Uniform random sampling.
    #[default]
    #[value(name = "random")]
    Random,
    /// Category-aware stratified sampling.
    #[value(name = "stratified")]
    Stratified,
}

/// Category filter mode.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum CategoryModeArg {
    /// Keep whole images that contain at least one selected category.
    #[default]
    #[value(name = "images")]
    Images,
    /// Keep only selected-category annotations.
    #[value(name = "annotations")]
    Annotations,
}

/// HF bbox format interpretation.
#[derive(Copy, Clone, Debug, Default, ValueEnum)]
enum HfBboxFormatArg {
    /// `[x, y, width, height]`
    #[default]
    #[value(name = "xywh")]
    Xywh,
    /// `[x1, y1, x2, y2]`
    #[value(name = "xyxy")]
    Xyxy,
}

impl HfBboxFormatArg {
    fn to_hf_bbox_format(self) -> ir::io_hf_imagefolder::HfBboxFormat {
        match self {
            HfBboxFormatArg::Xywh => ir::io_hf_imagefolder::HfBboxFormat::Xywh,
            HfBboxFormatArg::Xyxy => ir::io_hf_imagefolder::HfBboxFormat::Xyxy,
        }
    }
}

/// Arguments for the validate subcommand.
#[derive(clap::Args)]
pub(crate) struct ValidateArgs {
    /// Input path to validate.
    input: PathBuf,

    /// Input format.
    #[arg(long, value_enum, default_value_t = ConvertFormat::IrJson)]
    format: ConvertFormat,

    /// Treat warnings as errors (exit non-zero if any warnings).
    #[arg(long)]
    strict: bool,

    /// Output format for the report.
    #[arg(
        long = "output-format",
        visible_alias = "output",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,
}

/// Arguments for the stats subcommand.
#[derive(clap::Args)]
pub(crate) struct StatsArgs {
    /// Input path to analyze.
    input: PathBuf,

    /// Input format ('ir-json', 'coco', 'cvat', 'label-studio', 'tfod', 'tfrecord', 'yolo', 'voc', or 'hf').
    ///
    /// If omitted, panlabel auto-detects the format. If detection fails for a JSON
    /// file, stats falls back to reading as ir-json.
    #[arg(long, value_enum)]
    format: Option<ConvertFormat>,

    /// Number of top labels / pairs to show.
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Tolerance in pixels for out-of-bounds checks.
    #[arg(long, default_value_t = 0.5)]
    tolerance: f64,

    /// Output format for the stats report.
    #[arg(
        long = "output-format",
        visible_alias = "output",
        value_enum,
        default_value_t = StatsOutputFormat::Text
    )]
    output_format: StatsOutputFormat,
}

/// Arguments for the diff subcommand.
#[derive(clap::Args)]
pub(crate) struct DiffArgs {
    /// First dataset path.
    input_a: PathBuf,

    /// Second dataset path.
    input_b: PathBuf,

    /// Format for the first input (or auto-detect).
    #[arg(long = "format-a", value_enum, default_value = "auto")]
    format_a: ConvertFromFormat,

    /// Format for the second input (or auto-detect).
    #[arg(long = "format-b", value_enum, default_value = "auto")]
    format_b: ConvertFromFormat,

    /// Annotation matching strategy.
    #[arg(long, value_enum, default_value = "id")]
    match_by: DiffMatchBy,

    /// IoU threshold used with --match-by iou.
    #[arg(long, default_value_t = 0.5)]
    iou_threshold: f64,

    /// Include item-level detail in output.
    #[arg(long)]
    detail: bool,

    /// Output format for diff report.
    #[arg(
        long = "output-format",
        visible_alias = "output",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,
}

/// Arguments for the sample subcommand.
#[derive(clap::Args)]
pub(crate) struct SampleArgs {
    /// Input path.
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output path.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Source format (or auto-detect).
    #[arg(long = "from", value_enum, default_value = "auto")]
    from: ConvertFromFormat,

    /// Target format.
    #[arg(long = "to", value_enum)]
    to: Option<ConvertFormat>,

    /// Number of images to sample.
    #[arg(short = 'n', long = "n")]
    n: Option<usize>,

    /// Fraction of images to sample.
    #[arg(long = "fraction")]
    fraction: Option<f64>,

    /// Optional random seed for deterministic sampling.
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Sampling strategy.
    #[arg(long, value_enum, default_value = "random")]
    strategy: SampleStrategyArg,

    /// Comma-separated category names to filter on.
    #[arg(long = "categories")]
    categories: Option<String>,

    /// Category filter mode.
    #[arg(long = "category-mode", value_enum, default_value = "images")]
    category_mode: CategoryModeArg,

    /// Allow lossy output format conversions.
    #[arg(long = "allow-lossy")]
    allow_lossy: bool,

    /// Run the sampling pipeline and report what would be written, without writing output files.
    #[arg(long = "dry-run")]
    dry_run: bool,

    /// Output format for the sampling report.
    #[arg(
        long = "output-format",
        visible_alias = "report",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,
}

/// Arguments for the convert subcommand.
#[derive(clap::Args)]
pub(crate) struct ConvertArgs {
    /// Source format (use 'auto' for automatic detection).
    #[arg(short = 'f', long = "from", value_enum)]
    from: ConvertFromFormat,

    /// Target format.
    #[arg(short = 't', long = "to", value_enum)]
    to: ConvertFormat,

    /// Input path (required for local inputs; optional with --hf-repo when --from hf).
    #[arg(short = 'i', long = "input")]
    input: Option<PathBuf>,

    /// Output path.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Treat validation warnings as errors.
    #[arg(long)]
    strict: bool,

    /// Skip input validation entirely.
    #[arg(long = "no-validate")]
    no_validate: bool,

    /// Allow conversions that drop information (e.g., metadata, images without annotations).
    #[arg(long = "allow-lossy")]
    allow_lossy: bool,

    /// Run detection/validation/reporting without writing output files.
    #[arg(long = "dry-run")]
    dry_run: bool,

    /// Output format for the conversion report.
    #[arg(
        long = "output-format",
        visible_alias = "report",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,

    /// HF bbox format for --from hf / --to hf (xywh or xyxy).
    #[arg(long = "hf-bbox-format", value_enum, default_value = "xywh")]
    hf_bbox_format: HfBboxFormatArg,

    /// Override the object container column in HF metadata (e.g. annotations).
    #[arg(long = "hf-objects-column")]
    hf_objects_column: Option<String>,

    /// JSON file mapping integer category IDs to names for HF import.
    #[arg(long = "hf-category-map")]
    hf_category_map: Option<PathBuf>,

    /// HF dataset repo ID or dataset page URL for remote import.
    #[arg(long = "hf-repo")]
    hf_repo: Option<String>,

    /// Split name (e.g. train/validation/test) for HF or YOLO imports.
    #[arg(long = "split")]
    split: Option<String>,

    /// HF revision (branch, tag, or commit SHA).
    #[arg(long = "revision")]
    revision: Option<String>,

    /// HF config/subset.
    #[arg(long = "config")]
    config: Option<String>,

    /// HF auth token (also supports HF_TOKEN env var).
    #[arg(long = "token", env = "HF_TOKEN")]
    token: Option<String>,
}

/// Arguments for the list-formats subcommand.
#[derive(clap::Args)]
pub(crate) struct ListFormatsArgs {
    /// Output format for the format catalog.
    #[arg(
        long = "output-format",
        visible_alias = "output",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,
}

#[derive(serde::Serialize)]
struct ListFormatEntry {
    name: &'static str,
    aliases: &'static [&'static str],
    read: bool,
    write: bool,
    lossiness: &'static str,
    description: &'static str,
    file_based: bool,
    directory_based: bool,
}

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();
    let output = OutputContext::detect();

    match cli.command {
        Some(Commands::Validate(args)) => commands::validate::run(args, output),
        Some(Commands::Convert(args)) => commands::convert::run(args, output),
        Some(Commands::Stats(args)) => commands::stats::run(args, output),
        Some(Commands::Diff(args)) => commands::diff::run(args, output),
        Some(Commands::Sample(args)) => commands::sample::run(args, output),
        Some(Commands::ListFormats(args)) => commands::list_formats::run(args, output),
        None => {
            // No subcommand: just print help hint and exit successfully
            // This keeps backward compatibility with the existing test
            println!("panlabel {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("The universal annotation converter.");
            println!();
            println!("Run 'panlabel --help' for usage information.");
            Ok(())
        }
    }
}

fn write_json_stdout<T: serde::Serialize>(
    value: &T,
    output: OutputContext,
) -> Result<(), PanlabelError> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    match output.json_style() {
        JsonStyle::Pretty => serde_json::to_writer_pretty(&mut handle, value),
        JsonStyle::Compact => serde_json::to_writer(&mut handle, value),
    }
    .map_err(|source| PanlabelError::ReportJsonWrite { source })?;
    writeln!(handle).map_err(PanlabelError::Io)?;
    handle.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

/// Emit a conversion report to stdout in the requested format, then flush.
///
/// Used by both `convert` and `sample` to emit reports on both success and
/// blocked-lossy paths. By flushing stdout before returning, we ensure the
/// report is fully written before any subsequent stderr output from `main()`.
fn emit_conversion_report(
    report: &conversion::ConversionReport,
    format: ReportFormat,
    output: OutputContext,
) -> Result<(), PanlabelError> {
    match format {
        ReportFormat::Text => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            write!(handle, "{}", report).map_err(PanlabelError::Io)?;
            handle.flush().map_err(PanlabelError::Io)?;
        }
        ReportFormat::Json => write_json_stdout(report, output)?,
    }
    Ok(())
}

#[cfg(feature = "hf-remote")]
fn remote_payload_to_convert_format(payload: hf::acquire::HfAcquirePayloadFormat) -> ConvertFormat {
    match payload {
        hf::acquire::HfAcquirePayloadFormat::HfImagefolder => ConvertFormat::HfImagefolder,
        hf::acquire::HfAcquirePayloadFormat::Yolo => ConvertFormat::Yolo,
        hf::acquire::HfAcquirePayloadFormat::Voc => ConvertFormat::Voc,
        hf::acquire::HfAcquirePayloadFormat::Coco => ConvertFormat::Coco,
    }
}

fn resolve_from_format(
    from: ConvertFromFormat,
    path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    match from.as_concrete() {
        Some(format) => Ok(format),
        None => format_detection::detect_format(path),
    }
}

fn resolve_stats_format(
    format: Option<ConvertFormat>,
    path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    if let Some(format) = format {
        return Ok(format);
    }

    match format_detection::detect_format(path) {
        Ok(format) => Ok(format),
        Err(error) => {
            // If JSON itself is malformed, surface that directly — don't mask
            // it with an IR fallback that would produce a confusing error.
            if matches!(&error, PanlabelError::FormatDetectionJsonParse { .. }) {
                return Err(error);
            }

            let is_json_file = path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("json"))
                    .unwrap_or(false);

            if is_json_file {
                Ok(ConvertFormat::IrJson)
            } else {
                Err(error)
            }
        }
    }
}

fn parse_categories_arg(raw: Option<String>) -> Vec<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn load_hf_category_map(
    path: Option<&Path>,
) -> Result<std::collections::BTreeMap<i64, String>, PanlabelError> {
    let Some(path) = path else {
        return Ok(Default::default());
    };

    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::HfLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("invalid JSON in category map: {source}"),
        })?;

    let mut map = std::collections::BTreeMap::new();
    match value {
        serde_json::Value::Object(obj) => {
            for (raw_key, raw_value) in obj {
                let key = raw_key
                    .parse::<i64>()
                    .map_err(|_| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!("category-map key '{}' is not a valid integer", raw_key),
                    })?;
                let label = raw_value
                    .as_str()
                    .ok_or_else(|| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!(
                            "category-map value for key '{}' must be a string",
                            raw_key
                        ),
                    })?;
                map.insert(key, label.to_string());
            }
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.into_iter().enumerate() {
                let label = item
                    .as_str()
                    .ok_or_else(|| PanlabelError::HfLayoutInvalid {
                        path: path.to_path_buf(),
                        message: format!("category-map array entry {} must be a string", idx),
                    })?;
                map.insert(idx as i64, label.to_string());
            }
        }
        _ => {
            return Err(PanlabelError::HfLayoutInvalid {
                path: path.to_path_buf(),
                message:
                    "category map must be either a JSON object {\"0\":\"person\"} or string array"
                        .to_string(),
            });
        }
    }

    Ok(map)
}

fn validate_hf_flag_usage(
    args: &ConvertArgs,
    from_format: ConvertFormat,
) -> Result<(), PanlabelError> {
    let hf_involved =
        from_format == ConvertFormat::HfImagefolder || args.to == ConvertFormat::HfImagefolder;

    // --split is valid for HF and YOLO source formats, not just HF
    let split_allowed = hf_involved || from_format == ConvertFormat::Yolo;

    if args.split.is_some() && !split_allowed {
        return Err(PanlabelError::UnsupportedFormat(
            "--split can only be used with --from hf or --from yolo".to_string(),
        ));
    }

    // HF-specific flags (excluding --split, which is shared)
    let hf_specific_flags_used = args.hf_repo.is_some()
        || args.hf_objects_column.is_some()
        || args.hf_category_map.is_some()
        || args.revision.is_some()
        || args.config.is_some()
        || !matches!(args.hf_bbox_format, HfBboxFormatArg::Xywh);

    if hf_specific_flags_used && !hf_involved {
        return Err(PanlabelError::UnsupportedFormat(
            "HF-specific flags (--hf-*) can only be used with --from hf or --to hf".to_string(),
        ));
    }

    if args.hf_repo.is_some() && from_format != ConvertFormat::HfImagefolder {
        return Err(PanlabelError::UnsupportedFormat(
            "--hf-repo can only be used with --from hf".to_string(),
        ));
    }

    if args.hf_repo.is_none() && (args.revision.is_some() || args.config.is_some()) {
        return Err(PanlabelError::UnsupportedFormat(
            "--revision/--config require --hf-repo".to_string(),
        ));
    }

    if from_format == ConvertFormat::HfImagefolder && args.hf_repo.is_none() && args.input.is_none()
    {
        return Err(PanlabelError::UnsupportedFormat(
            "--from hf requires either --input <path> or --hf-repo <namespace/dataset>".to_string(),
        ));
    }

    Ok(())
}

fn ensure_unique_image_file_names(dataset: &ir::Dataset, side: &str) -> Result<(), PanlabelError> {
    let mut seen = std::collections::HashSet::new();
    for image in &dataset.images {
        if !seen.insert(image.file_name.clone()) {
            return Err(PanlabelError::DiffFailed {
                message: format!(
                    "duplicate image file_name '{}' found in dataset {}. Use unique image names for reliable diffing.",
                    image.file_name, side
                ),
            });
        }
    }
    Ok(())
}

/// Read a dataset from a file in the specified format.
fn read_dataset(format: ConvertFormat, path: &Path) -> Result<ir::Dataset, PanlabelError> {
    read_dataset_with_options(
        format,
        path,
        &ir::io_hf_imagefolder::HfReadOptions::default(),
        &ir::io_yolo::YoloReadOptions::default(),
    )
}

fn read_dataset_with_options(
    format: ConvertFormat,
    path: &Path,
    hf_options: &ir::io_hf_imagefolder::HfReadOptions,
    yolo_options: &ir::io_yolo::YoloReadOptions,
) -> Result<ir::Dataset, PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::read_ir_json(path),
        ConvertFormat::Coco => ir::io_coco_json::read_coco_json(path),
        ConvertFormat::IbmCloudAnnotations => {
            ir::io_cloud_annotations_json::read_cloud_annotations_json(path)
        }
        ConvertFormat::Cvat => ir::io_cvat_xml::read_cvat_xml(path),
        ConvertFormat::LabelStudio => ir::io_label_studio_json::read_label_studio_json(path),
        ConvertFormat::Labelbox => ir::io_labelbox_json::read_labelbox_json(path),
        ConvertFormat::ScaleAi => ir::io_scale_ai_json::read_scale_ai_json(path),
        ConvertFormat::UnityPerception => {
            ir::io_unity_perception_json::read_unity_perception_json(path)
        }
        ConvertFormat::Tfod => ir::io_tfod_csv::read_tfod_csv(path),
        ConvertFormat::Tfrecord => ir::io_tfrecord::read_tfrecord(path),
        ConvertFormat::VottCsv => ir::io_vott_csv::read_vott_csv(path),
        ConvertFormat::VottJson => ir::io_vott_json::read_vott_json(path),
        ConvertFormat::Yolo => ir::io_yolo::read_yolo_dir_with_options(path, yolo_options),
        ConvertFormat::YoloKeras => ir::io_yolo_keras_txt::read_yolo_keras_txt(path),
        ConvertFormat::YoloV4Pytorch => ir::io_yolo_keras_txt::read_yolov4_pytorch_txt(path),
        ConvertFormat::Voc => ir::io_voc_xml::read_voc_dir(path),
        ConvertFormat::HfImagefolder => read_hf_dataset_with_options(path, hf_options),
        ConvertFormat::SageMaker => ir::io_sagemaker_manifest::read_sagemaker_manifest(path),
        ConvertFormat::LabelMe => ir::io_labelme_json::read_labelme_json(path),
        ConvertFormat::SuperAnnotate => ir::io_superannotate_json::read_superannotate_json(path),
        ConvertFormat::Supervisely => ir::io_supervisely_json::read_supervisely_json(path),
        ConvertFormat::Cityscapes => ir::io_cityscapes_json::read_cityscapes_json(path),
        ConvertFormat::Marmot => ir::io_marmot_xml::read_marmot_xml(path),
        ConvertFormat::CreateMl => ir::io_createml_json::read_createml_json(path),
        ConvertFormat::Kitti => ir::io_kitti::read_kitti_dir(path),
        ConvertFormat::Via => ir::io_via_json::read_via_json(path),
        ConvertFormat::Retinanet => ir::io_retinanet_csv::read_retinanet_csv(path),
        ConvertFormat::OpenImages => ir::io_openimages_csv::read_openimages_csv(path),
        ConvertFormat::Datumaro => ir::io_datumaro_json::read_datumaro_json(path),
        ConvertFormat::WiderFace => ir::io_wider_face_txt::read_wider_face_txt(path),
        ConvertFormat::Oidv4 => ir::io_oidv4_txt::read_oidv4_txt(path),
        ConvertFormat::Bdd100k => ir::io_bdd100k_json::read_bdd100k_json(path),
        ConvertFormat::V7Darwin => ir::io_v7_darwin_json::read_v7_darwin_json(path),
        ConvertFormat::EdgeImpulse => ir::io_edge_impulse_labels::read_edge_impulse_labels(path),
        ConvertFormat::OpenLabel => ir::io_openlabel_json::read_openlabel_json(path),
        ConvertFormat::ViaCsv => ir::io_via_csv::read_via_csv(path),
        ConvertFormat::KaggleWheat => ir::io_kaggle_wheat_csv::read_kaggle_wheat_csv(path),
        ConvertFormat::AutoMlVision => ir::io_automl_vision_csv::read_automl_vision_csv(path),
        ConvertFormat::Udacity => ir::io_udacity_csv::read_udacity_csv(path),
    }
}

/// Write a dataset to a file in the specified format.
fn write_dataset(
    format: ConvertFormat,
    path: &Path,
    dataset: &ir::Dataset,
) -> Result<(), PanlabelError> {
    write_dataset_with_options(
        format,
        path,
        dataset,
        &ir::io_hf_imagefolder::HfWriteOptions::default(),
    )
}

fn write_dataset_with_options(
    format: ConvertFormat,
    path: &Path,
    dataset: &ir::Dataset,
    hf_options: &ir::io_hf_imagefolder::HfWriteOptions,
) -> Result<(), PanlabelError> {
    match format {
        ConvertFormat::IrJson => ir::io_json::write_ir_json(path, dataset),
        ConvertFormat::Coco => ir::io_coco_json::write_coco_json(path, dataset),
        ConvertFormat::IbmCloudAnnotations => {
            ir::io_cloud_annotations_json::write_cloud_annotations_json(path, dataset)
        }
        ConvertFormat::Cvat => ir::io_cvat_xml::write_cvat_xml(path, dataset),
        ConvertFormat::LabelStudio => {
            ir::io_label_studio_json::write_label_studio_json(path, dataset)
        }
        ConvertFormat::Labelbox => ir::io_labelbox_json::write_labelbox_json(path, dataset),
        ConvertFormat::ScaleAi => ir::io_scale_ai_json::write_scale_ai_json(path, dataset),
        ConvertFormat::UnityPerception => {
            ir::io_unity_perception_json::write_unity_perception_json(path, dataset)
        }
        ConvertFormat::Tfod => ir::io_tfod_csv::write_tfod_csv(path, dataset),
        ConvertFormat::Tfrecord => ir::io_tfrecord::write_tfrecord(path, dataset),
        ConvertFormat::VottCsv => ir::io_vott_csv::write_vott_csv(path, dataset),
        ConvertFormat::VottJson => ir::io_vott_json::write_vott_json(path, dataset),
        ConvertFormat::Yolo => ir::io_yolo::write_yolo_dir(path, dataset),
        ConvertFormat::YoloKeras => ir::io_yolo_keras_txt::write_yolo_keras_txt(path, dataset),
        ConvertFormat::YoloV4Pytorch => {
            ir::io_yolo_keras_txt::write_yolov4_pytorch_txt(path, dataset)
        }
        ConvertFormat::Voc => ir::io_voc_xml::write_voc_dir(path, dataset),
        ConvertFormat::HfImagefolder => {
            ir::io_hf_imagefolder::write_hf_imagefolder_with_options(path, dataset, hf_options)
        }
        ConvertFormat::SageMaker => {
            ir::io_sagemaker_manifest::write_sagemaker_manifest(path, dataset)
        }
        ConvertFormat::LabelMe => ir::io_labelme_json::write_labelme_json(path, dataset),
        ConvertFormat::SuperAnnotate => {
            ir::io_superannotate_json::write_superannotate_json(path, dataset)
        }
        ConvertFormat::Supervisely => {
            ir::io_supervisely_json::write_supervisely_json(path, dataset)
        }
        ConvertFormat::Cityscapes => ir::io_cityscapes_json::write_cityscapes_json(path, dataset),
        ConvertFormat::Marmot => ir::io_marmot_xml::write_marmot_xml(path, dataset),
        ConvertFormat::CreateMl => ir::io_createml_json::write_createml_json(path, dataset),
        ConvertFormat::Kitti => ir::io_kitti::write_kitti_dir(path, dataset),
        ConvertFormat::Via => ir::io_via_json::write_via_json(path, dataset),
        ConvertFormat::Retinanet => ir::io_retinanet_csv::write_retinanet_csv(path, dataset),
        ConvertFormat::OpenImages => ir::io_openimages_csv::write_openimages_csv(path, dataset),
        ConvertFormat::Datumaro => ir::io_datumaro_json::write_datumaro_json(path, dataset),
        ConvertFormat::WiderFace => ir::io_wider_face_txt::write_wider_face_txt(path, dataset),
        ConvertFormat::Oidv4 => ir::io_oidv4_txt::write_oidv4_txt(path, dataset),
        ConvertFormat::Bdd100k => ir::io_bdd100k_json::write_bdd100k_json(path, dataset),
        ConvertFormat::V7Darwin => ir::io_v7_darwin_json::write_v7_darwin_json(path, dataset),
        ConvertFormat::EdgeImpulse => {
            ir::io_edge_impulse_labels::write_edge_impulse_labels(path, dataset)
        }
        ConvertFormat::OpenLabel => ir::io_openlabel_json::write_openlabel_json(path, dataset),
        ConvertFormat::ViaCsv => ir::io_via_csv::write_via_csv(path, dataset),
        ConvertFormat::KaggleWheat => {
            ir::io_kaggle_wheat_csv::write_kaggle_wheat_csv(path, dataset)
        }
        ConvertFormat::AutoMlVision => {
            ir::io_automl_vision_csv::write_automl_vision_csv(path, dataset)
        }
        ConvertFormat::Udacity => ir::io_udacity_csv::write_udacity_csv(path, dataset),
    }
}

fn read_hf_dataset_with_options(
    path: &Path,
    options: &ir::io_hf_imagefolder::HfReadOptions,
) -> Result<ir::Dataset, PanlabelError> {
    #[cfg(feature = "hf-parquet")]
    {
        if should_read_hf_parquet(path, options.split.as_deref())? {
            return ir::io_hf_parquet::read_hf_parquet_with_options(path, options);
        }
    }

    ir::io_hf_imagefolder::read_hf_imagefolder_with_options(path, options)
}

#[cfg(feature = "hf-parquet")]
fn should_read_hf_parquet(path: &Path, split: Option<&str>) -> Result<bool, PanlabelError> {
    let has_jsonl = hf_has_metadata(path, split, "metadata.jsonl")?;
    let has_parquet_layout =
        hf_has_metadata(path, split, "metadata.parquet")? || hf_has_any_parquet_file(path, split)?;
    Ok(has_parquet_layout && !has_jsonl)
}

#[cfg(feature = "hf-parquet")]
fn hf_has_metadata(
    path: &Path,
    split: Option<&str>,
    metadata_file_name: &str,
) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    if path.join(metadata_file_name).is_file() {
        return Ok(true);
    }

    if let Some(split_name) = split {
        let normalized = normalize_split_hint(split_name);
        return Ok(path.join(&normalized).join(metadata_file_name).is_file());
    }

    for entry in std::fs::read_dir(path).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if entry_path.is_dir() && entry_path.join(metadata_file_name).is_file() {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(feature = "hf-parquet")]
fn hf_has_any_parquet_file(path: &Path, split: Option<&str>) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    let normalized_split = split.map(normalize_split_hint);

    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::HfLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("failed while scanning parquet files: {source}"),
        })?;
        if !entry.file_type().is_file() {
            continue;
        }

        let entry_path = entry.path();
        let is_parquet = entry_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("parquet"))
            .unwrap_or(false);
        if !is_parquet {
            continue;
        }

        if entry_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("metadata.parquet"))
            .unwrap_or(false)
        {
            return Ok(true);
        }

        if let Some(split_name) = normalized_split.as_deref() {
            if parquet_path_matches_split(entry_path, split_name) {
                return Ok(true);
            }
            continue;
        }

        return Ok(true);
    }

    Ok(false)
}

#[cfg(feature = "hf-parquet")]
fn parquet_path_matches_split(path: &Path, split: &str) -> bool {
    let split = normalize_split_hint(split);

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_default();

    if file_name.starts_with(&format!("{split}-")) {
        return true;
    }

    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .map(|value| normalize_split_hint(value) == split)
            .unwrap_or(false)
    })
}

#[cfg(feature = "hf-parquet")]
fn normalize_split_hint(value: &str) -> String {
    match value.to_ascii_lowercase().as_str() {
        "val" | "valid" => "validation".to_string(),
        "validation" => "validation".to_string(),
        "train" => "train".to_string(),
        "test" => "test".to_string(),
        "dev" => "dev".to_string(),
        _ => value.to_ascii_lowercase(),
    }
}

/// Get a human-readable name for a format.
fn format_name(format: ConvertFormat) -> &'static str {
    format.to_conversion_format().name()
}

fn list_format_entries() -> Vec<ListFormatEntry> {
    format_catalog::FORMAT_CATALOG
        .iter()
        .map(|entry| ListFormatEntry {
            name: entry.format.name(),
            aliases: entry.aliases,
            read: true,
            write: true,
            lossiness: format_catalog::lossiness_name(entry.format.lossiness_relative_to_ir()),
            description: entry.description,
            file_based: entry.file_based,
            directory_based: entry.directory_based,
        })
        .collect()
}
