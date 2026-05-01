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

pub mod conversion;
pub mod diff;
pub mod error;
#[cfg(feature = "hf-remote")]
pub mod hf;
pub mod ir;
pub mod sample;
pub mod stats;
pub mod validation;

use std::fs::File;
use std::io::{BufRead, BufReader, IsTerminal, Write};
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
enum ConvertFormat {
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
struct OutputContext {
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
struct ValidateArgs {
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
struct StatsArgs {
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
struct DiffArgs {
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
struct SampleArgs {
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
struct ConvertArgs {
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
struct ListFormatsArgs {
    /// Output format for the format catalog.
    #[arg(
        long = "output-format",
        visible_alias = "output",
        value_enum,
        default_value_t = ReportFormat::Text
    )]
    output_format: ReportFormat,
}

struct FormatCatalogEntry {
    format: ConvertFormat,
    aliases: &'static [&'static str],
    description: &'static str,
    file_based: bool,
    directory_based: bool,
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

const FORMAT_CATALOG: &[FormatCatalogEntry] = &[
    FormatCatalogEntry {
        format: ConvertFormat::IrJson,
        aliases: &[],
        description: "Panlabel's intermediate representation (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Coco,
        aliases: &["coco-json"],
        description: "COCO object detection format (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::IbmCloudAnnotations,
        aliases: &[
            "cloud-annotations",
            "cloud-annotations-json",
            "ibm-cloud-annotations-json",
        ],
        description: "IBM Cloud Annotations localization JSON",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Cvat,
        aliases: &["cvat-xml"],
        description: "CVAT for images XML annotation export",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::LabelStudio,
        aliases: &["label-studio-json", "ls"],
        description: "Label Studio task export (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Labelbox,
        aliases: &["labelbox-json", "labelbox-ndjson"],
        description: "Labelbox current export rows (JSON/NDJSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::ScaleAi,
        aliases: &["scale", "scale-ai-json"],
        description: "Scale AI image annotation task/response JSON",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::UnityPerception,
        aliases: &["unity", "unity-perception-json", "solo"],
        description: "Unity Perception / SOLO bbox JSON dataset",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Tfod,
        aliases: &["tfod-csv"],
        description: "TensorFlow Object Detection format (CSV)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Tfrecord,
        aliases: &["tfrecords", "tf-record", "tfod-tfrecord", "tfod-tfrerecord"],
        description: "TensorFlow Object Detection API TFRecord Examples",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::VottCsv,
        aliases: &["vott"],
        description: "Microsoft VoTT CSV export",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::VottJson,
        aliases: &["vott-json-export"],
        description: "Microsoft VoTT JSON export",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Yolo,
        aliases: &[
            "ultralytics",
            "yolov8",
            "yolov5",
            "scaled-yolov4",
            "scaled-yolov4-txt",
        ],
        description: "YOLO .txt labels (directory/list-file based)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::YoloKeras,
        aliases: &["yolo-keras-txt", "keras-yolo"],
        description: "YOLO Keras absolute-coordinate TXT annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::YoloV4Pytorch,
        aliases: &["yolov4-pytorch-txt", "pytorch-yolov4"],
        description: "YOLOv4 PyTorch absolute-coordinate TXT annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Voc,
        aliases: &["pascal-voc", "voc-xml"],
        description: "Pascal VOC XML (directory-based)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::HfImagefolder,
        aliases: &["hf-imagefolder", "huggingface"],
        description: "Hugging Face ImageFolder metadata (metadata.jsonl/parquet)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::SageMaker,
        aliases: &[
            "sagemaker-manifest",
            "sagemaker-ground-truth",
            "ground-truth",
            "groundtruth",
            "aws-sagemaker",
        ],
        description: "AWS SageMaker Ground Truth object-detection manifest",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::LabelMe,
        aliases: &["labelme-json"],
        description: "LabelMe per-image JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::SuperAnnotate,
        aliases: &["superannotate-json", "sa"],
        description: "SuperAnnotate JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Supervisely,
        aliases: &["supervisely-json", "sly"],
        description: "Supervisely JSON annotation/project format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Cityscapes,
        aliases: &["cityscapes-json"],
        description: "Cityscapes polygon JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Marmot,
        aliases: &["marmot-xml"],
        description: "Marmot XML document-layout annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::CreateMl,
        aliases: &["createml", "create-ml-json"],
        description: "Apple CreateML annotation format (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Kitti,
        aliases: &["kitti-txt"],
        description: "KITTI object detection label files (directory-based)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Via,
        aliases: &["via-json", "vgg-via"],
        description: "VGG Image Annotator (VIA) JSON format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Retinanet,
        aliases: &["retinanet-csv", "keras-retinanet"],
        description: "keras-retinanet CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::OpenImages,
        aliases: &["openimages-csv", "open-images"],
        description: "Google OpenImages CSV annotation format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::KaggleWheat,
        aliases: &["kaggle-wheat-csv"],
        description: "Kaggle Global Wheat Detection CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::AutoMlVision,
        aliases: &["automl-vision-csv", "google-cloud-automl"],
        description: "Google Cloud AutoML Vision CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: ConvertFormat::Udacity,
        aliases: &["udacity-csv", "self-driving-car"],
        description: "Udacity Self-Driving Car Dataset CSV format",
        file_based: true,
        directory_based: false,
    },
];

/// Run the panlabel CLI.
///
/// This is the main entry point for the CLI, called from `main.rs`.
pub fn run() -> Result<(), PanlabelError> {
    let cli = Cli::parse();
    let output = OutputContext::detect();

    match cli.command {
        Some(Commands::Validate(args)) => run_validate(args, output),
        Some(Commands::Convert(args)) => run_convert(args, output),
        Some(Commands::Stats(args)) => run_stats(args, output),
        Some(Commands::Diff(args)) => run_diff(args, output),
        Some(Commands::Sample(args)) => run_sample(args, output),
        Some(Commands::ListFormats(args)) => run_list_formats(args, output),
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

/// Execute the stats subcommand.
fn run_stats(args: StatsArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let format = resolve_stats_format(args.format, &args.input)?;
    let dataset = read_dataset(format, &args.input)?;

    let opts = stats::StatsOptions {
        top_labels: args.top,
        top_pairs: args.top,
        oob_tolerance_px: args.tolerance,
        bar_width: 20,
    };

    let report = stats::stats_dataset(&dataset, &opts);

    match args.output_format {
        StatsOutputFormat::Text => print!("{}", report.display(output.stats_text_style())),
        StatsOutputFormat::Json => write_json_stdout(&report, output)?,
        StatsOutputFormat::Html => {
            let html = stats::html::render_html(&report)?;
            print!("{html}");
        }
    }

    Ok(())
}

/// Execute the diff subcommand.
fn run_diff(args: DiffArgs, output: OutputContext) -> Result<(), PanlabelError> {
    if matches!(args.match_by, DiffMatchBy::Iou)
        && !(0.0 < args.iou_threshold && args.iou_threshold <= 1.0)
    {
        return Err(PanlabelError::DiffFailed {
            message: "--iou-threshold must be in the interval (0.0, 1.0] when --match-by iou"
                .to_string(),
        });
    }

    let format_a = resolve_from_format(args.format_a, &args.input_a)?;
    let format_b = resolve_from_format(args.format_b, &args.input_b)?;

    let dataset_a = read_dataset(format_a, &args.input_a)?;
    let dataset_b = read_dataset(format_b, &args.input_b)?;

    ensure_unique_image_file_names(&dataset_a, "A")?;
    ensure_unique_image_file_names(&dataset_b, "B")?;

    let match_by = match args.match_by {
        DiffMatchBy::Id => diff::MatchBy::Id,
        DiffMatchBy::Iou => diff::MatchBy::Iou,
    };

    let opts = diff::DiffOptions {
        match_by,
        iou_threshold: args.iou_threshold,
        detail: args.detail,
        max_items: 20,
        bbox_eps: 1e-6,
    };

    let report = diff::diff_datasets(&dataset_a, &dataset_b, &opts);

    match args.output_format {
        ReportFormat::Text => {
            println!(
                "Dataset Diff: {} vs {}",
                args.input_a.display(),
                args.input_b.display()
            );
            println!();
            print!("{}", report);
        }
        ReportFormat::Json => write_json_stdout(&report, output)?,
    }

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

/// Execute the sample subcommand.
fn run_sample(args: SampleArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let from_format = resolve_from_format(args.from, &args.input)?;
    let to_format = match args.to {
        Some(target) => target,
        None => args.from.as_concrete().unwrap_or(ConvertFormat::IrJson),
    };

    let dataset = read_dataset(from_format, &args.input)?;

    let strategy = match args.strategy {
        SampleStrategyArg::Random => sample::SampleStrategy::Random,
        SampleStrategyArg::Stratified => sample::SampleStrategy::Stratified,
    };
    let category_mode = match args.category_mode {
        CategoryModeArg::Images => sample::CategoryMode::Images,
        CategoryModeArg::Annotations => sample::CategoryMode::Annotations,
    };

    let sample_opts = sample::SampleOptions {
        n: args.n,
        fraction: args.fraction,
        seed: args.seed,
        strategy,
        categories: parse_categories_arg(args.categories),
        category_mode,
    };

    let sampled_dataset = sample::sample_dataset(&dataset, &sample_opts)?;

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

/// Execute the validate subcommand.
fn run_validate(args: ValidateArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let dataset = read_dataset(args.format, &args.input)?;

    // Validate
    let opts = validation::ValidateOptions {
        strict: args.strict,
    };
    let report = validation::validate_dataset(&dataset, &opts);

    // Output results
    match args.output_format {
        ReportFormat::Json => write_json_stdout(&report.as_json(), output)?,
        ReportFormat::Text => print!("{}", report),
    }

    // Determine exit status
    let has_errors = report.error_count() > 0;
    let has_warnings = report.warning_count() > 0;

    if has_errors || (args.strict && has_warnings) {
        Err(PanlabelError::ValidationFailed {
            error_count: report.error_count(),
            warning_count: report.warning_count(),
            report,
        })
    } else {
        Ok(())
    }
}

/// Execute the convert subcommand.
fn run_convert(args: ConvertArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let from_format = match args.from.as_concrete() {
        Some(format) => format,
        None => {
            let input = args.input.as_ref().ok_or_else(|| {
                PanlabelError::UnsupportedFormat("--from auto requires --input <path>".to_string())
            })?;
            detect_format(input)?
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

    let (effective_input, source_display, effective_from_format) =
        if from_format == ConvertFormat::HfImagefolder && args.hf_repo.is_some() {
            #[cfg(feature = "hf-remote")]
            {
                let repo_input = args.hf_repo.as_deref().expect("checked is_some");
                let repo_ref = hf::resolve::parse_hf_input(
                    repo_input,
                    args.revision.as_deref(),
                    args.config.as_deref(),
                    args.split.as_deref(),
                )?;

                let preflight = hf::preflight::run_preflight(&repo_ref, args.token.as_deref());
                if preflight.is_none() {
                    eprintln!("Note: HF viewer API unavailable; proceeding with direct download.");
                }

                if let Some(preflight_data) = preflight.as_ref() {
                    if hf_read_options.objects_column.is_none() {
                        hf_read_options.objects_column =
                            preflight_data.detected_objects_column.clone();
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

                let acquired =
                    hf::acquire::acquire(&repo_ref, preflight.as_ref(), args.token.as_deref())?;
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

                if acquired.payload_format == hf::acquire::HfAcquirePayloadFormat::HfImagefolder
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

    // Step 1: Read the dataset
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

    // Step 2: Optionally validate the input
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

    // Step 3: Build conversion report and check for lossiness
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

    // Step 4: Write the dataset
    if !args.dry_run {
        write_dataset_with_options(args.to, &args.output, &dataset, &hf_write_options)?;
    }

    // Step 5: Output the report
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
        None => detect_format(path),
    }
}

fn resolve_stats_format(
    format: Option<ConvertFormat>,
    path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    if let Some(format) = format {
        return Ok(format);
    }

    match detect_format(path) {
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
    match format {
        ConvertFormat::IrJson => "ir-json",
        ConvertFormat::Coco => "coco",
        ConvertFormat::IbmCloudAnnotations => "ibm-cloud-annotations",
        ConvertFormat::Cvat => "cvat",
        ConvertFormat::LabelStudio => "label-studio",
        ConvertFormat::Labelbox => "labelbox",
        ConvertFormat::ScaleAi => "scale-ai",
        ConvertFormat::UnityPerception => "unity-perception",
        ConvertFormat::Tfod => "tfod",
        ConvertFormat::Tfrecord => "tfrecord",
        ConvertFormat::VottCsv => "vott-csv",
        ConvertFormat::VottJson => "vott-json",
        ConvertFormat::Yolo => "yolo",
        ConvertFormat::YoloKeras => "yolo-keras",
        ConvertFormat::YoloV4Pytorch => "yolov4-pytorch",
        ConvertFormat::Voc => "voc",
        ConvertFormat::HfImagefolder => "hf",
        ConvertFormat::SageMaker => "sagemaker",
        ConvertFormat::LabelMe => "labelme",
        ConvertFormat::SuperAnnotate => "superannotate",
        ConvertFormat::Supervisely => "supervisely",
        ConvertFormat::Cityscapes => "cityscapes",
        ConvertFormat::Marmot => "marmot",
        ConvertFormat::CreateMl => "create-ml",
        ConvertFormat::Kitti => "kitti",
        ConvertFormat::Via => "via",
        ConvertFormat::Retinanet => "retinanet",
        ConvertFormat::OpenImages => "openimages",
        ConvertFormat::KaggleWheat => "kaggle-wheat",
        ConvertFormat::AutoMlVision => "automl-vision",
        ConvertFormat::Udacity => "udacity",
    }
}

fn lossiness_name(lossiness: conversion::IrLossiness) -> &'static str {
    match lossiness {
        conversion::IrLossiness::Lossless => "lossless",
        conversion::IrLossiness::Conditional => "conditional",
        conversion::IrLossiness::Lossy => "lossy",
    }
}

fn list_format_entries() -> Vec<ListFormatEntry> {
    FORMAT_CATALOG
        .iter()
        .map(|entry| ListFormatEntry {
            name: format_name(entry.format),
            aliases: entry.aliases,
            read: true,
            write: true,
            lossiness: lossiness_name(
                entry
                    .format
                    .to_conversion_format()
                    .lossiness_relative_to_ir(),
            ),
            description: entry.description,
            file_based: entry.file_based,
            directory_based: entry.directory_based,
        })
        .collect()
}

/// Execute the list-formats subcommand.
fn run_list_formats(args: ListFormatsArgs, output: OutputContext) -> Result<(), PanlabelError> {
    let entries = list_format_entries();

    match args.output_format {
        ReportFormat::Text => {
            println!("Supported formats:");
            println!();
            println!(
                "  {:<12} {:<6} {:<6} {:<12} DESCRIPTION",
                "FORMAT", "READ", "WRITE", "LOSSINESS"
            );
            println!(
                "  {:<12} {:<6} {:<6} {:<12} -----------",
                "------", "----", "-----", "---------"
            );

            for entry in &entries {
                println!(
                    "  {:<12} {:<6} {:<6} {:<12} {}",
                    entry.name,
                    if entry.read { "yes" } else { "no" },
                    if entry.write { "yes" } else { "no" },
                    entry.lossiness,
                    entry.description
                );
            }

            println!();
            println!("Lossiness key:");
            println!("  lossless    - Format preserves all IR information");
            println!("  conditional - Format may lose info depending on dataset content");
            println!("  lossy       - Format always loses some IR information");
            println!();
            println!("Tip: Use '--from auto' with 'convert' for automatic format detection.");
            Ok(())
        }
        ReportFormat::Json => write_json_stdout(&entries, output),
    }
}

/// Detect the format of an input path based on extension/content (files)
/// or structure (directories).
fn detect_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if path.is_dir() {
        return detect_dir_format(path);
    }

    // Catch missing files early with a path-contextual message, instead of
    // letting downstream File::open produce a bare "IO error: No such file".
    if !path.exists() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "file does not exist".to_string(),
        });
    }

    // First try extension-based detection
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "csv" => return detect_csv_format(path),
            "tfrecord" | "tfrecords" => return detect_tfrecord_format(path),
            "json" => return detect_json_format(path),
            "jsonl" | "ndjson" | "manifest" => return detect_jsonl_format(path),
            "xml" => return detect_xml_format(path),
            "txt" => return detect_txt_format(path),
            _ => {}
        }
    }

    // Keep message stable (existing CLI tests assert this substring).
    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized file extension (expected .json, .jsonl, .ndjson, .manifest, .csv, .xml, .txt, or .tfrecord). Use --from to specify format explicitly.".to_string(),
    })
}

/// Evidence collected while probing a directory for a specific format.
struct FormatProbe {
    name: &'static str,
    format: ConvertFormat,
    found: Vec<String>,
    missing: Vec<String>,
}

impl FormatProbe {
    fn new(name: &'static str, format: ConvertFormat) -> Self {
        Self {
            name,
            format,
            found: Vec::new(),
            missing: Vec::new(),
        }
    }

    fn is_detected(&self) -> bool {
        !self.found.is_empty() && self.missing.is_empty()
    }

    fn is_partial(&self) -> bool {
        !self.found.is_empty() && !self.missing.is_empty()
    }
}

fn detect_dir_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let probes = probe_dir_formats(path)?;

    let detected: Vec<&FormatProbe> = probes.iter().filter(|p| p.is_detected()).collect();
    let partial: Vec<&FormatProbe> = probes.iter().filter(|p| p.is_partial()).collect();

    if detected.len() == 1 {
        return Ok(detected[0].format);
    }

    if detected.len() > 1 {
        let names: Vec<&str> = detected.iter().map(|p| p.name).collect();
        let header = if detected.len() == 2 {
            format!(
                "directory matches both {} and {} layouts",
                names[0], names[1]
            )
        } else {
            format!("directory matches multiple layouts ({})", names.join(", "))
        };

        let mut reason = format!("{}:\n", header);
        for p in &detected {
            reason.push_str(&format!("  - {}: found {}\n", p.name, p.found.join(", ")));
        }
        reason.push_str("Use --from to specify format explicitly.");

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason,
        });
    }

    // No complete match — check for partial matches (e.g. labels/ without images/).
    if !partial.is_empty() {
        let mut reason = String::new();
        for p in &partial {
            reason.push_str(&format!(
                "found {}-style markers ({}), but missing: {}\n",
                p.name,
                p.found.join(", "),
                p.missing.join(", "),
            ));
        }
        reason.push_str("Use --from to specify format explicitly, or fix the directory layout.");

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason,
        });
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "unrecognized directory layout. Expected one of:\n  \
                 - YOLO: labels/ with .txt files and sibling images/\n  \
                 - YOLO Keras / YOLOv4 PyTorch TXT: yolo_keras.txt, yolov4_pytorch.txt, annotations.txt, or train.txt\n  \
                 - VOC: Annotations/ with .xml files\n  \
                 - CVAT: annotations.xml at directory root\n  \
                 - IBM Cloud Annotations: _annotations.json at directory root\n  \
                 - VoTT JSON: vott-json-export/panlabel-export.json at directory root\n  \
                 - Scale AI: annotations/ with Scale AI .json files, or co-located .json files\n  \
                 - Unity Perception: SOLO frame/captures .json files\n  \
                 - HF: metadata.jsonl, metadata.parquet, or parquet shard files\n  \
                 - LabelMe: annotations/ with LabelMe .json files, or co-located .json files\n  \
                 - SuperAnnotate: annotations/ with SuperAnnotate .json files, or co-located .json files\n  \
                 - Cityscapes: gtFine/<split>/<city>/*_gtFine_polygons.json files\n  \
                 - Marmot: .xml files with Page@CropBox plus same-stem companion images\n  \
                 - Supervisely: ann/ with .json files, or project meta.json with dataset ann/ directories\n  \
                 - KITTI: label_2/ with .txt files and sibling image_2/\n\
                 Use --from to specify format explicitly."
            .to_string(),
    })
}

/// Probe a directory for all supported format markers. Returns one probe
/// per format, each with the markers it found and what's missing (if any).
fn probe_dir_formats(path: &Path) -> Result<Vec<FormatProbe>, PanlabelError> {
    let mut probes = Vec::with_capacity(4);

    // --- YOLO ---
    // Aligned with io_yolo::discover_layout/discover_source: requires labels/ with
    // .txt AND images/ for flat layout, OR data.yaml with split keys for split-aware.
    let mut yolo = FormatProbe::new("YOLO", ConvertFormat::Yolo);
    let (labels_dir_exists, has_txt) = if path.join("labels").is_dir() {
        (true, dir_contains_txt_files(&path.join("labels"))?)
    } else if is_labels_dir(path) {
        (true, dir_contains_txt_files(path)?)
    } else {
        (false, false)
    };
    if labels_dir_exists && has_txt {
        yolo.found.push("labels/ with .txt files".into());
        // Check for images/ sibling — aligned with reader requirement.
        let images_exists = if is_labels_dir(path) {
            path.parent()
                .map(|p| p.join("images").is_dir())
                .unwrap_or(false)
        } else {
            path.join("images").is_dir()
        };
        if images_exists {
            yolo.found.push("images/ directory".into());
        } else {
            yolo.missing.push("images/ directory".into());
        }
    }
    // Also detect split-aware YOLO via data.yaml with train/val/test keys.
    if yolo.found.is_empty() {
        if let Some(split_keys) = data_yaml_has_split_keys(path) {
            yolo.found.push(format!(
                "data.yaml with split keys: {}",
                split_keys.join(", ")
            ));
        }
    }
    let yolo_complete = yolo.is_detected();
    probes.push(yolo);

    // --- YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT ---
    probes.push(probe_yolo_keras_txt_dir(
        path,
        "YOLO Keras TXT",
        ConvertFormat::YoloKeras,
        &ir::io_yolo_keras_txt::YOLO_KERAS_ANNOTATION_CANDIDATES,
        !yolo_complete,
    )?);
    probes.push(probe_yolo_keras_txt_dir(
        path,
        "YOLOv4 PyTorch TXT",
        ConvertFormat::YoloV4Pytorch,
        &ir::io_yolo_keras_txt::YOLOV4_PYTORCH_ANNOTATION_CANDIDATES,
        !yolo_complete,
    )?);

    // --- VOC ---
    // Aligned with io_voc_xml::discover_layout: requires Annotations/ with
    // top-level .xml files, but JPEGImages/ is optional.
    let mut voc = FormatProbe::new("VOC", ConvertFormat::Voc);
    let (ann_dir, has_top_level_xml) = if path.join("Annotations").is_dir() {
        let ann = path.join("Annotations");
        (true, dir_contains_top_level_xml_files(&ann)?)
    } else if is_annotations_dir(path) {
        (true, dir_contains_top_level_xml_files(path)?)
    } else {
        (false, false)
    };
    if ann_dir && has_top_level_xml {
        voc.found
            .push("Annotations/ with top-level .xml files".into());
    }
    probes.push(voc);

    // --- CVAT ---
    let mut cvat = FormatProbe::new("CVAT", ConvertFormat::Cvat);
    if path.join("annotations.xml").is_file() {
        cvat.found.push("annotations.xml at root".into());
    }
    probes.push(cvat);

    // --- IBM Cloud Annotations ---
    let mut cloud_annotations =
        FormatProbe::new("IBM Cloud Annotations", ConvertFormat::IbmCloudAnnotations);
    let cloud_annotations_path = path.join("_annotations.json");
    if cloud_annotations_path.is_file() {
        if let Ok(contents) = std::fs::read_to_string(&cloud_annotations_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                if is_likely_cloud_annotations_file(&value) {
                    cloud_annotations
                        .found
                        .push("_annotations.json localization file".into());
                }
            }
        }
    }
    probes.push(cloud_annotations);

    // --- VoTT JSON ---
    let mut vott_json = FormatProbe::new("VoTT JSON", ConvertFormat::VottJson);
    let vott_export_path = path.join("vott-json-export").join("panlabel-export.json");
    let root_vott_export_path = path.join("panlabel-export.json");
    for candidate in [&vott_export_path, &root_vott_export_path] {
        if candidate.is_file() {
            if let Ok(contents) = std::fs::read_to_string(candidate) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if is_likely_vott_json_file(&value) {
                        vott_json.found.push(format!(
                            "{} VoTT JSON export",
                            candidate.strip_prefix(path).unwrap_or(candidate).display()
                        ));
                        break;
                    }
                }
            }
        }
    }
    probes.push(vott_json);

    // --- Scale AI ---
    let mut scale_ai = FormatProbe::new("Scale AI", ConvertFormat::ScaleAi);
    let scale_ann_dir = path.join("annotations");
    if scale_ann_dir.is_dir() && dir_contains_scale_ai_json(&scale_ann_dir)? {
        scale_ai
            .found
            .push("annotations/ with Scale AI .json files".into());
    } else if dir_contains_top_level_scale_ai_json(path)? {
        scale_ai
            .found
            .push("co-located Scale AI .json files".into());
    }
    probes.push(scale_ai);

    // --- Unity Perception ---
    let mut unity = FormatProbe::new("Unity Perception", ConvertFormat::UnityPerception);
    if dir_contains_unity_perception_json(path)? {
        unity.found.push("SOLO frame/captures .json files".into());
    }
    probes.push(unity);

    // --- HF ---
    let mut hf = FormatProbe::new("HF", ConvertFormat::HfImagefolder);
    if dir_contains_hf_metadata(path)? {
        hf.found.push("metadata.jsonl or metadata.parquet".into());
    } else if dir_has_parquet_shards(path)? {
        hf.found.push("parquet shard files".into());
    }
    probes.push(hf);

    // --- KITTI ---
    let mut kitti = FormatProbe::new("KITTI", ConvertFormat::Kitti);
    let kitti_labels_dir = if path.join("label_2").is_dir() {
        Some(path.join("label_2"))
    } else if is_dir_named_ci(path, "label_2") {
        Some(path.to_path_buf())
    } else {
        None
    };
    if let Some(ref labels_dir) = kitti_labels_dir {
        if dir_contains_top_level_txt_files(labels_dir)? {
            kitti.found.push("label_2/ with .txt files".into());
            let images_exists = if is_dir_named_ci(path, "label_2") {
                path.parent()
                    .map(|p| p.join("image_2").is_dir())
                    .unwrap_or(false)
            } else {
                path.join("image_2").is_dir()
            };
            if images_exists {
                kitti.found.push("image_2/ directory".into());
            } else {
                kitti.missing.push("image_2/ directory".into());
            }
        }
    }
    probes.push(kitti);

    // --- LabelMe ---
    let mut labelme = FormatProbe::new("LabelMe", ConvertFormat::LabelMe);
    let labelme_ann_dir = path.join("annotations");
    if labelme_ann_dir.is_dir() && dir_contains_labelme_json(&labelme_ann_dir)? {
        labelme
            .found
            .push("annotations/ with LabelMe .json files".into());
    } else if dir_contains_labelme_json(path)? {
        labelme.found.push("co-located LabelMe .json files".into());
    }
    probes.push(labelme);

    // --- SuperAnnotate ---
    let mut superannotate = FormatProbe::new("SuperAnnotate", ConvertFormat::SuperAnnotate);
    let superannotate_ann_dir = path.join("annotations");
    if superannotate_ann_dir.is_dir() {
        superannotate.found.push("annotations/ directory".into());
        if dir_contains_superannotate_json(&superannotate_ann_dir)? {
            superannotate.found.push("SuperAnnotate .json files".into());
        } else {
            superannotate
                .missing
                .push("SuperAnnotate .json files".into());
        }
    } else if dir_contains_top_level_superannotate_json(path)? {
        superannotate
            .found
            .push("co-located SuperAnnotate .json files".into());
    }
    probes.push(superannotate);

    // --- Cityscapes ---
    let mut cityscapes = FormatProbe::new("Cityscapes", ConvertFormat::Cityscapes);
    if path.join("gtFine").is_dir() {
        if dir_contains_cityscapes_json(&path.join("gtFine"))? {
            cityscapes
                .found
                .push("gtFine/ with Cityscapes polygon JSON files".into());
        } else {
            cityscapes
                .missing
                .push("Cityscapes *_gtFine_polygons.json files".into());
        }
    } else if dir_contains_cityscapes_json(path)? {
        cityscapes
            .found
            .push("Cityscapes polygon JSON files".into());
    }
    probes.push(cityscapes);

    // --- Marmot ---
    let mut marmot = FormatProbe::new("Marmot", ConvertFormat::Marmot);
    let marmot_status = dir_contains_marmot_xml(path)?;
    if marmot_status.found_xml {
        marmot.found.push("Marmot Page XML files".into());
        if marmot_status.missing_companion_images == 0 {
            marmot.found.push("same-stem companion images".into());
        } else {
            marmot.missing.push(format!(
                "same-stem companion image(s) for {} Marmot XML file(s)",
                marmot_status.missing_companion_images
            ));
        }
    }
    probes.push(marmot);

    // --- Supervisely ---
    let mut supervisely = FormatProbe::new("Supervisely", ConvertFormat::Supervisely);
    if path.join("ann").is_dir() {
        supervisely.found.push("ann/ directory".into());
        if dir_contains_supervisely_json(&path.join("ann"))? {
            supervisely.found.push("Supervisely .json files".into());
        } else {
            supervisely.missing.push("Supervisely .json files".into());
        }
    } else if path.join("meta.json").is_file() {
        supervisely.found.push("meta.json".into());
        let mut dataset_ann_dirs = 0usize;
        for entry in
            std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: format!("failed while inspecting directory: {source}"),
            })?
        {
            let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: format!("failed while inspecting directory: {source}"),
            })?;
            let ann_dir = entry.path().join("ann");
            if entry.path().is_dir() && ann_dir.is_dir() && dir_contains_supervisely_json(&ann_dir)?
            {
                dataset_ann_dirs += 1;
            }
        }
        if dataset_ann_dirs > 0 {
            supervisely.found.push(format!(
                "meta.json with {dataset_ann_dirs} dataset ann/ director{}",
                if dataset_ann_dirs == 1 { "y" } else { "ies" }
            ));
        } else {
            supervisely
                .missing
                .push("dataset ann/ directories with Supervisely .json files".into());
        }
    }
    probes.push(supervisely);

    Ok(probes)
}

fn probe_yolo_keras_txt_dir(
    path: &Path,
    name: &'static str,
    format: ConvertFormat,
    candidates: &[&str],
    allow_generic_train_txt: bool,
) -> Result<FormatProbe, PanlabelError> {
    let mut probe = FormatProbe::new(name, format);
    for candidate_name in candidates {
        if *candidate_name == "train.txt" && !allow_generic_train_txt {
            continue;
        }
        let candidate = path.join(candidate_name);
        if !candidate.is_file() {
            continue;
        }
        if ir::io_yolo_keras_txt::looks_like_yolo_keras_txt_file(&candidate)? {
            probe.found.push(format!(
                "{} absolute-coordinate annotation file",
                candidate_name
            ));
            break;
        } else if candidate_name.contains("yolo") || *candidate_name == "annotations.txt" {
            probe
                .missing
                .push(format!("valid {} row grammar in {}", name, candidate_name));
        }
    }
    Ok(probe)
}

fn dir_contains_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    dir_contains_extension_files(path, "txt")
}

fn dir_contains_hf_metadata(path: &Path) -> Result<bool, PanlabelError> {
    if path.join("metadata.jsonl").is_file() || path.join("metadata.parquet").is_file() {
        return Ok(true);
    }

    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir()
            && (entry_path.join("metadata.jsonl").is_file()
                || entry_path.join("metadata.parquet").is_file())
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Check for HF parquet shard files (e.g. `data/train-00000-of-00001.parquet`).
/// Scans up to depth 2 for .parquet files that aren't `metadata.parquet`.
fn dir_has_parquet_shards(path: &Path) -> Result<bool, PanlabelError> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return Ok(false),
    };
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if !entry_path.is_dir() {
            continue;
        }
        let sub_entries = match std::fs::read_dir(&entry_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for sub_entry in sub_entries {
            let sub_entry = sub_entry.map_err(PanlabelError::Io)?;
            let sub_path = sub_entry.path();
            if sub_path.is_file()
                && sub_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("parquet"))
                    .unwrap_or(false)
                && sub_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| !n.eq_ignore_ascii_case("metadata.parquet"))
                    .unwrap_or(false)
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Check if a directory contains .xml files at the top level only (non-recursive).
/// Aligned with VOC reader's `collect_xml_files()` which uses `fs::read_dir`.
fn dir_contains_top_level_xml_files(path: &Path) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("xml"))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dir_contains_extension_files(path: &Path, extension: &str) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;

        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn is_labels_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("labels"))
        .unwrap_or(false)
}

fn is_annotations_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("annotations"))
        .unwrap_or(false)
}

fn is_dir_named_ci(path: &Path, dir_name: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(dir_name))
        .unwrap_or(false)
}

fn dir_contains_top_level_txt_files(path: &Path) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(path).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if a directory contains at least one LabelMe JSON file.
///
/// Quick structural check: looks for a .json file with a `shapes` array key.
fn dir_contains_labelme_json(dir: &Path) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(dir).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: dir.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = std::fs::read_to_string(&entry_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if is_likely_labelme_file(&value) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Check if a directory contains at least one SuperAnnotate annotation JSON file.
fn dir_contains_superannotate_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, is_likely_superannotate_file)
}

/// Check if a directory contains at least one top-level SuperAnnotate JSON file.
fn dir_contains_top_level_superannotate_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_json_matching(dir, is_likely_superannotate_file)
}

fn dir_contains_scale_ai_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, ir::io_scale_ai_json::is_likely_scale_ai_file)
}

fn dir_contains_top_level_scale_ai_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_top_level_json_matching(dir, ir::io_scale_ai_json::is_likely_scale_ai_file)
}

fn dir_contains_unity_perception_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(
        dir,
        ir::io_unity_perception_json::is_likely_unity_perception_file,
    )
}

struct MarmotDirStatus {
    found_xml: bool,
    missing_companion_images: usize,
}

fn dir_contains_marmot_xml(dir: &Path) -> Result<MarmotDirStatus, PanlabelError> {
    let mut found_xml = false;
    let mut missing_companion_images = 0usize;
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if !entry.file_type().is_file()
            || !entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("xml"))
                .unwrap_or(false)
        {
            continue;
        }
        if ir::io_marmot_xml::is_likely_marmot_xml_file(entry_path)? {
            found_xml = true;
            if !ir::io_marmot_xml::has_companion_image(entry_path) {
                missing_companion_images += 1;
            }
        }
    }
    Ok(MarmotDirStatus {
        found_xml,
        missing_companion_images,
    })
}

fn dir_contains_cityscapes_json(dir: &Path) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if !entry.file_type().is_file()
            || !entry_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("_gtFine_polygons.json"))
                .unwrap_or(false)
        {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(entry_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                if ir::io_cityscapes_json::is_likely_cityscapes_file(&value) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Check if a directory contains at least one Supervisely annotation JSON file.
fn dir_contains_supervisely_json(dir: &Path) -> Result<bool, PanlabelError> {
    dir_contains_json_matching(dir, is_likely_supervisely_file)
}

fn dir_contains_top_level_json_matching(
    dir: &Path,
    predicate: fn(&serde_json::Value) -> bool,
) -> Result<bool, PanlabelError> {
    for entry in std::fs::read_dir(dir).map_err(|source| PanlabelError::FormatDetectionFailed {
        path: dir.to_path_buf(),
        reason: format!("failed while inspecting directory: {source}"),
    })? {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = std::fs::read_to_string(&entry_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if predicate(&value) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

fn dir_contains_json_matching(
    dir: &Path,
    predicate: fn(&serde_json::Value) -> bool,
) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: dir.to_path_buf(),
            reason: format!("failed while inspecting directory: {source}"),
        })?;
        let entry_path = entry.path();
        if entry.file_type().is_file()
            && entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = std::fs::read_to_string(entry_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if predicate(&value) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Check if `data.yaml` exists and contains split keys (train/val/test).
/// Returns `Some(vec!["train", ...])` if found, `None` otherwise.
fn data_yaml_has_split_keys(path: &Path) -> Option<Vec<String>> {
    let yaml_path = path.join("data.yaml");
    let content = std::fs::read_to_string(&yaml_path).ok()?;
    let mapping: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;
    let map = mapping.as_mapping()?;
    let mut found = Vec::new();
    for key in ["train", "val", "test"] {
        if map.contains_key(serde_yaml::Value::String(key.to_string())) {
            found.push(key.to_string());
        }
    }
    if found.is_empty() {
        None
    } else {
        Some(found)
    }
}

/// Detect whether a JSON file is Label Studio, COCO, or IR JSON format.
/// Detect CSV sub-format by sniffing column count and header.
///
/// Heuristics:
/// - 8 columns (filename,width,height,class,xmin,ymin,xmax,ymax) -> TFOD
/// - 6 columns (path,x1,y1,x2,y2,class_name or headerless data) -> RetinaNet
fn detect_tfrecord_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    if ir::io_tfrecord::is_supported_tfrecord_file(path)? {
        Ok(ConvertFormat::Tfrecord)
    } else {
        Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "TFRecord framing is valid only for uncompressed TFOD-style tf.train.Example records in v1, or the file is not a TFRecord. Use --from to specify format explicitly."
                .to_string(),
        })
    }
}

fn detect_txt_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let looks_like = ir::io_yolo_keras_txt::looks_like_yolo_keras_txt_file(path)?;
    if !looks_like {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "TXT file does not match the YOLO Keras / YOLOv4 PyTorch absolute-coordinate grammar. Use --from to specify format explicitly.".to_string(),
        });
    }

    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let normalized = filename.replace('-', "_");

    if normalized.contains("yolo_keras") || normalized.contains("keras_yolo") {
        return Ok(ConvertFormat::YoloKeras);
    }
    if normalized.contains("yolov4_pytorch") || normalized.contains("pytorch_yolov4") {
        return Ok(ConvertFormat::YoloV4Pytorch);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!(
            "TXT file matches both yolo-keras and yolov4-pytorch absolute-coordinate layouts, but filename '{}' is generic. Use --from yolo-keras or --from yolov4-pytorch to specify the intended public format.",
            filename
        ),
    })
}

fn detect_csv_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let file = std::fs::File::open(path).map_err(PanlabelError::Io)?;
    let reader = std::io::BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    // Read up to 8 records for sniffing
    let mut records: Vec<csv::StringRecord> = Vec::new();
    for result in csv_reader.records().take(8) {
        let record = result.map_err(|_| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "failed to parse CSV row while detecting format".to_string(),
        })?;
        records.push(record);
    }

    if records.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason:
                "CSV file is empty; cannot determine format. Use --from to specify format explicitly."
                    .to_string(),
        });
    }

    let first = &records[0];
    let ncols = first.len();

    // --- Header-based detection ---
    let col0 = first.get(0).unwrap_or("");
    let col1 = first.get(1).unwrap_or("");
    let col3 = first.get(3).unwrap_or("");

    // Kaggle Wheat: 5 columns, header starting with "image_id"
    if ncols == 5 && col0.eq_ignore_ascii_case("image_id") && col3.eq_ignore_ascii_case("bbox") {
        return Ok(ConvertFormat::KaggleWheat);
    }

    // Kaggle Wheat: 5 columns, headerless — col3 looks like bracketed bbox
    if ncols == 5 {
        let looks_like_bbox = col3.trim().starts_with('[') && col3.trim().ends_with(']');
        let col1_is_int = col1.parse::<u32>().is_ok();
        if looks_like_bbox && col1_is_int {
            return Ok(ConvertFormat::KaggleWheat);
        }
    }

    // VoTT CSV: exact 6-column header image,xmin,ymin,xmax,ymax,label.
    if ncols == 6
        && col0.eq_ignore_ascii_case("image")
        && col1.eq_ignore_ascii_case("xmin")
        && first
            .get(2)
            .map(|v| v.eq_ignore_ascii_case("ymin"))
            .unwrap_or(false)
        && col3.eq_ignore_ascii_case("xmax")
        && first
            .get(4)
            .map(|v| v.eq_ignore_ascii_case("ymax"))
            .unwrap_or(false)
        && first
            .get(5)
            .map(|v| v.eq_ignore_ascii_case("label"))
            .unwrap_or(false)
    {
        return Ok(ConvertFormat::VottCsv);
    }

    // RetinaNet: 6 columns
    if ncols == 6 {
        return Ok(ConvertFormat::Retinanet);
    }

    // OpenImages: 8 or 13 columns with header starting with "ImageID"
    if (ncols == 8 || ncols == 13) && col0.eq_ignore_ascii_case("ImageID") {
        return Ok(ConvertFormat::OpenImages);
    }

    // AutoML Vision: 9 or 11 columns
    if ncols == 9 || ncols == 11 {
        // Check for ML_USE-like first column or header alias
        let c0_lower = col0.to_ascii_lowercase();
        let is_automl_header = c0_lower == "set" || c0_lower == "ml_use";
        let is_automl_data = matches!(
            c0_lower.as_str(),
            "train" | "validation" | "test" | "unassigned"
        );
        if is_automl_header || is_automl_data {
            return Ok(ConvertFormat::AutoMlVision);
        }
        // Check if cols 5/6 (in 11-col form) are empty placeholders
        if ncols == 11 {
            let col5 = first.get(5).unwrap_or("_");
            let col6 = first.get(6).unwrap_or("_");
            if col5.is_empty() && col6.is_empty() {
                return Ok(ConvertFormat::AutoMlVision);
            }
        }
    }

    // 8-column formats: TFOD vs Udacity vs headerless OpenImages
    if ncols == 8 {
        // Check for OpenImages: column order is ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax
        // where Source is non-numeric and Confidence is a float
        let col1_str = first.get(1).unwrap_or("");
        let col3_str = first.get(3).unwrap_or("");
        let col1_not_numeric = col1_str.parse::<f64>().is_err();
        let col3_is_float = col3_str.parse::<f64>().is_ok();

        // Check for TFOD/Udacity header
        if col0.eq_ignore_ascii_case("filename") {
            // Has header — sniff data rows to distinguish TFOD vs Udacity
            return detect_tfod_vs_udacity(&records[1..], path);
        }

        // Headerless 8-column: OpenImages if col1 is non-numeric and col3 looks like confidence
        if col1_not_numeric && col3_is_float && !col0.is_empty() {
            // Further check: are cols 4-7 in [0,1]? OpenImages uses normalized coords
            let all_normalized = (4..8).all(|i| {
                first
                    .get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|v| (0.0..=1.0).contains(&v))
                    .unwrap_or(false)
            });
            if all_normalized {
                return Ok(ConvertFormat::OpenImages);
            }
        }

        // Headerless 8-column TFOD/Udacity
        return detect_tfod_vs_udacity(&records, path);
    }

    // 13-column: likely OpenImages extended
    if ncols == 13 {
        return Ok(ConvertFormat::OpenImages);
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: format!(
            "CSV has {ncols} columns; not recognized as any supported format. Use --from to specify format explicitly."
        ),
    })
}

/// Distinguishes TFOD (normalized) from Udacity (absolute pixel) by inspecting coordinate values.
fn detect_tfod_vs_udacity(
    data_records: &[csv::StringRecord],
    _path: &Path,
) -> Result<ConvertFormat, PanlabelError> {
    // If any sampled bbox coordinate is outside [0,1], it's Udacity (absolute pixels)
    for record in data_records {
        if record.len() < 8 {
            continue;
        }
        for i in 4..8 {
            if let Some(Ok(v)) = record.get(i).map(|s| s.parse::<f64>()) {
                if !(0.0..=1.0).contains(&v) {
                    return Ok(ConvertFormat::Udacity);
                }
            }
        }
    }
    // All in [0,1] or no data rows — default to TFOD
    Ok(ConvertFormat::Tfod)
}

/// Detect whether a JSON Lines file is a SageMaker Ground Truth manifest.
///
/// Heuristic: first non-empty line is an object with a string `source-ref`
/// and exactly one object-detection label attribute. The label attribute is
/// dynamic, so we accept either a sibling `<label>-metadata.type` of
/// `groundtruth/object-detection` or the canonical `annotations` +
/// `image_size` label-object shape.
fn detect_jsonl_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let mut first_non_empty = None;
    for line in reader.lines() {
        let line = line.map_err(PanlabelError::Io)?;
        if !line.trim().is_empty() {
            first_non_empty = Some(line);
            break;
        }
    }

    let Some(line) = first_non_empty else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "JSON Lines file is empty; cannot determine format. Use --from to specify format explicitly."
                .to_string(),
        });
    };

    let value: serde_json::Value =
        serde_json::from_str(&line).map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!(
                "failed to parse first JSON Lines row while detecting format: {source}"
            ),
        })?;

    if ir::io_labelbox_json::is_likely_labelbox_row(&value) {
        Ok(ConvertFormat::Labelbox)
    } else if is_likely_sagemaker_manifest_row(&value) {
        Ok(ConvertFormat::SageMaker)
    } else {
        Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "JSON Lines file not recognized as Labelbox export rows or a SageMaker Ground Truth object-detection manifest. Use --from to specify format explicitly."
                .to_string(),
        })
    }
}

///
/// Heuristics:
/// - Array-root JSON: Label Studio task export
/// - Object-root JSON: inspect `annotations[0].bbox`
///   - array of 4 numbers -> COCO
///   - object min/max or xmin/ymin/xmax/ymax -> IR JSON
fn detect_json_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value: serde_json::Value = serde_json::from_reader(reader).map_err(|source| {
        PanlabelError::FormatDetectionJsonParse {
            path: path.to_path_buf(),
            source,
        }
    })?;

    if let Some(items) = value.as_array() {
        if items.is_empty() {
            // Empty array is ambiguous between Label Studio and CreateML
            return Err(PanlabelError::FormatDetectionFailed {
                path: path.to_path_buf(),
                reason: "empty JSON array is ambiguous (could be Label Studio or CreateML). \
                         Use --from to specify format explicitly."
                    .to_string(),
            });
        }

        if ir::io_labelbox_json::is_likely_labelbox_row(&items[0]) {
            return Ok(ConvertFormat::Labelbox);
        }

        if ir::io_scale_ai_json::is_likely_scale_ai_file(&items[0]) {
            return Ok(ConvertFormat::ScaleAi);
        }

        if ir::io_unity_perception_json::is_likely_unity_perception_file(&items[0]) {
            return Ok(ConvertFormat::UnityPerception);
        }

        if is_likely_label_studio_task(&items[0]) {
            return Ok(ConvertFormat::LabelStudio);
        }

        if is_likely_createml_item(&items[0]) {
            return Ok(ConvertFormat::CreateMl);
        }

        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "array-root JSON not recognized (expected Labelbox export-row array, Scale AI task/response array, Unity Perception frame array, Label Studio task array, or CreateML image array). Use --from to specify format explicitly.".to_string(),
        });
    }

    // Object-root: check for Labelbox export row before COCO/IR heuristic.
    if ir::io_labelbox_json::is_likely_labelbox_row(&value) {
        return Ok(ConvertFormat::Labelbox);
    }

    // Object-root: check for Scale AI task/response JSON before COCO/IR heuristic.
    if ir::io_scale_ai_json::is_likely_scale_ai_file(&value) {
        return Ok(ConvertFormat::ScaleAi);
    }

    // Object-root: check for Unity Perception/SOLO frame or captures JSON.
    if ir::io_unity_perception_json::is_likely_unity_perception_file(&value) {
        return Ok(ConvertFormat::UnityPerception);
    }

    // Object-root: check for LabelMe (has "shapes" key) before COCO/IR heuristic
    if is_likely_labelme_file(&value) {
        return Ok(ConvertFormat::LabelMe);
    }

    // Object-root: check for IBM Cloud Annotations before COCO/IR heuristic.
    if is_likely_cloud_annotations_file(&value) {
        return Ok(ConvertFormat::IbmCloudAnnotations);
    }

    // Object-root: check for VoTT JSON before COCO/IR heuristic.
    if is_likely_vott_json_file(&value) {
        return Ok(ConvertFormat::VottJson);
    }

    // Object-root: check for new per-image JSON formats before COCO/IR heuristic.
    if is_likely_superannotate_file(&value) {
        return Ok(ConvertFormat::SuperAnnotate);
    }

    if ir::io_cityscapes_json::is_likely_cityscapes_file(&value) {
        return Ok(ConvertFormat::Cityscapes);
    }

    if is_likely_supervisely_file(&value) {
        return Ok(ConvertFormat::Supervisely);
    }

    // Object-root: check for VIA project (entries with filename + regions)
    if is_likely_via_project(&value) {
        return Ok(ConvertFormat::Via);
    }

    // Object-root detection: COCO-vs-IR heuristic.

    // Get annotations array
    let annotations = value.get("annotations").and_then(|v| v.as_array());

    let Some(annotations) = annotations else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "missing or invalid 'annotations' array. Cannot determine format.".to_string(),
        });
    };

    if annotations.is_empty() {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "empty 'annotations' array. Cannot determine format from empty dataset. Use --from to specify format explicitly.".to_string(),
        });
    }

    // Inspect the first annotation's bbox
    let first_ann = &annotations[0];
    let bbox = first_ann.get("bbox");

    let Some(bbox) = bbox else {
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "first annotation has no 'bbox' field. Cannot determine format.".to_string(),
        });
    };

    // Check if bbox is an array (COCO) or object (IR JSON)
    if let Some(arr) = bbox.as_array() {
        // COCO uses [x, y, width, height] - array of 4 numbers
        if arr.len() == 4 && arr.iter().all(|v| v.is_number()) {
            return Ok(ConvertFormat::Coco);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!(
                "bbox is an array but not [x,y,w,h] format (found {} elements). Cannot determine format.",
                arr.len()
            ),
        });
    }

    if let Some(obj) = bbox.as_object() {
        // IR JSON uses {min: {x, y}, max: {x, y}} or {xmin, ymin, xmax, ymax}
        // Check for the serialized format from our bbox.rs
        if obj.contains_key("min") && obj.contains_key("max") {
            return Ok(ConvertFormat::IrJson);
        }
        // Alternative flat format
        if obj.contains_key("xmin")
            && obj.contains_key("ymin")
            && obj.contains_key("xmax")
            && obj.contains_key("ymax")
        {
            return Ok(ConvertFormat::IrJson);
        }
        return Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "bbox is an object but doesn't match IR JSON format (expected min/max or xmin/ymin/xmax/ymax). Cannot determine format.".to_string(),
        });
    }

    Err(PanlabelError::FormatDetectionFailed {
        path: path.to_path_buf(),
        reason: "bbox has unexpected type (expected array or object). Cannot determine format."
            .to_string(),
    })
}

/// Detect whether an XML file is CVAT XML.
///
/// Heuristic:
/// - root `<annotations>` => CVAT
/// - root `<annotation>` => looks like a single VOC XML (not auto-detected)
fn detect_xml_format(path: &Path) -> Result<ConvertFormat, PanlabelError> {
    let xml = std::fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let doc = roxmltree::Document::parse(&xml).map_err(|source| {
        PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("failed to parse XML while detecting format: {source}"),
        }
    })?;

    match doc.root_element().tag_name().name() {
        "annotations" => Ok(ConvertFormat::Cvat),
        "Page" => {
            if ir::io_marmot_xml::is_likely_marmot_xml_str(&xml, path)? {
                Ok(ConvertFormat::Marmot)
            } else {
                Err(PanlabelError::FormatDetectionFailed {
                    path: path.to_path_buf(),
                    reason: "XML root is <Page>, but Page@CropBox is missing or malformed; cannot determine format. Use --from to specify format explicitly.".to_string(),
                })
            }
        }
        "annotation" => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: "XML root is <annotation> (looks like a single VOC file). Panlabel expects VOC as a directory layout; use --from voc with a VOC dataset directory.".to_string(),
        }),
        other => Err(PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: format!("unrecognized XML root element <{other}>; cannot determine format. Use --from to specify format explicitly."),
        }),
    }
}

fn is_likely_label_studio_task(value: &serde_json::Value) -> bool {
    let Some(task_obj) = value.as_object() else {
        return false;
    };

    let Some(data_obj) = task_obj.get("data").and_then(|v| v.as_object()) else {
        return false;
    };

    data_obj
        .get("image")
        .map(|value| value.is_string())
        .unwrap_or(false)
}

/// Detect whether a JSON array element looks like a CreateML image row.
///
/// Heuristic: object with `image` (string) and `annotations` (array) keys.
fn is_likely_createml_item(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_image = obj.get("image").map(|v| v.is_string()).unwrap_or(false);

    let has_annotations = obj
        .get("annotations")
        .map(|v| v.is_array())
        .unwrap_or(false);

    has_image && has_annotations
}

/// Detect whether a JSON object looks like a LabelMe annotation file.
///
/// Heuristic: object with `shapes` (array) key.
fn is_likely_labelme_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    obj.get("shapes").map(|v| v.is_array()).unwrap_or(false)
}

/// Detect whether a JSON object looks like an IBM Cloud Annotations localization file.
///
/// Heuristic: object with `type: "localization"`, `labels` array, and image-keyed
/// `annotations` object.
fn is_likely_cloud_annotations_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    obj.get("type").and_then(|v| v.as_str()) == Some("localization")
        && obj.get("labels").map(|v| v.is_array()).unwrap_or(false)
        && obj
            .get("annotations")
            .map(|v| v.is_object())
            .unwrap_or(false)
}

/// Detect whether a JSON object looks like a Microsoft VoTT JSON export.
///
/// Heuristic:
/// - aggregate project: top-level `assets` object/array whose first entry has
///   `asset` and `regions`
/// - per-asset file: top-level `asset` object and `regions` array
fn is_likely_vott_json_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    if obj.get("asset").map(|v| v.is_object()).unwrap_or(false)
        && obj.get("regions").map(|v| v.is_array()).unwrap_or(false)
    {
        return true;
    }

    let Some(assets) = obj.get("assets") else {
        return false;
    };

    if let Some(asset_map) = assets.as_object() {
        return asset_map.values().any(is_likely_vott_asset_entry);
    }

    if let Some(asset_array) = assets.as_array() {
        return asset_array.iter().any(is_likely_vott_asset_entry);
    }

    false
}

fn is_likely_vott_asset_entry(value: &serde_json::Value) -> bool {
    value.get("asset").map(|v| v.is_object()).unwrap_or(false)
        && value.get("regions").map(|v| v.is_array()).unwrap_or(false)
}

/// Detect whether a JSON object looks like a SuperAnnotate annotation file.
///
/// Heuristic: object with `metadata.width`, `metadata.height`, and an `instances` array.
fn is_likely_superannotate_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_instances = obj.get("instances").map(|v| v.is_array()).unwrap_or(false);
    let has_dimensions = obj
        .get("metadata")
        .and_then(|metadata| metadata.as_object())
        .map(|metadata| {
            metadata
                .get("width")
                .is_some_and(serde_json::Value::is_number)
                && metadata
                    .get("height")
                    .is_some_and(serde_json::Value::is_number)
        })
        .unwrap_or(false);

    has_instances && has_dimensions
}

/// Detect whether a JSON object looks like a Supervisely annotation file.
///
/// Heuristic: object with `size.width`, `size.height`, and an `objects` array.
fn is_likely_supervisely_file(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    let has_objects = obj.get("objects").map(|v| v.is_array()).unwrap_or(false);
    let has_dimensions = obj
        .get("size")
        .and_then(|size| size.as_object())
        .map(|size| {
            size.get("width").is_some_and(serde_json::Value::is_number)
                && size.get("height").is_some_and(serde_json::Value::is_number)
        })
        .unwrap_or(false);

    has_objects && has_dimensions
}

fn is_likely_via_project(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    // VIA project JSON: top-level keys are image identifiers whose values are
    // objects containing "filename" and "regions" keys.
    obj.values()
        .any(|v| v.is_object() && v.get("filename").is_some() && v.get("regions").is_some())
}

fn is_likely_sagemaker_manifest_row(value: &serde_json::Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    if !obj
        .get("source-ref")
        .and_then(|value| value.as_str())
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return false;
    }

    obj.iter()
        .filter(|(key, value)| is_likely_sagemaker_label_attribute(obj, key, value))
        .count()
        == 1
}

fn is_likely_sagemaker_label_attribute(
    row: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &serde_json::Value,
) -> bool {
    if key == "source-ref" || key.ends_with("-metadata") {
        return false;
    }

    let Some(label_obj) = value.as_object() else {
        return false;
    };

    let metadata_key = format!("{key}-metadata");
    let metadata_says_object_detection = row
        .get(&metadata_key)
        .and_then(|metadata| metadata.as_object())
        .and_then(|metadata| metadata.get("type"))
        .and_then(|value| value.as_str())
        .map(|metadata_type| metadata_type == "groundtruth/object-detection")
        .unwrap_or(false);

    let has_detection_shape = label_obj
        .get("annotations")
        .map(|value| value.is_array())
        .unwrap_or(false)
        && label_obj
            .get("image_size")
            .map(|value| value.is_array())
            .unwrap_or(false);

    metadata_says_object_detection || has_detection_shape
}
