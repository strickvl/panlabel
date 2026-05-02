//! Clap-free catalog of supported annotation formats.
//!
//! This module is the shared source of truth for canonical format identity,
//! canonical user-facing names, lossiness metadata, and `list-formats` catalog
//! metadata. CLI parsing and aliases still live in `lib.rs`.

/// Format identifier for conversion reporting.
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    IrJson,
    Coco,
    IbmCloudAnnotations,
    Cvat,
    LabelStudio,
    Labelbox,
    ScaleAi,
    UnityPerception,
    Tfod,
    Tfrecord,
    VottCsv,
    VottJson,
    Yolo,
    YoloKeras,
    YoloV4Pytorch,
    Voc,
    HfImagefolder,
    SageMaker,
    LabelMe,
    SuperAnnotate,
    Supervisely,
    Cityscapes,
    Marmot,
    CreateMl,
    Kitti,
    Via,
    Retinanet,
    OpenImages,
    Datumaro,
    WiderFace,
    Oidv4,
    Bdd100k,
    V7Darwin,
    EdgeImpulse,
    OpenLabel,
    ViaCsv,
    KaggleWheat,
    AutoMlVision,
    Udacity,
}

/// Classification of how lossy a format is relative to the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrLossiness {
    /// Format can represent everything in the IR (round-trip safe).
    Lossless,
    /// Format may lose some information depending on dataset content.
    Conditional,
    /// Format always loses some IR information.
    Lossy,
}

impl Format {
    /// Canonical CLI/report name for the format.
    pub fn name(&self) -> &'static str {
        match self {
            Format::IrJson => "ir-json",
            Format::Coco => "coco",
            Format::IbmCloudAnnotations => "ibm-cloud-annotations",
            Format::Cvat => "cvat",
            Format::LabelStudio => "label-studio",
            Format::Labelbox => "labelbox",
            Format::ScaleAi => "scale-ai",
            Format::UnityPerception => "unity-perception",
            Format::Tfod => "tfod",
            Format::Tfrecord => "tfrecord",
            Format::VottCsv => "vott-csv",
            Format::VottJson => "vott-json",
            Format::Yolo => "yolo",
            Format::YoloKeras => "yolo-keras",
            Format::YoloV4Pytorch => "yolov4-pytorch",
            Format::Voc => "voc",
            Format::HfImagefolder => "hf",
            Format::SageMaker => "sagemaker",
            Format::LabelMe => "labelme",
            Format::SuperAnnotate => "superannotate",
            Format::Supervisely => "supervisely",
            Format::Cityscapes => "cityscapes",
            Format::Marmot => "marmot",
            Format::CreateMl => "create-ml",
            Format::Kitti => "kitti",
            Format::Via => "via",
            Format::Retinanet => "retinanet",
            Format::OpenImages => "openimages",
            Format::Datumaro => "datumaro",
            Format::WiderFace => "wider-face",
            Format::Oidv4 => "oidv4",
            Format::Bdd100k => "bdd100k",
            Format::V7Darwin => "v7-darwin",
            Format::EdgeImpulse => "edge-impulse",
            Format::OpenLabel => "openlabel",
            Format::ViaCsv => "via-csv",
            Format::KaggleWheat => "kaggle-wheat",
            Format::AutoMlVision => "automl-vision",
            Format::Udacity => "udacity",
        }
    }

    /// How lossy this format is relative to the IR.
    ///
    /// - `IrJson`: Lossless (it IS the IR)
    /// - `Coco`: Conditional (loses dataset name, may lose some attributes)
    /// - `LabelStudio`: Lossy (drops IR-level metadata fields not representable in task export)
    /// - `Tfod`: Lossy (loses metadata, licenses, images without annotations, etc.)
    /// - `Yolo`: Lossy (loses metadata, licenses, attributes, etc.)
    /// - `Voc`: Lossy (loses metadata, licenses, supercategory, confidence, etc.)
    pub fn lossiness_relative_to_ir(&self) -> IrLossiness {
        match self {
            Format::IrJson => IrLossiness::Lossless,
            Format::Coco => IrLossiness::Conditional,
            Format::IbmCloudAnnotations => IrLossiness::Lossy,
            Format::Cvat => IrLossiness::Lossy,
            Format::LabelStudio => IrLossiness::Lossy,
            Format::Labelbox => IrLossiness::Lossy,
            Format::ScaleAi => IrLossiness::Lossy,
            Format::UnityPerception => IrLossiness::Lossy,
            Format::Tfod => IrLossiness::Lossy,
            Format::Tfrecord => IrLossiness::Lossy,
            Format::VottCsv => IrLossiness::Lossy,
            Format::VottJson => IrLossiness::Lossy,
            Format::Yolo => IrLossiness::Lossy,
            Format::YoloKeras => IrLossiness::Lossy,
            Format::YoloV4Pytorch => IrLossiness::Lossy,
            Format::Voc => IrLossiness::Lossy,
            Format::HfImagefolder => IrLossiness::Lossy,
            Format::SageMaker => IrLossiness::Lossy,
            Format::LabelMe => IrLossiness::Lossy,
            Format::SuperAnnotate => IrLossiness::Lossy,
            Format::Supervisely => IrLossiness::Lossy,
            Format::Cityscapes => IrLossiness::Lossy,
            Format::Marmot => IrLossiness::Lossy,
            Format::CreateMl => IrLossiness::Lossy,
            Format::Kitti => IrLossiness::Lossy,
            Format::Via => IrLossiness::Lossy,
            Format::Retinanet => IrLossiness::Lossy,
            Format::OpenImages => IrLossiness::Lossy,
            Format::Datumaro => IrLossiness::Lossy,
            Format::WiderFace => IrLossiness::Lossy,
            Format::Oidv4 => IrLossiness::Lossy,
            Format::Bdd100k => IrLossiness::Lossy,
            Format::V7Darwin => IrLossiness::Lossy,
            Format::EdgeImpulse => IrLossiness::Lossy,
            Format::OpenLabel => IrLossiness::Lossy,
            Format::ViaCsv => IrLossiness::Lossy,
            Format::KaggleWheat => IrLossiness::Lossy,
            Format::AutoMlVision => IrLossiness::Lossy,
            Format::Udacity => IrLossiness::Lossy,
        }
    }
}

/// Stable string used in machine-readable and human-readable catalog output.
pub fn lossiness_name(lossiness: IrLossiness) -> &'static str {
    match lossiness {
        IrLossiness::Lossless => "lossless",
        IrLossiness::Conditional => "conditional",
        IrLossiness::Lossy => "lossy",
    }
}

pub struct FormatCatalogEntry {
    pub format: Format,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub file_based: bool,
    pub directory_based: bool,
}

pub const FORMAT_CATALOG: &[FormatCatalogEntry] = &[
    FormatCatalogEntry {
        format: Format::IrJson,
        aliases: &[],
        description: "Panlabel's intermediate representation (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Coco,
        aliases: &["coco-json"],
        description: "COCO object detection format (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::IbmCloudAnnotations,
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
        format: Format::Cvat,
        aliases: &["cvat-xml"],
        description: "CVAT for images XML annotation export",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::LabelStudio,
        aliases: &["label-studio-json", "ls"],
        description: "Label Studio task export (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Labelbox,
        aliases: &["labelbox-json", "labelbox-ndjson"],
        description: "Labelbox current export rows (JSON/NDJSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::ScaleAi,
        aliases: &["scale", "scale-ai-json"],
        description: "Scale AI image annotation task/response JSON",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::UnityPerception,
        aliases: &["unity", "unity-perception-json", "solo"],
        description: "Unity Perception / SOLO bbox JSON dataset",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Tfod,
        aliases: &["tfod-csv"],
        description: "TensorFlow Object Detection format (CSV)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Tfrecord,
        aliases: &["tfrecords", "tf-record", "tfod-tfrecord", "tfod-tfrerecord"],
        description: "TensorFlow Object Detection API TFRecord Examples",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::VottCsv,
        aliases: &["vott"],
        description: "Microsoft VoTT CSV export",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::VottJson,
        aliases: &["vott-json-export"],
        description: "Microsoft VoTT JSON export",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Yolo,
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
        format: Format::YoloKeras,
        aliases: &["yolo-keras-txt", "keras-yolo"],
        description: "YOLO Keras absolute-coordinate TXT annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::YoloV4Pytorch,
        aliases: &["yolov4-pytorch-txt", "pytorch-yolov4"],
        description: "YOLOv4 PyTorch absolute-coordinate TXT annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Voc,
        aliases: &["pascal-voc", "voc-xml"],
        description: "Pascal VOC XML (directory-based)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::HfImagefolder,
        aliases: &["hf-imagefolder", "huggingface"],
        description: "Hugging Face ImageFolder metadata (metadata.jsonl/parquet)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::SageMaker,
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
        format: Format::LabelMe,
        aliases: &["labelme-json"],
        description: "LabelMe per-image JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::SuperAnnotate,
        aliases: &["superannotate-json", "sa"],
        description: "SuperAnnotate JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Supervisely,
        aliases: &["supervisely-json", "sly"],
        description: "Supervisely JSON annotation/project format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Cityscapes,
        aliases: &["cityscapes-json"],
        description: "Cityscapes polygon JSON annotation format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Marmot,
        aliases: &["marmot-xml"],
        description: "Marmot XML document-layout annotations",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::CreateMl,
        aliases: &["createml", "create-ml-json"],
        description: "Apple CreateML annotation format (JSON)",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Kitti,
        aliases: &["kitti-txt"],
        description: "KITTI object detection label files (directory-based)",
        file_based: false,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Via,
        aliases: &["via-json", "vgg-via"],
        description: "VGG Image Annotator (VIA) JSON format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Retinanet,
        aliases: &["retinanet-csv", "keras-retinanet"],
        description: "keras-retinanet CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::OpenImages,
        aliases: &["openimages-csv", "open-images"],
        description: "Google OpenImages CSV annotation format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Datumaro,
        aliases: &["datumaro-json", "datumaro-dataset"],
        description: "Datumaro JSON annotation format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::WiderFace,
        aliases: &["widerface", "wider-face-txt"],
        description: "WIDER Face aggregate TXT annotation format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Oidv4,
        aliases: &["oidv4-txt", "openimages-v4-txt", "oid"],
        description: "OIDv4 Toolkit TXT label format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::Bdd100k,
        aliases: &["bdd100k-json", "scalabel", "scalabel-json"],
        description: "BDD100K / Scalabel JSON detection format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::V7Darwin,
        aliases: &["darwin", "darwin-json", "v7"],
        description: "V7 Darwin JSON annotation format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::EdgeImpulse,
        aliases: &[
            "edge-impulse-labels",
            "edge-impulse-bounding-boxes",
            "bounding-boxes-labels",
        ],
        description: "Edge Impulse bounding_boxes.labels format",
        file_based: true,
        directory_based: true,
    },
    FormatCatalogEntry {
        format: Format::OpenLabel,
        aliases: &["asam-openlabel", "openlabel-json", "asam-openlabel-json"],
        description: "ASAM OpenLABEL JSON 2D bbox subset",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::ViaCsv,
        aliases: &["vgg-via-csv"],
        description: "VGG Image Annotator CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::KaggleWheat,
        aliases: &["kaggle-wheat-csv"],
        description: "Kaggle Global Wheat Detection CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::AutoMlVision,
        aliases: &["automl-vision-csv", "google-cloud-automl"],
        description: "Google Cloud AutoML Vision CSV format",
        file_based: true,
        directory_based: false,
    },
    FormatCatalogEntry {
        format: Format::Udacity,
        aliases: &["udacity-csv", "self-driving-car"],
        description: "Udacity Self-Driving Car Dataset CSV format",
        file_based: true,
        directory_based: false,
    },
];
