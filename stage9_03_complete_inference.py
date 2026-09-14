from pathlib import Path
import re
import csv
import json
import math
from collections import defaultdict

import cv2
import joblib
import numpy as np

from ultralytics import YOLO
from runtime_config import DEVICE, get_output_root

# ============================================================
# Stage 9.3
# COMPLETE INTERNAL ULTRASONIC TARGET INFERENCE
#
# 输入：
#   已生成的 Stage7 Raw + Robust-Tanh fusion crop images
#
# 输出：
#   specimen-level physical targets
#
# Pipeline:
#
# Fusion crops
#   -> YOLO
#   -> shallow RF rescue
#   -> frozen physical filter
#   -> physical-coordinate clustering
#   -> reliability evaluation
#   -> final physical targets
#
# IMPORTANT:
# 这不是独立测试脚本。
# 这是最终工程推理管线。
# ============================================================


# ============================================================
# 输入目录
#
# 现在先用现成的 Fold 数据测试。
#
# Rot90示例：
# D:\ascan\stage7_dataset\raw_robust_fusion_yolo
# \folds\fold_A\images\val
#
# Rot00示例：
# D:\ascan\stage7_dataset\raw_robust_fusion_yolo
# \folds\fold_B\images\val
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RUNTIME_DIR = get_output_root("runtime")

INPUT_IMAGE_DIR = RUNTIME_DIR / "input_images"

YOLO_ROT90_PATH = MODEL_DIR / "fusion_fold_A_best.pt"
YOLO_ROT00_PATH = MODEL_DIR / "fusion_fold_B_best.pt"

RF_MODEL_PATH = MODEL_DIR / "shallow_rf_rot00.joblib"

ANOMALY_ROOT = RUNTIME_DIR / "anomaly_maps"

OUTPUT_ROOT = get_output_root("complete_inference")

RAW_DETECTION_CSV = (
    OUTPUT_ROOT
    / "raw_detections.csv"
)

CLUSTER_CSV = (
    OUTPUT_ROOT
    / "physical_clusters.csv"
)

FINAL_TARGET_CSV = (
    OUTPUT_ROOT
    / "final_targets.csv"
)

FINAL_JSON = (
    OUTPUT_ROOT
    / "final_result.json"
)


# ============================================================
# YOLO
# ============================================================

INFER_CONF = 0.001

BASE_CONF = 0.05

YOLO_NMS_IOU = 0.70

IMAGE_SIZE = 416


# ============================================================
# Crop physical dimensions
# ============================================================

CROP_X_WIDTH_MM = 400.0

CROP_DEPTH_HEIGHT_MM = 140.0


# ============================================================
# Stage8 shallow entrance
# ============================================================

SHALLOW_MAX_DEPTH_MM = 90.0


# ============================================================
# Stage8.7 frozen RF + physical filter
# ============================================================

RF_PROBABILITY_MIN = 0.50

FINAL_DEPTH_MAX_MM = 65.0

FINAL_PEAK_DISTANCE_MAX = 0.15

FINAL_ANOMALY_P95_MIN = 2.0

FINAL_BOX_HEIGHT_MAX_MM = 80.0


# ============================================================
# Stage9 physical clustering
# ============================================================

CLUSTER_X_DISTANCE_MM = 80.0

CLUSTER_DEPTH_DISTANCE_MM = 50.0


# ============================================================
# Stage9.2 engineering reliability
#
# 0.75 是开发数据上的工程阈值，
# 不代表独立验证最优阈值。
# ============================================================

RELIABILITY_THRESHOLD = 0.75


MEMBER_REFERENCE = 6.0

MAX_CONF_REFERENCE = 0.50

MEAN_CONF_REFERENCE = 0.20

SOURCE_IMAGE_REFERENCE = 3.0


WEIGHT_MEMBER = 0.30

WEIGHT_MAX_CONF = 0.35

WEIGHT_MEAN_CONF = 0.20

WEIGHT_SOURCE_COUNT = 0.15


# ============================================================
# Stage8 features
#
# 顺序必须和RF训练完全一致
# ============================================================

FEATURES = [
    "yolo_conf",
    "center_depth_mm",
    "box_width_mm",
    "box_height_mm",

    "anomaly_mean",
    "anomaly_median",
    "anomaly_p90",
    "anomaly_p95",
    "anomaly_max",
    "anomaly_center",

    "ring_mean",
    "ring_p95",

    "mean_contrast",
    "p95_contrast",
    "max_ring_ratio",

    "occupancy_gt2",
    "occupancy_gt3",
    "occupancy_gt4",

    "peak_center_distance",
]


# ============================================================
# Ring
# ============================================================

RING_EXPAND_X_MM = 40.0

RING_EXPAND_DEPTH_MM = 20.0

EPS = 1e-6


# ============================================================
# Basic helpers
# ============================================================

def list_images(folder):

    return sorted([
        p
        for p in folder.iterdir()
        if p.suffix.lower()
        in [
            ".png",
            ".jpg",
            ".jpeg"
        ]
    ])


def detect_specimen(filename):

    name = filename.lower()

    if "pk050" in name:
        return "pk050"

    if "pk266" in name:
        return "pk266"

    # 后面接真实数据时可改
    return "unknown"


def detect_rotation(filename):

    name = filename.lower()

    if "rot00" in name:
        return "shear_rot00"

    if "rot90" in name:
        return "shear_rot90"

    raise ValueError(
        f"Cannot detect rotation: {filename}"
    )


def parse_crop_position(filename):

    name = filename.lower()

    x_match = re.search(
        r"_x(\d+)",
        name
    )

    d_match = re.search(
        r"_d(\d+)",
        name
    )

    if (
        x_match is None
        or
        d_match is None
    ):

        raise ValueError(
            f"Cannot parse crop position: {filename}"
        )

    return (
        float(
            int(
                x_match.group(1)
            )
        ),

        float(
            int(
                d_match.group(1)
            )
        ),
    )


def clamp01(value):

    return max(
        0.0,
        min(
            1.0,
            value
        )
    )


# ============================================================
# YOLO model cache
# ============================================================

YOLO_CACHE = {}


def get_yolo_model(rotation):

    if rotation in YOLO_CACHE:

        return YOLO_CACHE[
            rotation
        ]


    if rotation == "shear_rot90":

        model_path = (
            YOLO_ROT90_PATH
        )

    elif rotation == "shear_rot00":

        model_path = (
            YOLO_ROT00_PATH
        )

    else:

        raise ValueError(
            rotation
        )


    if not model_path.exists():

        raise FileNotFoundError(
            model_path
        )


    print(
        "\nLoading YOLO:",
        model_path
    )


    model = YOLO(
        str(
            model_path
        )
    )


    YOLO_CACHE[
        rotation
    ] = model


    return model


# ============================================================
# Anomaly map
# ============================================================

ANOMALY_CACHE = {}


def load_anomaly_map(
    specimen,
    rotation
):

    key = (
        specimen,
        rotation
    )


    if key in ANOMALY_CACHE:

        return ANOMALY_CACHE[
            key
        ]


    prefix = (
        f"{specimen}_{rotation}"
    )


    anomaly_path = (
        ANOMALY_ROOT
        /
        f"{prefix}_anomaly.npy"
    )


    x_path = (
        ANOMALY_ROOT
        /
        f"{prefix}_x_mm.npy"
    )


    depth_path = (
        ANOMALY_ROOT
        /
        f"{prefix}_depth_mm.npy"
    )


    for path in [
        anomaly_path,
        x_path,
        depth_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    result = {

        "anomaly":
            np.load(
                anomaly_path
            ),

        "x_mm":
            np.load(
                x_path
            ),

        "depth_mm":
            np.load(
                depth_path
            ),
    }


    ANOMALY_CACHE[
        key
    ] = result


    return result


# ============================================================
# Pixel -> physical coordinate
# ============================================================

def pixel_box_to_physical(
    box,
    crop_x_start,
    crop_depth_start,
    image_width,
    image_height
):

    x1, y1, x2, y2 = box


    px1 = (
        crop_x_start
        +
        x1 / image_width
        *
        CROP_X_WIDTH_MM
    )


    px2 = (
        crop_x_start
        +
        x2 / image_width
        *
        CROP_X_WIDTH_MM
    )


    d1 = (
        crop_depth_start
        +
        y1 / image_height
        *
        CROP_DEPTH_HEIGHT_MM
    )


    d2 = (
        crop_depth_start
        +
        y2 / image_height
        *
        CROP_DEPTH_HEIGHT_MM
    )


    return [
        float(px1),
        float(d1),
        float(px2),
        float(d2),
    ]


# ============================================================
# Anomaly regions
# ============================================================

def extract_region(
    anomaly_data,
    physical_box
):

    anomaly = anomaly_data[
        "anomaly"
    ]

    x_axis = anomaly_data[
        "x_mm"
    ]

    depth_axis = anomaly_data[
        "depth_mm"
    ]


    x1, d1, x2, d2 = (
        physical_box
    )


    x_low = min(
        x1,
        x2
    )

    x_high = max(
        x1,
        x2
    )


    d_low = min(
        d1,
        d2
    )

    d_high = max(
        d1,
        d2
    )


    x_mask = (
        (x_axis >= x_low)
        &
        (x_axis <= x_high)
    )


    d_mask = (
        (depth_axis >= d_low)
        &
        (depth_axis <= d_high)
    )


    region = anomaly[
        x_mask,
        :
    ][
        :,
        d_mask
    ]


    return (
        region,
        x_mask,
        d_mask
    )


def extract_ring_region(
    anomaly_data,
    physical_box
):

    anomaly = anomaly_data[
        "anomaly"
    ]

    x_axis = anomaly_data[
        "x_mm"
    ]

    depth_axis = anomaly_data[
        "depth_mm"
    ]


    x1, d1, x2, d2 = (
        physical_box
    )


    x_low = min(
        x1,
        x2
    )

    x_high = max(
        x1,
        x2
    )


    d_low = min(
        d1,
        d2
    )

    d_high = max(
        d1,
        d2
    )


    outer_x_low = (
        x_low
        -
        RING_EXPAND_X_MM
    )

    outer_x_high = (
        x_high
        +
        RING_EXPAND_X_MM
    )


    outer_d_low = (
        d_low
        -
        RING_EXPAND_DEPTH_MM
    )

    outer_d_high = (
        d_high
        +
        RING_EXPAND_DEPTH_MM
    )


    outer_mask = (
        (
            x_axis[:, None]
            >= outer_x_low
        )
        &
        (
            x_axis[:, None]
            <= outer_x_high
        )
        &
        (
            depth_axis[None, :]
            >= outer_d_low
        )
        &
        (
            depth_axis[None, :]
            <= outer_d_high
        )
    )


    inner_mask = (
        (
            x_axis[:, None]
            >= x_low
        )
        &
        (
            x_axis[:, None]
            <= x_high
        )
        &
        (
            depth_axis[None, :]
            >= d_low
        )
        &
        (
            depth_axis[None, :]
            <= d_high
        )
    )


    ring_mask = (
        outer_mask
        &
        (~inner_mask)
    )


    return anomaly[
        ring_mask
    ]


def center_anomaly_score(
    anomaly_data,
    physical_box
):

    anomaly = anomaly_data[
        "anomaly"
    ]

    x_axis = anomaly_data[
        "x_mm"
    ]

    depth_axis = anomaly_data[
        "depth_mm"
    ]


    x1, d1, x2, d2 = (
        physical_box
    )


    cx = (
        x1 + x2
    ) / 2.0

    cd = (
        d1 + d2
    ) / 2.0


    xi = int(
        np.argmin(
            np.abs(
                x_axis - cx
            )
        )
    )


    di = int(
        np.argmin(
            np.abs(
                depth_axis - cd
            )
        )
    )


    return float(
        anomaly[
            xi,
            di
        ]
    )


def peak_center_distance(
    anomaly_data,
    physical_box
):

    region, x_mask, d_mask = (
        extract_region(
            anomaly_data,
            physical_box
        )
    )


    if region.size == 0:

        return 999.0


    local_idx = np.unravel_index(
        np.argmax(
            region
        ),
        region.shape
    )


    x_values = anomaly_data[
        "x_mm"
    ][
        x_mask
    ]


    d_values = anomaly_data[
        "depth_mm"
    ][
        d_mask
    ]


    peak_x = float(
        x_values[
            local_idx[
                0
            ]
        ]
    )


    peak_d = float(
        d_values[
            local_idx[
                1
            ]
        ]
    )


    x1, d1, x2, d2 = (
        physical_box
    )


    cx = (
        x1 + x2
    ) / 2.0


    cd = (
        d1 + d2
    ) / 2.0


    width = max(
        abs(
            x2 - x1
        ),
        EPS
    )


    height = max(
        abs(
            d2 - d1
        ),
        EPS
    )


    dx_norm = (
        peak_x - cx
    ) / width


    dd_norm = (
        peak_d - cd
    ) / height


    return float(
        math.sqrt(
            dx_norm ** 2
            +
            dd_norm ** 2
        )
    )


# ============================================================
# Stage8 feature extraction
# ============================================================

def extract_candidate_features(
    anomaly_data,
    physical_box
):

    region, _, _ = (
        extract_region(
            anomaly_data,
            physical_box
        )
    )


    if region.size == 0:

        return None


    ring = (
        extract_ring_region(
            anomaly_data,
            physical_box
        )
    )


    values = (
        region.flatten()
    )


    mean_value = float(
        np.mean(
            values
        )
    )


    median_value = float(
        np.median(
            values
        )
    )


    p90 = float(
        np.percentile(
            values,
            90
        )
    )


    p95 = float(
        np.percentile(
            values,
            95
        )
    )


    max_value = float(
        np.max(
            values
        )
    )


    if ring.size > 0:

        ring_mean = float(
            np.mean(
                ring
            )
        )

        ring_p95 = float(
            np.percentile(
                ring,
                95
            )
        )

    else:

        ring_mean = 0.0

        ring_p95 = 0.0


    center_value = (
        center_anomaly_score(
            anomaly_data,
            physical_box
        )
    )


    peak_distance = (
        peak_center_distance(
            anomaly_data,
            physical_box
        )
    )


    occupancy_2 = float(
        np.mean(
            values >= 2.0
        )
    )


    occupancy_3 = float(
        np.mean(
            values >= 3.0
        )
    )


    occupancy_4 = float(
        np.mean(
            values >= 4.0
        )
    )


    mean_contrast = (
        mean_value
        -
        ring_mean
    )


    p95_contrast = (
        p95
        -
        ring_p95
    )


    max_ring_ratio = (
        max_value
        /
        (
            ring_p95
            +
            EPS
        )
    )


    return {

        "anomaly_mean":
            mean_value,

        "anomaly_median":
            median_value,

        "anomaly_p90":
            p90,

        "anomaly_p95":
            p95,

        "anomaly_max":
            max_value,

        "anomaly_center":
            center_value,

        "ring_mean":
            ring_mean,

        "ring_p95":
            ring_p95,

        "mean_contrast":
            mean_contrast,

        "p95_contrast":
            p95_contrast,

        "max_ring_ratio":
            max_ring_ratio,

        "occupancy_gt2":
            occupancy_2,

        "occupancy_gt3":
            occupancy_3,

        "occupancy_gt4":
            occupancy_4,

        "peak_center_distance":
            peak_distance,
    }


# ============================================================
# RF
# ============================================================

def make_rf_vector(row):

    return np.array(
        [[
            float(
                row[
                    feature
                ]
            )
            for feature in FEATURES
        ]],
        dtype=np.float64
    )


def passes_frozen_rescue_filter(
    row
):

    if (
        row[
            "rf_probability"
        ]
        <
        RF_PROBABILITY_MIN
    ):

        return False


    if (
        row[
            "center_depth_mm"
        ]
        >
        FINAL_DEPTH_MAX_MM
    ):

        return False


    if (
        row[
            "peak_center_distance"
        ]
        >
        FINAL_PEAK_DISTANCE_MAX
    ):

        return False


    if (
        row[
            "anomaly_p95"
        ]
        <
        FINAL_ANOMALY_P95_MIN
    ):

        return False


    if (
        row[
            "box_height_mm"
        ]
        >
        FINAL_BOX_HEIGHT_MAX_MM
    ):

        return False


    return True


# ============================================================
# Union-Find
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

        self.rank = [
            0
        ] * n


    def find(self, x):

        while (
            self.parent[x]
            != x
        ):

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x


    def union(
        self,
        a,
        b
    ):

        ra = self.find(a)

        rb = self.find(b)


        if ra == rb:

            return


        if (
            self.rank[
                ra
            ]
            <
            self.rank[
                rb
            ]
        ):

            self.parent[
                ra
            ] = rb

        elif (
            self.rank[
                ra
            ]
            >
            self.rank[
                rb
            ]
        ):

            self.parent[
                rb
            ] = ra

        else:

            self.parent[
                rb
            ] = ra

            self.rank[
                ra
            ] += 1


# ============================================================
# Cluster reliability
# ============================================================

def calculate_reliability(
    member_count,
    source_count,
    max_conf,
    mean_conf
):

    member_score = clamp01(
        member_count
        /
        MEMBER_REFERENCE
    )


    max_conf_score = clamp01(
        max_conf
        /
        MAX_CONF_REFERENCE
    )


    mean_conf_score = clamp01(
        mean_conf
        /
        MEAN_CONF_REFERENCE
    )


    source_score = clamp01(
        source_count
        /
        SOURCE_IMAGE_REFERENCE
    )


    return float(

        WEIGHT_MEMBER
        *
        member_score

        +

        WEIGHT_MAX_CONF
        *
        max_conf_score

        +

        WEIGHT_MEAN_CONF
        *
        mean_conf_score

        +

        WEIGHT_SOURCE_COUNT
        *
        source_score
    )


# ============================================================
# Physical clustering
# ============================================================

def cluster_detections(
    detections
):

    if len(
        detections
    ) == 0:

        return []


    uf = UnionFind(
        len(
            detections
        )
    )


    # ========================================================
    # spatial grouping
    # ========================================================

    for i in range(
        len(
            detections
        )
    ):

        for j in range(
            i + 1,
            len(
                detections
            )
        ):

            a = detections[
                i
            ]

            b = detections[
                j
            ]


            dx = abs(
                a[
                    "center_x_mm"
                ]
                -
                b[
                    "center_x_mm"
                ]
            )


            dd = abs(
                a[
                    "center_depth_mm"
                ]
                -
                b[
                    "center_depth_mm"
                ]
            )


            if (
                dx
                <=
                CLUSTER_X_DISTANCE_MM

                and

                dd
                <=
                CLUSTER_DEPTH_DISTANCE_MM
            ):

                uf.union(
                    i,
                    j
                )


    groups = defaultdict(
        list
    )


    for index, detection in enumerate(
        detections
    ):

        root = uf.find(
            index
        )

        groups[
            root
        ].append(
            detection
        )


    clusters = []


    for cluster_index, members in enumerate(
        groups.values(),
        start=1
    ):


        # ====================================================
        # effective confidence
        #
        # baseline detection:
        #   YOLO conf
        #
        # RF rescue:
        #   RF probability
        #
        # 这样浅层RF补偿目标不会因为原始YOLO conf太低
        # 在cluster reliability里被再次误杀。
        # ====================================================

        effective_confidences = []


        for member in members:

            if (
                member[
                    "source"
                ]
                ==
                "RF_RESCUE"
            ):

                effective_confidences.append(
                    float(
                        member[
                            "rf_probability"
                        ]
                    )
                )

            else:

                effective_confidences.append(
                    float(
                        member[
                            "yolo_conf"
                        ]
                    )
                )


        effective_confidences = np.array(
            effective_confidences,
            dtype=float
        )


        weight_sum = float(
            np.sum(
                effective_confidences
            )
        )


        if weight_sum <= 0:

            weights = np.ones(
                len(
                    members
                ),
                dtype=float
            )

            weights /= len(
                members
            )

        else:

            weights = (
                effective_confidences
                /
                weight_sum
            )


        x_values = np.array([
            member[
                "center_x_mm"
            ]
            for member in members
        ])


        depth_values = np.array([
            member[
                "center_depth_mm"
            ]
            for member in members
        ])


        center_x = float(
            np.sum(
                weights
                *
                x_values
            )
        )


        center_depth = float(
            np.sum(
                weights
                *
                depth_values
            )
        )


        source_images = sorted({
            member[
                "filename"
            ]
            for member in members
        })


        source_count = len(
            source_images
        )


        member_count = len(
            members
        )


        max_conf = float(
            np.max(
                effective_confidences
            )
        )


        mean_conf = float(
            np.mean(
                effective_confidences
            )
        )


        reliability = (
            calculate_reliability(
                member_count,
                source_count,
                max_conf,
                mean_conf
            )
        )


        baseline_members = sum(
            1
            for member in members
            if member[
                "source"
            ]
            ==
            "YOLO_BASELINE"
        )


        rescue_members = sum(
            1
            for member in members
            if member[
                "source"
            ]
            ==
            "RF_RESCUE"
        )


        # ====================================================
        # Final acceptance
        #
        # normal clusters:
        #   reliability >= 0.75
        #
        # rescue-only cluster:
        #   RF已经通过冻结的物理规则，因此保留
        #
        # 这是工程整合策略。
        # ====================================================

        accepted_by_reliability = (
            reliability
            >=
            RELIABILITY_THRESHOLD
        )


        accepted_by_rescue = (
            rescue_members > 0
        )


        final_accept = (
            accepted_by_reliability
            or
            accepted_by_rescue
        )


        clusters.append({

            "cluster_id":
                cluster_index,

            "specimen":
                members[
                    0
                ][
                    "specimen"
                ],

            "rotation":
                members[
                    0
                ][
                    "rotation"
                ],

            "center_x_mm":
                center_x,

            "center_depth_mm":
                center_depth,

            "member_count":
                member_count,

            "source_image_count":
                source_count,

            "baseline_members":
                baseline_members,

            "rescue_members":
                rescue_members,

            "max_effective_confidence":
                max_conf,

            "mean_effective_confidence":
                mean_conf,

            "reliability_score":
                reliability,

            "accepted_by_reliability":
                int(
                    accepted_by_reliability
                ),

            "accepted_by_rescue":
                int(
                    accepted_by_rescue
                ),

            "final_accept":
                int(
                    final_accept
                ),

            "source_images":
                "|".join(
                    source_images
                ),
        })


    return clusters


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "# Stage 9.3 - "
        "Complete internal ultrasonic inference"
    )


    print(
        "\nInput:"
    )

    print(
        INPUT_IMAGE_DIR
    )


    if not INPUT_IMAGE_DIR.exists():

        raise FileNotFoundError(
            INPUT_IMAGE_DIR
        )


    if not RF_MODEL_PATH.exists():

        raise FileNotFoundError(
            RF_MODEL_PATH
        )


    # ========================================================
    # RF
    # ========================================================

    rf_model = joblib.load(
        RF_MODEL_PATH
    )


    if (
        hasattr(
            rf_model,
            "n_features_in_"
        )
        and
        rf_model.n_features_in_
        != len(
            FEATURES
        )
    ):

        raise RuntimeError(
            "RF feature dimension mismatch."
        )


    # ========================================================
    # images
    # ========================================================

    image_paths = list_images(
        INPUT_IMAGE_DIR
    )


    if not image_paths:

        raise RuntimeError(
            "No images found."
        )


    print(
        "\nImages:",
        len(
            image_paths
        )
    )


    all_detections = []


    # ========================================================
    # inference loop
    # ========================================================

    for image_index, image_path in enumerate(
        image_paths,
        start=1
    ):


        print(
            f"[{image_index:02d}/"
            f"{len(image_paths):02d}] "
            f"{image_path.name}"
        )


        specimen = detect_specimen(
            image_path.name
        )


        rotation = detect_rotation(
            image_path.name
        )


        crop_x_start, crop_depth_start = (
            parse_crop_position(
                image_path.name
            )
        )


        image = cv2.imread(
            str(
                image_path
            )
        )


        if image is None:

            raise RuntimeError(
                f"Cannot read image: "
                f"{image_path}"
            )


        image_height, image_width = (
            image.shape[:2]
        )


        yolo_model = (
            get_yolo_model(
                rotation
            )
        )


        results = yolo_model.predict(

            source=
                str(
                    image_path
                ),

            conf=
                INFER_CONF,

            iou=
                YOLO_NMS_IOU,

            imgsz=
                IMAGE_SIZE,

            device=
                DEVICE,

            verbose=
                False,
        )


        if (
            len(
                results
            ) == 0
            or
            results[
                0
            ].boxes
            is None
        ):

            continue


        boxes = results[
            0
        ].boxes


        xyxy_array = (
            boxes.xyxy
            .cpu()
            .numpy()
        )


        conf_array = (
            boxes.conf
            .cpu()
            .numpy()
        )


        for candidate_index, (
            box,
            conf

        ) in enumerate(

            zip(
                xyxy_array,
                conf_array
            ),

            start=1
        ):


            box = [
                float(v)
                for v in box
            ]


            conf = float(
                conf
            )


            physical_box = (
                pixel_box_to_physical(
                    box,
                    crop_x_start,
                    crop_depth_start,
                    image_width,
                    image_height
                )
            )


            (
                x1_mm,
                d1_mm,
                x2_mm,
                d2_mm

            ) = physical_box


            center_x_mm = (
                x1_mm
                +
                x2_mm
            ) / 2.0


            center_depth_mm = (
                d1_mm
                +
                d2_mm
            ) / 2.0


            box_width_mm = abs(
                x2_mm
                -
                x1_mm
            )


            box_height_mm = abs(
                d2_mm
                -
                d1_mm
            )


            # =================================================
            # Normal YOLO detection
            # =================================================

            if conf >= BASE_CONF:

                all_detections.append({

                    "filename":
                        image_path.name,

                    "candidate_index":
                        candidate_index,

                    "specimen":
                        specimen,

                    "rotation":
                        rotation,

                    "source":
                        "YOLO_BASELINE",

                    "yolo_conf":
                        conf,

                    "rf_probability":
                        0.0,

                    "center_x_mm":
                        center_x_mm,

                    "center_depth_mm":
                        center_depth_mm,

                    "box_width_mm":
                        box_width_mm,

                    "box_height_mm":
                        box_height_mm,
                })


                continue


            # =================================================
            # Low-confidence shallow candidate
            # =================================================

            if (
                center_depth_mm
                >
                SHALLOW_MAX_DEPTH_MM
            ):

                continue


            anomaly_data = (
                load_anomaly_map(
                    specimen,
                    rotation
                )
            )


            features = (
                extract_candidate_features(
                    anomaly_data,
                    physical_box
                )
            )


            if features is None:

                continue


            candidate_row = {

                "yolo_conf":
                    conf,

                "center_depth_mm":
                    center_depth_mm,

                "box_width_mm":
                    box_width_mm,

                "box_height_mm":
                    box_height_mm,
            }


            candidate_row.update(
                features
            )


            rf_vector = (
                make_rf_vector(
                    candidate_row
                )
            )


            rf_probability = float(
                rf_model.predict_proba(
                    rf_vector
                )[0, 1]
            )


            candidate_row[
                "rf_probability"
            ] = (
                rf_probability
            )


            # =================================================
            # Frozen rescue filter
            # =================================================

            if not passes_frozen_rescue_filter(
                candidate_row
            ):

                continue


            # =================================================
            # Accepted rescue
            # =================================================

            all_detections.append({

                "filename":
                    image_path.name,

                "candidate_index":
                    candidate_index,

                "specimen":
                    specimen,

                "rotation":
                    rotation,

                "source":
                    "RF_RESCUE",

                "yolo_conf":
                    conf,

                "rf_probability":
                    rf_probability,

                "center_x_mm":
                    center_x_mm,

                "center_depth_mm":
                    center_depth_mm,

                "box_width_mm":
                    box_width_mm,

                "box_height_mm":
                    box_height_mm,
            })


    # ========================================================
    # Save raw accepted detections
    # ========================================================

    if all_detections:

        with open(
            RAW_DETECTION_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_detections[
                        0
                    ].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                all_detections
            )


    # ========================================================
    # Group by specimen + rotation
    # ========================================================

    grouped_detections = defaultdict(
        list
    )


    for detection in (
        all_detections
    ):

        key = (

            detection[
                "specimen"
            ],

            detection[
                "rotation"
            ],
        )


        grouped_detections[
            key
        ].append(
            detection
        )


    all_clusters = []


    for (
        specimen,
        rotation

    ), detections in (
        grouped_detections.items()
    ):


        clusters = (
            cluster_detections(
                detections
            )
        )


        all_clusters.extend(
            clusters
        )


    # ========================================================
    # Save all clusters
    # ========================================================

    if all_clusters:

        with open(
            CLUSTER_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_clusters[
                        0
                    ].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                all_clusters
            )


    # ========================================================
    # Final targets
    # ========================================================

    final_targets = [
        cluster
        for cluster in all_clusters
        if cluster[
            "final_accept"
        ] == 1
    ]


    final_targets.sort(
        key=lambda x:
            (
                x[
                    "specimen"
                ],

                x[
                    "center_x_mm"
                ]
            )
    )


    # ========================================================
    # Assign final target IDs
    # ========================================================

    for index, target in enumerate(
        final_targets,
        start=1
    ):

        target[
            "target_id"
        ] = index


    # ========================================================
    # Save final CSV
    # ========================================================

    if final_targets:

        fieldnames = [
            "target_id"
        ] + [
            key
            for key in final_targets[
                0
            ].keys()
            if key != "target_id"
        ]


        with open(
            FINAL_TARGET_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                final_targets
            )


    # ========================================================
    # JSON output
    # ========================================================

    json_targets = []


    for target in (
        final_targets
    ):

        json_targets.append({

            "target_id":
                int(
                    target[
                        "target_id"
                    ]
                ),

            "specimen":
                target[
                    "specimen"
                ],

            "rotation":
                target[
                    "rotation"
                ],

            "x_mm":
                round(
                    float(
                        target[
                            "center_x_mm"
                        ]
                    ),
                    2
                ),

            "depth_mm":
                round(
                    float(
                        target[
                            "center_depth_mm"
                        ]
                    ),
                    2
                ),

            "reliability":
                round(
                    float(
                        target[
                            "reliability_score"
                        ]
                    ),
                    4
                ),

            "member_count":
                int(
                    target[
                        "member_count"
                    ]
                ),

            "source_image_count":
                int(
                    target[
                        "source_image_count"
                    ]
                ),

            "baseline_members":
                int(
                    target[
                        "baseline_members"
                    ]
                ),

            "rescue_members":
                int(
                    target[
                        "rescue_members"
                    ]
                ),
        })


    final_json_result = {

        "status":
            "success",

        "detected_target_count":
            len(
                json_targets
            ),

        "targets":
            json_targets,

        "engineering_notes": {

            "reliability_threshold":
                RELIABILITY_THRESHOLD,

            "cluster_x_distance_mm":
                CLUSTER_X_DISTANCE_MM,

            "cluster_depth_distance_mm":
                CLUSTER_DEPTH_DISTANCE_MM,

            "base_yolo_conf":
                BASE_CONF,

            "rf_rescue_enabled":
                True,
        }
    }


    with open(
        FINAL_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_json_result,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # Console output
    # ========================================================

    print("\n")
    print("=" * 90)

    print(
        "FINAL INTERNAL TARGET RESULTS"
    )

    print("=" * 90)


    print(
        "\nAccepted crop detections:",
        len(
            all_detections
        )
    )


    print(
        "Physical clusters:",
        len(
            all_clusters
        )
    )


    print(
        "Final targets:",
        len(
            final_targets
        )
    )


    if not final_targets:

        print(
            "\nNo reliable internal target detected."
        )


    else:

        print("")

        for target in (
            final_targets
        ):

            print(
                f"Target {target['target_id']}: "
                f"X={target['center_x_mm']:.2f} mm, "
                f"Depth={target['center_depth_mm']:.2f} mm, "
                f"Reliability="
                f"{target['reliability_score']:.4f}, "
                f"Members={target['member_count']}, "
                f"Baseline={target['baseline_members']}, "
                f"RF-rescue={target['rescue_members']}"
            )


    print("\n")
    print("=" * 90)

    print(
        "Stage 9.3 completed"
    )

    print("=" * 90)


    print(
        "\nSaved:"
    )

    print(
        RAW_DETECTION_CSV
    )

    print(
        CLUSTER_CSV
    )

    print(
        FINAL_TARGET_CSV
    )

    print(
        FINAL_JSON
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "This is the engineering inference pipeline."
    )

    print(
        "It does not use GT labels during inference."
    )

    print(
        "Current input is prepared Stage7 fusion crops."
    )

    print(
        "The next engineering step is to connect "
        "raw 3D ultrasound .npy preprocessing "
        "and expose this result through FastAPI."
    )