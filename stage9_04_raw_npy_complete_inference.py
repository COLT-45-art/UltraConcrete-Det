from pathlib import Path
from runtime_config import DEVICE, project_path, get_output_root
import argparse
import importlib.util
import json
import shutil
import csv
import os
import time

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter

# ============================================================
# 全局加载 Stage 9.3 模块（只加载一次）
# ============================================================
STAGE9_03_CACHE = None


def get_stage9_03():
    global STAGE9_03_CACHE
    if STAGE9_03_CACHE is not None:
        return STAGE9_03_CACHE

    if not STAGE9_03_PATH.exists():
        raise FileNotFoundError(f"Stage 9.3 script not found: {STAGE9_03_PATH}")

    spec = importlib.util.spec_from_file_location("stage9_03_module", STAGE9_03_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for Stage 9.3: {STAGE9_03_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    STAGE9_03_CACHE = module
    log("[CACHE] Stage 9.3 module loaded and cached globally.")
    return STAGE9_03_CACHE
BASE_DIR = Path(__file__).resolve().parent

STAGE9_03_PATH = BASE_DIR / "stage9_03_complete_inference.py"
DEFAULT_OUTPUT_ROOT = get_output_root("raw_npy_complete_inference")
FS = 2_000_000.0
VALID_Y_START = 5
VALID_Y_END = 76
X_MIN_MM = 0.0
X_MAX_MM = 2000.0
RAW_DEPTH_MIN_MM = 20.0
RAW_DEPTH_MAX_MM = 300.0
RAW_DEPTH_STEP_MM = 1.0
RAW_DEPTH_AXIS = np.arange(RAW_DEPTH_MIN_MM, RAW_DEPTH_MAX_MM + RAW_DEPTH_STEP_MM, RAW_DEPTH_STEP_MM, dtype=np.float32)
CROP_X_WIDTH_MM = 400.0
CROP_DEPTH_HEIGHT_MM = 140.0
RAW_OUTPUT_WIDTH = 640
RAW_OUTPUT_HEIGHT = 224
ROBUST_OUTPUT_SIZE = 416
Y_STRONG_PERCENTILE = 96.0
INFER_STRIDE_X_MM = 60.0
INFER_STRIDE_DEPTH_MM = 30.0

RAW_CALIBRATION = {
    ("pk050", "shear_rot00"): {"velocity": 3434.9, "time_zero": 61.237},
    ("pk050", "shear_rot90"): {"velocity": 3424.7, "time_zero": 59.909},
    ("pk266", "shear_rot00"): {"velocity": 2830.7, "time_zero": 57.540},
    ("pk266", "shear_rot90"): {"velocity": 2951.0526, "time_zero": 78.7350},
}

ROBUST_CALIBRATION = {
    "shear_rot00": {"velocity": 2830.7, "time_zero": 57.540},
    "shear_rot90": {"velocity": 2951.0526, "time_zero": 78.7350},
}

MAD_SCALE = 1.4826
EPS = 1e-6
SMOOTH_SIGMA_X = 1.0
SMOOTH_SIGMA_DEPTH = 1.0
TANH_SCALE = 3.0
ROBUST_DEPTH_MIN_MM = 0.0
ROBUST_DEPTH_MAX_MM = 320.0
DX_MM = 10.0


def log(*parts):
    """Flush progress immediately so deployment logs do not look frozen."""
    print(*parts, flush=True)


def env_int(name, default, minimum=None):
    """Read a positive/integer environment option safely."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, got {value}")
    return value


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def make_window_starts(minimum, maximum, window, stride):
    starts = []
    current = float(minimum)
    while current + window <= maximum + 1e-6:
        starts.append(round(current, 6))
        current += stride
    final_start = maximum - window
    if len(starts) == 0 or abs(starts[-1] - final_start) > 1e-6:
        starts.append(round(final_start, 6))
    return starts


def detect_from_filename(path):
    name = path.name.lower()
    specimen = "pk050" if "pk050" in name else ("pk266" if "pk266" in name else None)
    rotation = "shear_rot00" if "rot00" in name else ("shear_rot90" if "rot90" in name else None)
    return specimen, rotation


def normalize_specimen(value):
    if value is None:
        return None
    value = value.lower().strip()
    if value in {"pk050", "050"}:
        return "pk050"
    if value in {"pk266", "266"}:
        return "pk266"
    raise ValueError("--specimen must be pk050 or pk266")


def normalize_rotation(value):
    if value is None:
        return None
    value = value.lower().strip()
    if value in {"rot00", "shear_rot00"}:
        return "shear_rot00"
    if value in {"rot90", "shear_rot90"}:
        return "shear_rot90"
    raise ValueError("--rotation must be rot00 or rot90")


def build_raw_feature_map(npy_path, velocity, time_zero):
    """
    Build the RAW feature map without materialising the whole 3-D Hilbert
    envelope at once.

    Hilbert is still evaluated on the full A-scan for every trace, so the
    signal processing remains equivalent. Only the sample range needed by
    20-300 mm (plus a generous Gaussian margin) is retained in memory.
    """
    started = time.perf_counter()
    log("\n[RAW] Loading:", npy_path)
    data = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    log("[RAW] Raw shape:", data.shape, "dtype:", data.dtype)

    if data.ndim != 3:
        raise ValueError(f"Input volume must be 3D, got {data.shape}")
    if data.shape[1] < VALID_Y_END:
        raise ValueError(
            f"Input Y dimension is too small: {data.shape[1]} < required {VALID_Y_END}"
        )
    if data.shape[2] < 2:
        raise ValueError("Input A-scan length must be at least 2 samples")

    sample_axis = (
        time_zero + 2.0 * (RAW_DEPTH_AXIS / 1000.0) * FS / velocity
    ).astype(np.float32)
    idx0 = np.floor(sample_axis).astype(np.int32)
    idx1 = idx0 + 1
    if idx0.min() < 0 or idx1.max() >= data.shape[2]:
        raise ValueError(
            "RAW physical-depth samples exceed the input A-scan range: "
            f"needed {int(idx0.min())}..{int(idx1.max())}, available 0..{data.shape[2] - 1}"
        )

    # 8 sigma on the sample axis is effectively identical to filtering the
    # full 4000-sample envelope around the retained physical-depth interval.
    margin = env_int("RAW_SAMPLE_MARGIN", 24, minimum=12)
    sample_start = max(0, int(idx0.min()) - margin)
    sample_stop = min(data.shape[2], int(idx1.max()) + margin + 1)  # exclusive

    nx = data.shape[0]
    ny = VALID_Y_END - VALID_Y_START
    local_len = sample_stop - sample_start
    envelope_local = np.empty((nx, ny, local_len), dtype=np.float32)

    x_chunk = env_int("RAW_X_CHUNK", 16, minimum=1)
    total_chunks = (nx + x_chunk - 1) // x_chunk
    log(
        f"[RAW] Hilbert in {total_chunks} chunk(s), x_chunk={x_chunk}, "
        f"keeping samples [{sample_start}:{sample_stop})"
    )

    for chunk_no, x0 in enumerate(range(0, nx, x_chunk), start=1):
        x1 = min(nx, x0 + x_chunk)
        # Only the current X chunk is copied to RAM.
        chunk = np.asarray(
            data[x0:x1, VALID_Y_START:VALID_Y_END, :],
            dtype=np.float32,
        )
        analytic = hilbert(chunk, axis=2)
        envelope_local[x0:x1] = np.abs(
            analytic[:, :, sample_start:sample_stop]
        ).astype(np.float32, copy=False)
        del analytic, chunk

        if chunk_no == 1 or chunk_no == total_chunks or chunk_no % 3 == 0:
            log(f"[RAW] Hilbert progress: {chunk_no}/{total_chunks}")

    log("[RAW] Building background/enhancement map...")
    background = np.median(envelope_local, axis=0, keepdims=True).astype(np.float32)
    envelope_local -= background
    np.maximum(envelope_local, 0, out=envelope_local)
    del background

    # In-place output keeps peak RAM much lower than the original implementation.
    gaussian_filter(
        envelope_local,
        sigma=(1.0, 0.6, 3.0),
        output=envelope_local,
    )

    local_idx0 = idx0 - sample_start
    local_idx1 = idx1 - sample_start
    weight = (sample_axis - idx0).astype(np.float32)
    depth_volume = (
        envelope_local[:, :, local_idx0] * (1.0 - weight[None, None, :])
        + envelope_local[:, :, local_idx1] * weight[None, None, :]
    ).astype(np.float32)
    del envelope_local

    log("[RAW] Interpolated depth volume:", depth_volume.shape)

    intensity_map = np.median(depth_volume, axis=1).astype(np.float32)
    num_y = depth_volume.shape[1]
    persistence_count = np.zeros(
        (depth_volume.shape[0], depth_volume.shape[2]), dtype=np.float32
    )
    for y in range(num_y):
        slice_data = depth_volume[:, y, :]
        positive = slice_data[slice_data > 0]
        if positive.size == 0:
            continue
        threshold = np.percentile(positive, Y_STRONG_PERCENTILE)
        persistence_count += (slice_data >= threshold).astype(np.float32)
    persistence_map = persistence_count / float(num_y)

    intensity_log = np.log1p(intensity_map)
    local_background = gaussian_filter(intensity_log, sigma=(8, 15))
    contrast_map = intensity_log - local_background
    contrast_map[contrast_map < 0] = 0

    feature_map = (
        0.35 * normalize(intensity_log)
        + 0.45 * normalize(persistence_map)
        + 0.20 * normalize(contrast_map)
    )
    feature_map = gaussian_filter(feature_map, sigma=(1.2, 1.5))
    feature_map = normalize(feature_map).astype(np.float32)
    log(
        "[RAW] Feature map:",
        feature_map.shape,
        f"elapsed={time.perf_counter() - started:.1f}s",
    )
    return feature_map


def robust_depth_mm_to_sample(depth_mm, velocity, time_zero):
    return int(round(time_zero + 2.0 * FS * (depth_mm / 1000.0) / velocity))


def robust_sample_to_depth_mm(sample, velocity, time_zero):
    return velocity * (sample - time_zero) / (2.0 * FS) * 1000.0


def build_robust_anomaly_map(npy_path, velocity, time_zero):
    started = time.perf_counter()
    log("\n[ROBUST] Loading:", npy_path)
    volume = np.load(npy_path, mmap_mode="r", allow_pickle=False)

    if volume.ndim != 3:
        raise ValueError(f"Input volume must be 3D, got {volume.shape}")
    if volume.shape[1] < VALID_Y_END:
        raise ValueError(
            f"Input Y dimension is too small: {volume.shape[1]} < required {VALID_Y_END}"
        )

    volume_valid = volume[:, VALID_Y_START:VALID_Y_END, :]
    log("[ROBUST] Averaging Y and computing Hilbert envelope...")
    mean_ascans = np.mean(volume_valid, axis=1)
    envelope = np.abs(hilbert(mean_ascans, axis=1)).astype(np.float32)
    del mean_ascans

    sample_min = max(
        0, robust_depth_mm_to_sample(ROBUST_DEPTH_MIN_MM, velocity, time_zero)
    )
    sample_max = min(
        envelope.shape[1] - 1,
        robust_depth_mm_to_sample(ROBUST_DEPTH_MAX_MM, velocity, time_zero),
    )
    if sample_min >= sample_max:
        raise ValueError(
            f"Invalid robust sample range: {sample_min}..{sample_max} "
            f"for A-scan length {envelope.shape[1]}"
        )
    envelope_local = envelope[:, sample_min : sample_max + 1]
    del envelope

    median_depth = np.median(envelope_local, axis=0)
    abs_dev = np.abs(envelope_local - median_depth[None, :])
    mad_depth = np.median(abs_dev, axis=0)
    robust_sigma = MAD_SCALE * mad_depth + EPS
    zmap = (envelope_local - median_depth[None, :]) / robust_sigma[None, :]
    positive = np.maximum(zmap, 0)
    anomaly = gaussian_filter(
        positive, sigma=(SMOOTH_SIGMA_X, SMOOTH_SIGMA_DEPTH)
    ).astype(np.float32)

    samples = np.arange(sample_min, sample_max + 1, dtype=np.float32)
    depths_mm = (
        velocity * (samples - float(time_zero)) / (2.0 * FS) * 1000.0
    ).astype(np.float32)
    # Equivalent to 10 mm spacing for the normal 201-position scans, while
    # remaining consistent with RAW X coordinates if a different X count appears.
    x_mm = np.linspace(
        X_MIN_MM, X_MAX_MM, anomaly.shape[0], dtype=np.float32
    )

    log(
        "[ROBUST] Anomaly map:",
        anomaly.shape,
        f"elapsed={time.perf_counter() - started:.1f}s",
    )
    return anomaly, x_mm, depths_mm


def crop_raw_feature(feature_map, x_start_mm, depth_start_mm):
    x_axis = np.linspace(X_MIN_MM, X_MAX_MM, feature_map.shape[0])
    x_end_mm = x_start_mm + CROP_X_WIDTH_MM
    depth_end_mm = depth_start_mm + CROP_DEPTH_HEIGHT_MM
    x_indices = np.where((x_axis >= x_start_mm) & (x_axis <= x_end_mm))[0]
    d_indices = np.where((RAW_DEPTH_AXIS >= depth_start_mm) & (RAW_DEPTH_AXIS <= depth_end_mm))[0]
    if x_indices.size == 0 or d_indices.size == 0:
        raise RuntimeError(f"Empty RAW crop x={x_start_mm} d={depth_start_mm}")
    return feature_map[x_indices[0]:x_indices[-1] + 1, d_indices[0]:d_indices[-1] + 1]


def raw_crop_to_image(crop):
    crop = normalize(crop)
    image_uint8 = (crop * 255.0).clip(0, 255).astype(np.uint8).T
    image = Image.fromarray(image_uint8, mode="L")
    image = image.resize((RAW_OUTPUT_WIDTH, RAW_OUTPUT_HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def crop_robust_map(anomaly, x_axis, depth_axis, x_start_mm, depth_start_mm):
    x_end_mm = x_start_mm + CROP_X_WIDTH_MM
    depth_end_mm = depth_start_mm + CROP_DEPTH_HEIGHT_MM
    x_mask = (x_axis >= x_start_mm) & (x_axis <= x_end_mm)
    d_mask = (depth_axis >= depth_start_mm) & (depth_axis <= depth_end_mm)
    crop = anomaly[x_mask, :][:, d_mask]
    if crop.size == 0:
        raise RuntimeError(f"Empty ROBUST crop x={x_start_mm} d={depth_start_mm}")
    return crop


def robust_crop_to_image(crop):
    positive = np.maximum(crop, 0)
    compressed = np.tanh(positive / TANH_SCALE)
    image = (compressed * 255.0).astype(np.uint8).T
    return cv2.resize(image, (ROBUST_OUTPUT_SIZE, ROBUST_OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)


def build_fusion_image(raw_gray, robust_gray):
    if raw_gray.shape != robust_gray.shape:
        robust_gray = cv2.resize(robust_gray, (raw_gray.shape[1], raw_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    raw_f = raw_gray.astype(np.float32)
    robust_f = robust_gray.astype(np.float32)
    fusion_mean = np.clip((raw_f + robust_f) / 2.0, 0, 255).astype(np.uint8)
    return cv2.merge([raw_gray, robust_gray, fusion_mean])


def make_crop_filename(specimen, rotation, index, x_start, d_start):
    rot_text = "rot00" if rotation == "shear_rot00" else "rot90"
    return f"{specimen}_shear_{rot_text}_infer_{index:04d}_x{int(round(x_start)):04d}_d{int(round(d_start)):03d}.png"


def run_final_inference(s93, fusion_dir, specimen, rotation, anomaly_data, output_root):
    started = time.perf_counter()
    log("\n" + "=" * 90)
    log("STAGE 9.3 ENGINEERING INFERENCE")
    log("=" * 90)

    log("[INFER] Loading Random Forest...")
    rf_model = joblib.load(s93.RF_MODEL_PATH)

    log(f"[INFER] Loading YOLO model for {rotation} on device={DEVICE}...")
    yolo_model = s93.get_yolo_model(rotation)

    image_paths = sorted(fusion_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No fusion crop images found in: {fusion_dir}")

    all_detections = []
    default_batch = 4 if str(DEVICE).lower().startswith("cpu") else 16
    batch_size = env_int("YOLO_BATCH_SIZE", default_batch, minimum=1)
    log(f"[INFER] Images={len(image_paths)}, YOLO_BATCH_SIZE={batch_size}")

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        first_index = batch_start + 1
        last_index = batch_start + len(batch_paths)
        log(f"[INFER] Predicting {first_index}-{last_index}/{len(image_paths)}")

        source_list = [str(path) for path in batch_paths]
        try:
            batch_results = yolo_model.predict(
                source=source_list,
                conf=s93.INFER_CONF,
                iou=s93.YOLO_NMS_IOU,
                imgsz=s93.IMAGE_SIZE,
                device=DEVICE,
                batch=batch_size,
                verbose=False,
            )
        except TypeError:
            # Compatibility fallback for older Ultralytics versions.
            batch_results = yolo_model.predict(
                source=source_list,
                conf=s93.INFER_CONF,
                iou=s93.YOLO_NMS_IOU,
                imgsz=s93.IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )

        if len(batch_results) != len(batch_paths):
            raise RuntimeError(
                f"YOLO returned {len(batch_results)} result(s) for "
                f"{len(batch_paths)} input image(s)"
            )

        for image_path, result in zip(batch_paths, batch_results):
            crop_x_start, crop_depth_start = s93.parse_crop_position(image_path.name)

            if getattr(result, "orig_shape", None) is not None:
                image_height, image_width = map(int, result.orig_shape)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Cannot read fusion crop: {image_path}")
                image_height, image_width = image.shape[:2]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy_array = result.boxes.xyxy.cpu().numpy()
            conf_array = result.boxes.conf.cpu().numpy()

            for candidate_index, (box, conf) in enumerate(
                zip(xyxy_array, conf_array), start=1
            ):
                box = [float(v) for v in box]
                conf = float(conf)
                physical_box = s93.pixel_box_to_physical(
                    box,
                    crop_x_start,
                    crop_depth_start,
                    image_width,
                    image_height,
                )
                x1_mm, d1_mm, x2_mm, d2_mm = physical_box
                center_x_mm = (x1_mm + x2_mm) / 2.0
                center_depth_mm = (d1_mm + d2_mm) / 2.0
                box_width_mm = abs(x2_mm - x1_mm)
                box_height_mm = abs(d2_mm - d1_mm)

                if conf >= s93.BASE_CONF:
                    all_detections.append(
                        {
                            "filename": image_path.name,
                            "candidate_index": candidate_index,
                            "specimen": specimen,
                            "rotation": rotation,
                            "source": "YOLO_BASELINE",
                            "yolo_conf": conf,
                            "rf_probability": 0.0,
                            "center_x_mm": center_x_mm,
                            "center_depth_mm": center_depth_mm,
                            "box_width_mm": box_width_mm,
                            "box_height_mm": box_height_mm,
                        }
                    )
                    continue

                if center_depth_mm > s93.SHALLOW_MAX_DEPTH_MM:
                    continue

                features = s93.extract_candidate_features(
                    anomaly_data, physical_box
                )
                if features is None:
                    continue

                candidate_row = {
                    "yolo_conf": conf,
                    "center_depth_mm": center_depth_mm,
                    "box_width_mm": box_width_mm,
                    "box_height_mm": box_height_mm,
                }
                candidate_row.update(features)
                rf_vector = s93.make_rf_vector(candidate_row)
                rf_probability = float(rf_model.predict_proba(rf_vector)[0, 1])
                candidate_row["rf_probability"] = rf_probability

                if not s93.passes_frozen_rescue_filter(candidate_row):
                    continue

                all_detections.append(
                    {
                        "filename": image_path.name,
                        "candidate_index": candidate_index,
                        "specimen": specimen,
                        "rotation": rotation,
                        "source": "RF_RESCUE",
                        "yolo_conf": conf,
                        "rf_probability": rf_probability,
                        "center_x_mm": center_x_mm,
                        "center_depth_mm": center_depth_mm,
                        "box_width_mm": box_width_mm,
                        "box_height_mm": box_height_mm,
                    }
                )

    clusters = s93.cluster_detections(all_detections)
    final_targets = [c for c in clusters if c["final_accept"] == 1]
    final_targets.sort(key=lambda x: x["center_x_mm"])
    for index, target in enumerate(final_targets, start=1):
        target["target_id"] = index

    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv = output_root / "raw_detections.csv"
    cluster_csv = output_root / "physical_clusters.csv"
    final_csv = output_root / "final_targets.csv"
    final_json = output_root / "final_result.json"

    if all_detections:
        with open(raw_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_detections[0].keys())
            )
            writer.writeheader()
            writer.writerows(all_detections)

    if clusters:
        with open(cluster_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(clusters[0].keys()))
            writer.writeheader()
            writer.writerows(clusters)

    if final_targets:
        fields = ["target_id"] + [
            k for k in final_targets[0].keys() if k != "target_id"
        ]
        with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(final_targets)

    json_targets = []
    for target in final_targets:
        json_targets.append(
            {
                "target_id": int(target["target_id"]),
                "x_mm": round(float(target["center_x_mm"]), 2),
                "depth_mm": round(float(target["center_depth_mm"]), 2),
                "reliability": round(
                    float(target["reliability_score"]), 4
                ),
                "member_count": int(target["member_count"]),
                "source_image_count": int(target["source_image_count"]),
                "baseline_members": int(target["baseline_members"]),
                "rescue_members": int(target["rescue_members"]),
            }
        )

    result = {
        "status": "success",
        "input_mode": "raw_3d_npy",
        "specimen": specimen,
        "rotation": rotation,
        "detected_target_count": len(json_targets),
        "targets": json_targets,
        "engineering_notes": {
            "gt_used": False,
            "crop_policy": "dense_gt_free_sliding_window",
            "x_stride_mm": INFER_STRIDE_X_MM,
            "depth_stride_mm": INFER_STRIDE_DEPTH_MM,
            "reliability_threshold": s93.RELIABILITY_THRESHOLD,
            "warning": (
                "The deployment sliding grid is an engineering generalization "
                "of the target-centered Stage 3 positive crops and needs "
                "external validation."
            ),
        },
    }
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    log(
        f"[INFER] Finished: detections={len(all_detections)}, "
        f"clusters={len(clusters)}, targets={len(final_targets)}, "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    return all_detections, clusters, final_targets, final_json


def main():
    total_started = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Raw 3D ultrasonic .npy -> internal target JSON"
    )
    parser.add_argument("--input", required=True, help="Path to raw 3D .npy")
    parser.add_argument(
        "--specimen",
        default=None,
        help="pk050 or pk266; auto-detected from filename when possible",
    )
    parser.add_argument(
        "--rotation",
        default=None,
        help="rot00 or rot90; auto-detected from filename when possible",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory",
    )
    parser.add_argument(
        "--keep-crops",
        action="store_true",
        help="Keep generated intermediate maps/crop images",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. 0/unset means full scan. "
            "Environment variable MAX_CROPS is also supported."
        ),
    )
    args = parser.parse_args()

    npy_path = project_path(args.input)
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)
    if not npy_path.is_file():
        raise ValueError(f"Input is not a file: {npy_path}")
    if npy_path.suffix.lower() != ".npy":
        raise ValueError("Input must be a .npy file")

    auto_specimen, auto_rotation = detect_from_filename(npy_path)
    specimen = normalize_specimen(args.specimen) or auto_specimen
    rotation = normalize_rotation(args.rotation) or auto_rotation
    if specimen is None:
        raise ValueError("Cannot detect specimen. Add --specimen pk050 or pk266")
    if rotation is None:
        raise ValueError("Cannot detect rotation. Add --rotation rot00 or rot90")

    # IMPORTANT: keep this assignment outside the `if rotation is None` block.
    # The previous bad indentation caused UnboundLocalError on deployment.
    output_root = project_path(args.output)

    # Try requested output path first; if the mounted project is read-only,
    # fall back to runtime_config's writable area.
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        output_root = Path(get_output_root("raw_npy_complete_inference"))
        output_root.mkdir(parents=True, exist_ok=True)
        log(
            f"[WARN] Output directory is not writable ({exc}); "
            f"falling back to: {output_root}"
        )

    run_root = output_root / f"{npy_path.stem}_stage9_04"
    work_root = run_root / "work"
    raw_crop_dir = work_root / "raw_crops"
    robust_crop_dir = work_root / "robust_crops"
    fusion_crop_dir = work_root / "fusion_crops"
    final_output_dir = run_root / "result"

    if run_root.exists():
        log("[SETUP] Removing previous run:", run_root)
        shutil.rmtree(run_root)

    for directory in (
        raw_crop_dir,
        robust_crop_dir,
        fusion_crop_dir,
        final_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    log("=" * 90)
    log("Stage 9.4 - RAW NPY COMPLETE INFERENCE")
    log("=" * 90)
    log("Input:", npy_path)
    log("Specimen:", specimen)
    log("Rotation:", rotation)
    log("GT used: False")
    log("Inference device:", DEVICE)
    log("Output:", run_root)

    raw_cfg = RAW_CALIBRATION[(specimen, rotation)]
    robust_cfg = ROBUST_CALIBRATION[rotation]

    if specimen == "pk050":
        log(
            "[WARN] ROBUST_CALIBRATION is rotation-only and currently uses "
            "the same values as PK266. This behavior is preserved for model "
            "compatibility; verify it against the Stage 9.3/training pipeline "
            "before changing calibration."
        )

    try:
        log("\n[PHASE 1/4] Building RAW feature map...")
        raw_feature_map = build_raw_feature_map(
            npy_path, raw_cfg["velocity"], raw_cfg["time_zero"]
        )
        if args.keep_crops:
            np.save(work_root / "raw_feature_map.npy", raw_feature_map)

        log("\n[PHASE 2/4] Building robust anomaly map...")
        anomaly, anomaly_x_mm, anomaly_depth_mm = build_robust_anomaly_map(
            npy_path, robust_cfg["velocity"], robust_cfg["time_zero"]
        )
        if args.keep_crops:
            np.save(work_root / "robust_anomaly.npy", anomaly)
            np.save(work_root / "robust_x_mm.npy", anomaly_x_mm)
            np.save(work_root / "robust_depth_mm.npy", anomaly_depth_mm)

        x_starts = make_window_starts(
            X_MIN_MM, X_MAX_MM, CROP_X_WIDTH_MM, INFER_STRIDE_X_MM
        )
        depth_starts = make_window_starts(
            RAW_DEPTH_MIN_MM,
            RAW_DEPTH_MAX_MM,
            CROP_DEPTH_HEIGHT_MM,
            INFER_STRIDE_DEPTH_MM,
        )
        full_crop_count = len(x_starts) * len(depth_starts)
        log(
            f"\n[PHASE 3/4] GT-free crops: {len(x_starts)} x "
            f"{len(depth_starts)} = {full_crop_count}"
        )

        if args.max_crops is None:
            max_crops = env_int("MAX_CROPS", 0, minimum=0)
        else:
            if args.max_crops < 0:
                raise ValueError("--max-crops must be >= 0")
            max_crops = args.max_crops

        if max_crops > 0:
            log(
                f"[LIMIT] MAX_CROPS={max_crops}; this is a debugging limit "
                "and does not scan the full specimen."
            )

        crop_index = 0
        stop_flag = False

        for x_start in x_starts:
            if stop_flag:
                break

            for depth_start in depth_starts:
                if max_crops > 0 and crop_index >= max_crops:
                    stop_flag = True
                    break

                crop_index += 1
                raw_gray = raw_crop_to_image(
                    crop_raw_feature(
                        raw_feature_map, x_start, depth_start
                    )
                )
                robust_gray = robust_crop_to_image(
                    crop_robust_map(
                        anomaly,
                        anomaly_x_mm,
                        anomaly_depth_mm,
                        x_start,
                        depth_start,
                    )
                )
                fusion = build_fusion_image(raw_gray, robust_gray)
                filename = make_crop_filename(
                    specimen,
                    rotation,
                    crop_index,
                    x_start,
                    depth_start,
                )

                # RAW/ROBUST PNGs are debug artifacts only. Avoid unnecessary
                # disk I/O in deployment when --keep-crops is not requested.
                if args.keep_crops:
                    if not cv2.imwrite(
                        str(raw_crop_dir / filename), raw_gray
                    ):
                        raise RuntimeError(
                            f"Failed to save RAW crop: {filename}"
                        )
                    if not cv2.imwrite(
                        str(robust_crop_dir / filename), robust_gray
                    ):
                        raise RuntimeError(
                            f"Failed to save ROBUST crop: {filename}"
                        )

                if not cv2.imwrite(
                    str(fusion_crop_dir / filename), fusion
                ):
                    raise RuntimeError(
                        f"Failed to save fusion crop: {filename}"
                    )

                if (
                    crop_index == 1
                    or crop_index % 25 == 0
                    or crop_index == min(full_crop_count, max_crops or full_crop_count)
                ):
                    log(
                        f"[CROP] {crop_index}/"
                        f"{min(full_crop_count, max_crops or full_crop_count)}"
                    )

        if crop_index == 0:
            raise RuntimeError("No inference crops were generated")

        log(
            f"[PHASE 3/4] Generated {crop_index} fusion crop(s)."
        )

        log("\n[PHASE 4/4] Running Stage 9.3/YOLO/RF inference...")
        s93 = get_stage9_03()
        anomaly_data = {
            "anomaly": anomaly,
            "x_mm": anomaly_x_mm,
            "depth_mm": anomaly_depth_mm,
        }
        (
            all_detections,
            clusters,
            final_targets,
            final_json,
        ) = run_final_inference(
            s93,
            fusion_crop_dir,
            specimen,
            rotation,
            anomaly_data,
            final_output_dir,
        )

        log("\n" + "=" * 90)
        log("FINAL INTERNAL TARGET RESULTS")
        log("=" * 90)
        log("Accepted crop detections:", len(all_detections))
        log("Physical clusters:", len(clusters))
        log("Final targets:", len(final_targets))
        for target in final_targets:
            log(
                f"Target {target['target_id']}: "
                f"X={target['center_x_mm']:.2f} mm, "
                f"Depth={target['center_depth_mm']:.2f} mm, "
                f"Reliability={target['reliability_score']:.4f}, "
                f"Baseline={target['baseline_members']}, "
                f"RF-rescue={target['rescue_members']}"
            )
        log("Final JSON:", final_json)

    except Exception:
        # Stage 10 uploads use unique filenames. Without cleanup, each failed
        # request can leave a new set of PNGs in /tmp and eventually fill disk.
        if not args.keep_crops:
            shutil.rmtree(work_root, ignore_errors=True)
            log("[CLEANUP] Removed temporary work files after failure.")
        raise

    if not args.keep_crops:
        shutil.rmtree(work_root, ignore_errors=True)
        log("[CLEANUP] Temporary maps/crop images removed.")

    log(
        "\nScientific note: Stage 3 positive training crops were "
        "target-centered using labels. Stage 9.4 replaces that with a dense "
        "GT-free sliding grid, so this raw-NPY pipeline must be checked on "
        "the known Rot00/Rot90 volumes and later on independent specimens "
        "before deployment claims."
    )
    log(f"[DONE] Total elapsed: {time.perf_counter() - total_started:.1f}s")


if __name__ == "__main__":
    main()
