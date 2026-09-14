# -*- coding: utf-8 -*-
# ============================================================
# 【重要提醒】请勿删除本文件末尾的 run_internal_inference 函数！
# 该函数是 FastAPI 调用的唯一入口，一旦缺失，云端会报 ImportError。
# ============================================================

from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime_config import get_output_root

BASE_DIR = Path(__file__).resolve().parent
STAGE9_04_SCRIPT = BASE_DIR / "stage9_04_raw_npy_complete_inference.py"

# 输出根目录改到可写区（/tmp 或 ULTRASONIC_OUTPUT_ROOT）
DEFAULT_OUTPUT_ROOT = get_output_root("internal_inference")

# 聚类参数
SEED_X_RADIUS_MM = 100.0
SEED_DEPTH_RADIUS_MM = 60.0
HIGH_CONF_THRESHOLD = 0.30

# 最终过滤阈值
MIN_MEMBER_COUNT = 7
MIN_CONF_MAX = 0.40
MIN_HIGH_CONF_FRACTION = 0.25
MAX_RANGE_ASPECT_RATIO = 8.0
EPS = 1e-9


def normalize_specimen(specimen):
    value = str(specimen).lower().strip()
    if value in ["pk266", "266"]:
        return "pk266"
    if value in ["pk050", "050"]:
        return "pk050"
    raise ValueError("specimen must be pk266 or pk050")


def normalize_rotation(rotation):
    value = str(rotation).lower().strip()
    if value in ["rot00", "shear_rot00"]:
        return "rot00"
    if value in ["rot90", "shear_rot90"]:
        return "rot90"
    raise ValueError("rotation must be rot00 or rot90")


def get_run_paths(input_path, output_root):
    run_name = input_path.stem + "_stage9_04"
    run_root = output_root / run_name
    result_root = run_root / "result"
    work_root = run_root / "work"

    return {
        "run_root": run_root,
        "result_root": result_root,
        "work_root": work_root,
        "raw_detections": result_root / "raw_detections.csv",
        "stage9_04_json": result_root / "final_result.json",
        "visualization": result_root / "internal_targets.png",
    }


def run_stage9_04(input_path, specimen, rotation, output_root):
    if not STAGE9_04_SCRIPT.exists():
        raise FileNotFoundError(
            f"Stage 9.4 script not found: {STAGE9_04_SCRIPT}"
        )

    command = [
        sys.executable,
        str(STAGE9_04_SCRIPT),
        "--input", str(input_path),
        "--output", str(output_root),
        "--specimen", specimen,
        "--rotation", rotation,
    ]

    print("\n" + "=" * 100, flush=True)
    print("STAGE 10 -> RUNNING STAGE 9.4", flush=True)
    print("=" * 100, flush=True)
    print("\nInput:", input_path, flush=True)
    print("Specimen:", specimen, flush=True)
    print("Rotation:", rotation, flush=True)

    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    tail = deque(maxlen=400)
    assert process.stdout is not None

    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        print(f"[STAGE 9.4] {line}", flush=True)

    return_code = process.wait()

    if return_code != 0:
        tail_text = "\n".join(tail)
        raise RuntimeError(
            "Stage 9.4 inference failed. "
            f"Return code = {return_code}. "
            f"Last subprocess output:\n{tail_text}"
        )


def load_detections(csv_path):
    rows = []
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return rows

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["source"]
            yolo_conf = float(row["yolo_conf"])
            rf_probability = float(row["rf_probability"])
            effective_conf = rf_probability if source == "RF_RESCUE" else yolo_conf

            rows.append({
                "filename": row["filename"],
                "candidate_index": int(float(row["candidate_index"])),
                "source": source,
                "yolo_conf": yolo_conf,
                "rf_probability": rf_probability,
                "effective_conf": effective_conf,
                "center_x_mm": float(row["center_x_mm"]),
                "center_depth_mm": float(row["center_depth_mm"]),
            })
    return rows


def seed_centered_clustering(detections):
    if not detections:
        return []

    order = sorted(range(len(detections)), key=lambda index: detections[index]["effective_conf"], reverse=True)
    assigned = np.zeros(len(detections), dtype=bool)
    clusters = []

    for seed_index in order:
        if assigned[seed_index]:
            continue

        seed = detections[seed_index]
        seed_x = seed["center_x_mm"]
        seed_depth = seed["center_depth_mm"]
        member_indices = []

        for candidate_index in order:
            if assigned[candidate_index]:
                continue

            candidate = detections[candidate_index]
            dx = abs(candidate["center_x_mm"] - seed_x)
            dd = abs(candidate["center_depth_mm"] - seed_depth)

            if dx <= SEED_X_RADIUS_MM and dd <= SEED_DEPTH_RADIUS_MM:
                member_indices.append(candidate_index)

        for index in member_indices:
            assigned[index] = True

        clusters.append({
            "seed": seed,
            "members": [detections[index] for index in member_indices],
        })

    return clusters


def analyze_cluster(cluster_id, cluster):
    seed = cluster["seed"]
    members = cluster["members"]

    conf = np.array([member["effective_conf"] for member in members], dtype=float)
    x = np.array([member["center_x_mm"] for member in members], dtype=float)
    depth = np.array([member["center_depth_mm"] for member in members], dtype=float)

    conf_sum = float(conf.sum())
    if conf_sum > 0:
        weights = conf / conf_sum
    else:
        weights = np.ones(len(members), dtype=float) / len(members)

    center_x = float(np.sum(weights * x))
    center_depth = float(np.sum(weights * depth))

    member_count = len(members)
    conf_max = float(np.max(conf))
    conf_mean = float(np.mean(conf))
    conf_median = float(np.median(conf))
    high_conf_fraction = float(np.mean(conf >= HIGH_CONF_THRESHOLD))

    x_min, x_max = float(np.min(x)), float(np.max(x))
    depth_min, depth_max = float(np.min(depth)), float(np.max(depth))
    x_range = x_max - x_min
    depth_range = depth_max - depth_min
    range_aspect_ratio = float((x_range + EPS) / (depth_range + EPS))

    baseline_count = sum(1 for member in members if member["source"] == "YOLO_BASELINE")
    rescue_count = sum(1 for member in members if member["source"] == "RF_RESCUE")

    pass_member_count = member_count >= MIN_MEMBER_COUNT
    pass_conf = conf_max >= MIN_CONF_MAX
    pass_high_fraction = high_conf_fraction >= MIN_HIGH_CONF_FRACTION
    pass_aspect = range_aspect_ratio <= MAX_RANGE_ASPECT_RATIO
    final_accept = pass_member_count and pass_conf and pass_high_fraction and pass_aspect

    return {
        "cluster_id": cluster_id,
        "seed_x_mm": float(seed["center_x_mm"]),
        "seed_depth_mm": float(seed["center_depth_mm"]),
        "seed_conf": float(seed["effective_conf"]),
        "center_x_mm": center_x,
        "center_depth_mm": center_depth,
        "member_count": member_count,
        "baseline_count": baseline_count,
        "rf_rescue_count": rescue_count,
        "conf_max": conf_max,
        "conf_mean": conf_mean,
        "conf_median": conf_median,
        "high_conf_fraction": high_conf_fraction,
        "x_range_mm": x_range,
        "depth_range_mm": depth_range,
        "range_aspect_ratio": range_aspect_ratio,
        "final_accept": int(final_accept),
    }


def save_cluster_csv(rows, result_root):
    result_root.mkdir(parents=True, exist_ok=True)
    output_path = result_root / "stage10_all_clusters.csv"

    if not rows:
        return output_path

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def generate_target_visualization(accepted, result_root, specimen, rotation):
    result_root.mkdir(parents=True, exist_ok=True)
    output_path = result_root / "internal_targets.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 2000)
    ax.set_ylim(300, 0)

    for target_id, row in enumerate(accepted, start=1):
        x = float(row["center_x_mm"])
        depth = float(row["center_depth_mm"])
        confidence = float(row["conf_max"])

        ax.scatter(x, depth, s=180)
        label = f"T{target_id}\nX={x:.1f} mm\nD={depth:.1f} mm\nConf={confidence:.2f}"
        ax.annotate(label, xy=(x, depth), xytext=(12, -10), textcoords="offset points",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.7))

    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"Concrete Internal Ultrasonic Target Detection\n{specimen.upper()} | {rotation.upper()}")
    ax.grid(True, alpha=0.3)
    ax.text(0.01, 0.02, f"Detected targets: {len(accepted)}", transform=ax.transAxes,
            fontsize=10, verticalalignment="bottom")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_result(input_path, specimen, rotation, accepted, visualization_path):
    targets = []
    for target_id, row in enumerate(accepted, start=1):
        targets.append({
            "target_id": target_id,
            "class": "internal_target",
            "x_mm": round(float(row["center_x_mm"]), 2),
            "depth_mm": round(float(row["center_depth_mm"]), 2),
            "confidence": round(float(row["conf_max"]), 4),
            "member_count": int(row["member_count"]),
            "high_conf_fraction": round(float(row["high_conf_fraction"]), 4),
            "range_aspect_ratio": round(float(row["range_aspect_ratio"]), 4),
            "baseline_members": int(row["baseline_count"]),
            "rf_rescue_members": int(row["rf_rescue_count"]),
        })

    return {
        "status": "success",
        "model": "concrete_internal_ultrasonic_target_detector",
        "specimen": specimen,
        "rotation": rotation,
        "input_file": input_path.name,
        "detected_target_count": len(targets),
        "targets": targets,
        "visualization_path": str(visualization_path),
        "scientific_status": (
            "Development pipeline. Rot00 and Rot90 were used during "
            "post-processing development. Independent specimen validation "
            "is still required."
        ),
    }


# ============================================================
# MAIN PUBLIC FUNCTION
# 必须确保 FastAPI 能成功导入这个函数
# ============================================================
def run_internal_inference(input_npy, specimen, rotation, output_root=None, force_rerun=True):
    input_path = Path(input_npy).expanduser()
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() != ".npy":
        raise ValueError("Input file must be .npy")

    specimen = normalize_specimen(specimen)
    rotation = normalize_rotation(rotation)

    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT
    output_root = Path(output_root).expanduser()
    if not output_root.is_absolute():
        output_root = get_output_root("internal_inference") / output_root.name

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        output_root = get_output_root("internal_inference")
        output_root.mkdir(parents=True, exist_ok=True)

    paths = get_run_paths(input_path, output_root)
    existing = paths["raw_detections"].exists()

    if force_rerun or not existing:
        run_stage9_04(input_path, specimen, rotation, output_root)

    if not paths["raw_detections"].exists():
        zero_result = False
        if paths["stage9_04_json"].exists():
            try:
                with open(paths["stage9_04_json"], "r", encoding="utf-8") as f:
                    stage9_result = json.load(f)
                zero_result = int(stage9_result.get("detected_target_count", 0)) == 0
            except Exception:
                zero_result = False

        if not zero_result:
            raise RuntimeError(
                "Stage 9.4 completed but raw_detections.csv is missing, "
                "and final_result.json does not confirm a valid zero-detection run."
            )

    if paths["stage9_04_json"].exists():
        legacy_path = paths["result_root"] / "final_result_stage9_04_legacy.json"
        if not legacy_path.exists():
            shutil.copy2(paths["stage9_04_json"], legacy_path)

    # Stage9.5 postprocessing
    detections = load_detections(paths["raw_detections"])
    clusters = seed_centered_clustering(detections)
    rows = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        rows.append(analyze_cluster(cluster_id, cluster))

    rows.sort(key=lambda row: row["center_x_mm"])
    accepted = [row for row in rows if row["final_accept"] == 1]
    accepted.sort(key=lambda row: row["center_x_mm"])

    save_cluster_csv(rows, paths["result_root"])

    visualization_path = generate_target_visualization(accepted, paths["result_root"], specimen, rotation)
    result = build_result(input_path, specimen, rotation, accepted, visualization_path)

    final_json_path = paths["result_root"] / "final_result.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    result["result_json"] = str(final_json_path)

    print("\n" + "=" * 100)
    print("STAGE 10 INTERNAL INFERENCE COMPLETE")
    print("=" * 100)
    print("\nDetected targets:", len(accepted))
    print("Visualization:", visualization_path)
    print("JSON:", final_json_path)

    return result