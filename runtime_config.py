"""Shared deployment configuration; inference thresholds remain in Stage 9."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch is required for ultrasonic AI inference but could not be imported. "
        "Install a torch/torchvision pair compatible with this environment. "
        f"Original import error: {exc}"
    ) from exc

_requested = os.environ.get("INFERENCE_DEVICE", "auto").strip().lower()

if _requested == "auto":
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
elif _requested == "cpu":
    DEVICE = "cpu"
elif _requested in {"0", "cuda", "cuda:0"}:
    if not torch.cuda.is_available():
        raise RuntimeError("INFERENCE_DEVICE requests CUDA, but CUDA is unavailable")
    DEVICE = 0
else:
    raise ValueError("INFERENCE_DEVICE must be auto, cpu, 0, cuda or cuda:0")

print(
    f"[RUNTIME] torch={torch.__version__}, "
    f"cuda_available={torch.cuda.is_available()}, device={DEVICE}",
    flush=True,
)


def project_path(value):
    """Resolve relative CLI/service paths against this project, not shell cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


# ============================================================
# 可写目录工具（创空间部署用）
# 说明：
#   创空间的代码仓库目录通常是只读的，
#   所有运行时产物（上传文件、中间图、结果图）必须写到可写目录。
#   优先使用环境变量 ULTRASONIC_OUTPUT_ROOT，否则落到 /tmp。
# ============================================================

def get_writable_root():
    """返回当前环境可写的根目录。"""
    env_root = os.environ.get("ULTRASONIC_OUTPUT_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
    else:
        root = Path("/tmp") / "ultrasonic_outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_output_root(subdir=""):
    """获取某个子目录的可写输出路径，自动创建。"""
    root = get_writable_root()
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root
