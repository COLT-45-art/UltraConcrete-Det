# -*- coding: utf-8 -*-

import os
import re
import time
import uuid
import shutil
import asyncio
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from internal_inference_service import run_internal_inference
from runtime_config import get_output_root

BASE_DIR = Path(__file__).resolve().parent
STATIC_ROOT = BASE_DIR / "static"
UPLOAD_ROOT = get_output_root("uploads")
OUTPUT_ROOT = get_output_root("outputs")

print(f"[PATH] UPLOAD_ROOT = {UPLOAD_ROOT}", flush=True)
print(f"[PATH] OUTPUT_ROOT = {OUTPUT_ROOT}", flush=True)
print(f"[PATH] STATIC_ROOT = {STATIC_ROOT}", flush=True)

if not STATIC_ROOT.exists():
    print(f"[WARN] static 目录不存在: {STATIC_ROOT}", flush=True)

app = FastAPI(
    title="混凝土内部超声智能检测系统",
    description="基于低频超声与人工智能的混凝土内部目标智能识别系统",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_ROOT.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_ROOT)),
        name="static"
    )

inference_lock = asyncio.Lock()
TASKS = {}
TASKS_LOCK = asyncio.Lock()
BACKGROUND_JOBS = set()


async def _set_task(task_id, **updates):
    async with TASKS_LOCK:
        task = TASKS.setdefault(task_id, {})
        task.update(updates)
        task["updated_at"] = time.time()


async def _get_task(task_id):
    async with TASKS_LOCK:
        task = TASKS.get(task_id)
        return dict(task) if task is not None else None


async def _run_detection_task(
    task_id,
    upload_path,
    original_filename,
    specimen,
    rotation,
):
    started_at = time.time()

    await _set_task(
        task_id,
        status="processing",
        message="AI 推理任务已启动",
        started_at=started_at,
        elapsed_seconds=0,
    )

    try:
        async with inference_lock:
            await _set_task(
                task_id,
                message="正在执行超声信号处理与 AI 推理",
            )

            result = await asyncio.to_thread(
                run_internal_inference,
                input_npy=str(upload_path),
                specimen=specimen,
                rotation=rotation,
                output_root=str(OUTPUT_ROOT),
                force_rerun=True,
            )

        result["request_id"] = task_id
        result["task_id"] = task_id
        result["original_filename"] = original_filename
        result["visualization_url"] = (
            f"/api/internal-result-image/{task_id}"
        )

        await _set_task(
            task_id,
            status="success",
            message="检测完成",
            result=result,
            elapsed_seconds=round(
                time.time() - started_at,
                1
            ),
        )

    except Exception as exc:
        tb = traceback.format_exc()

        print(
            f"\n[TASK {task_id}] AI inference failed:\n{tb}",
            flush=True,
        )

        await _set_task(
            task_id,
            status="error",
            message="AI 推理失败",
            error=str(exc),
            traceback=tb,
            elapsed_seconds=round(
                time.time() - started_at,
                1
            ),
        )


def _track_background_job(task):
    BACKGROUND_JOBS.add(task)
    task.add_done_callback(
        BACKGROUND_JOBS.discard
    )


@app.get("/config")
async def config():
    return {
        "status": "ok",
        "message": "FastAPI app is running"
    }


@app.get("/info")
async def info():
    return {
        "named_endpoints": {},
        "unnamed_endpoints": {}
    }


@app.get("/", include_in_schema=False)
async def home():
    html_path = STATIC_ROOT / "index.html"

    if not html_path.exists():
        return {
            "status": "ok",
            "message": (
                "FastAPI 服务已启动。"
                "前端 static/index.html 未部署，"
                "可用 /docs 或 /api/health 测试接口。"
            ),
        }

    return FileResponse(str(html_path))


@app.get(
    "/api/health",
    summary="服务状态检查"
)
async def health():
    return {
        "status": "ok",
        "message": "混凝土内部超声智能检测服务运行正常",
        "active_tasks": sum(
            1
            for item in TASKS.values()
            if item.get("status") == "processing"
        ),
    }


@app.post(
    "/api/internal-detect",
    status_code=202,
    summary="提交混凝土内部超声智能检测任务",
    description=(
        "上传原始三维 .npy 超声扫描数据并立即返回 task_id，"
        "前端轮询任务状态。"
    ),
)
async def internal_detect(
    file: UploadFile = File(
        ...,
        description="三维超声扫描 .npy 文件"
    ),
    specimen: str = Form(
        "pk266",
        description="试件类型：pk266 或 pk050"
    ),
    rotation: str = Form(
        "rot90",
        description="扫描方向：rot00 或 rot90"
    ),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="未检测到上传文件"
        )

    if not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="当前仅支持 .npy 超声扫描文件"
        )

    specimen = specimen.lower().strip()
    rotation = rotation.lower().strip()

    if specimen not in {"pk266", "pk050"}:
        raise HTTPException(
            status_code=400,
            detail="试件类型必须为 pk266 或 pk050"
        )

    if rotation not in {"rot00", "rot90"}:
        raise HTTPException(
            status_code=400,
            detail="扫描方向必须为 rot00 或 rot90"
        )

    request_id = uuid.uuid4().hex[:12]

    specimen_name = (
        "Pk266"
        if specimen == "pk266"
        else "Pk050"
    )

    rotation_name = (
        "Rot90"
        if rotation == "rot90"
        else "Rot00"
    )

    saved_filename = (
        f"{specimen_name}_API_Shear_"
        f"{rotation_name}_{request_id}.npy"
    )

    upload_path = (
        UPLOAD_ROOT / saved_filename
    )

    try:
        with open(
            upload_path,
            "wb"
        ) as buffer:
            await asyncio.to_thread(
                shutil.copyfileobj,
                file.file,
                buffer,
            )

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"文件保存失败：{exc}"
        )

    # 注意：
    # request_id 已经作为第一个参数传给 _set_task
    # 这里不能再写 task_id=request_id
    await _set_task(
        request_id,
        request_id=request_id,
        status="accepted",
        message="文件上传完成，等待 AI 推理",
        original_filename=file.filename,
        specimen=specimen,
        rotation=rotation,
        created_at=time.time(),
        elapsed_seconds=0,
    )

    job = asyncio.create_task(
        _run_detection_task(
            request_id,
            upload_path,
            file.filename,
            specimen,
            rotation,
        )
    )

    _track_background_job(job)

    return {
        "status": "accepted",
        "task_id": request_id,
        "request_id": request_id,
        "message": "检测任务已创建",
        "original_filename": file.filename,
    }


@app.get(
    "/api/internal-task/{task_id}",
    summary="查询内部超声检测任务状态"
)
async def internal_task(task_id: str):

    if not re.fullmatch(
        r"[0-9a-f]{12}",
        task_id
    ):
        raise HTTPException(
            status_code=404,
            detail="非法的任务 ID"
        )

    task = await _get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="未找到对应检测任务"
        )

    started_at = (
        task.get("started_at")
        or task.get("created_at")
        or time.time()
    )

    elapsed = round(
        time.time() - started_at,
        1
    )

    if task.get("status") == "success":

        result = dict(
            task.get("result") or {}
        )

        result["status"] = "success"
        result["task_id"] = task_id
        result["request_id"] = task_id
        result["elapsed_seconds"] = (
            task.get(
                "elapsed_seconds",
                elapsed
            )
        )

        return result

    if task.get("status") == "error":

        return {
            "status": "error",
            "task_id": task_id,
            "request_id": task_id,
            "message": task.get(
                "message",
                "AI 推理失败"
            ),
            "error": task.get(
                "error",
                "未知错误"
            ),
            "elapsed_seconds": task.get(
                "elapsed_seconds",
                elapsed
            ),
        }

    return {
        "status": task.get(
            "status",
            "processing"
        ),
        "task_id": task_id,
        "request_id": task_id,
        "message": task.get(
            "message",
            "AI 正在分析"
        ),
        "elapsed_seconds": elapsed,
    }


@app.get(
    "/api/internal-result-image/{request_id}",
    summary="获取内部目标检测结果图"
)
async def internal_result_image(
    request_id: str
):

    if not re.fullmatch(
        r"[0-9a-f]{12}",
        request_id
    ):
        raise HTTPException(
            status_code=404,
            detail="非法的任务 ID"
        )

    matching_dirs = sorted(
        OUTPUT_ROOT.glob(
            f"*{request_id}*"
        ),
        key=lambda p: (
            p.stat().st_mtime
            if p.exists()
            else 0
        ),
        reverse=True,
    )

    if not matching_dirs:
        raise HTTPException(
            status_code=404,
            detail="未找到对应检测任务"
        )

    image_path = (
        matching_dirs[0]
        / "result"
        / "internal_targets.png"
    )

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="检测结果图尚未生成或不存在"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=(
            f"internal_targets_"
            f"{request_id}.png"
        ),
    )


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )