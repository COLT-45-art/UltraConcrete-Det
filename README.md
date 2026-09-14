# UltraConcrete-Det

AI-based ultrasonic internal target detection and localization system for concrete structures.

基于低频超声数据与深度学习的混凝土内部目标自动检测与定位系统。

---

## 项目简介

UltraConcrete-Det 是一个面向混凝土内部超声检测的 AI 推理与可视化系统。

系统以三维超声 `.npy` 数据作为输入，通过超声信号处理、特征构建、深度学习检测与辅助模型融合，实现混凝土内部目标的自动识别、定位和结果可视化。

当前系统已经完成 Web 化部署，可自动输出：

- 内部目标编号
- 水平位置 X / mm
- 埋深 / mm
- 检测置信度
- 内部目标位置可视化结果

项目已成功部署至 ModelScope 创空间。

---

## 主要功能

- 支持原始超声 `.npy` 数据输入
- 自动完成超声信号预处理
- 自动构建检测所需特征
- 基于 YOLO 模型进行内部目标检测
- 结合浅层机器学习模型辅助推理
- 自动估计目标水平位置
- 自动估计目标埋深
- 自动输出检测置信度
- 自动生成检测结果可视化
- 提供 FastAPI Web 服务
- 支持 Docker 部署
- 支持 CPU 推理环境

---

## 技术栈

本项目主要使用以下技术：

- Python
- PyTorch
- Ultralytics YOLO
- Scikit-learn
- NumPy
- SciPy
- OpenCV
- Pillow
- Matplotlib
- FastAPI
- Uvicorn
- Docker

---

## 项目结构

```text
UltraConcrete-Det/
│
├── stage9_03_complete_inference.py
├── stage9_04_raw_npy_complete_inference.py
├── stage10_01_fastapi.py
├── internal_inference_service.py
├── runtime_config.py
│
├── static/
│
├── requirements.txt
├── Dockerfile
├── docker.yaml
├── .gitignore
├── .gitattributes
└── README.md
```

主要文件说明：

- `stage9_03_complete_inference.py`  
  完整推理流程

- `stage9_04_raw_npy_complete_inference.py`  
  原始 `.npy` 超声数据推理流程

- `stage10_01_fastapi.py`  
  FastAPI Web 服务入口

- `internal_inference_service.py`  
  内部目标检测服务逻辑

- `runtime_config.py`  
  推理设备、路径和运行环境配置

- `static/`  
  Web 前端静态资源

- `Dockerfile`  
  Docker 部署配置

- `requirements.txt`  
  Python 项目依赖

---

## 安装环境

建议使用：

```text
Python 3.10
```

首先克隆项目：

```bash
git clone https://github.com/COLT-45-art/UltraConcrete-Det.git
cd UltraConcrete-Det
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

---

## PyTorch 安装

PyTorch 建议根据自己的 CPU / CUDA 环境单独安装。

CPU 环境可参考：

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

如果使用 NVIDIA GPU，请根据本机 CUDA 版本安装对应版本的 PyTorch。

---

## 启动 Web 服务

运行：

```bash
uvicorn stage10_01_fastapi:app --host 0.0.0.0 --port 7860
```

启动成功后，在浏览器访问：

```text
http://localhost:7860
```

即可进入超声内部目标检测系统。

---

## 输入数据

当前系统主要支持三维超声 `.npy` 数据。

示例：

```text
Pk266_3D_Dataset_Shear_Rot90.npy
```

系统读取原始超声数据后，会自动执行后续处理与推理流程。

---

## 推理流程

整体推理流程如下：

```text
原始超声 .npy 数据
        ↓
超声信号预处理
        ↓
特征构建
        ↓
深度学习模型检测
        ↓
辅助模型融合
        ↓
内部目标筛选
        ↓
水平位置与埋深换算
        ↓
置信度计算
        ↓
检测结果可视化
```

---

## 检测结果

对于每一个检测到的内部目标，系统输出：

```text
Target ID
Horizontal Position X / mm
Depth / mm
Confidence
```

示例检测结果：

| 目标编号 | 水平位置 X / mm | 埋深 / mm | 置信度 |
|---|---:|---:|---:|
| T1 | 289.89 | 246.58 | 83.5% |
| T2 | 755.85 | 176.13 | 85.7% |
| T3 | 1256.03 | 124.45 | 67.8% |
| T4 | 1734.65 | 69.33 | 41.9% |

---

## 模型

当前推理流程主要使用：

```text
fusion_fold_A_best.pt
fusion_fold_B_best.pt
shallow_rf_rot00.joblib
```

其中包括：

- YOLO Fusion Model A
- YOLO Fusion Model B
- Shallow Random Forest Model

当前预训练权重暂不直接包含在 GitHub 仓库中。

模型权重和在线部署版本可托管于 ModelScope。

---

## 在线部署

本项目已成功部署至 ModelScope 创空间。

部署入口：

```text
stage10_01_fastapi.py
```

启动方式：

```bash
uvicorn stage10_01_fastapi:app --host 0.0.0.0 --port 7860
```

Docker 部署文件已包含在项目中。

---

## Docker 部署

构建 Docker 镜像：

```bash
docker build -t ultraconcrete-det .
```

运行：

```bash
docker run -p 7860:7860 ultraconcrete-det
```

随后访问：

```text
http://localhost:7860
```

---

## 运行环境配置

项目支持通过环境变量控制推理设备。

默认：

```text
INFERENCE_DEVICE=auto
```

系统会自动判断是否存在 CUDA 环境。

也可以手动指定：

```text
INFERENCE_DEVICE=cpu
```

或：

```text
INFERENCE_DEVICE=cuda
```

运行时输出目录可通过：

```text
ULTRASONIC_OUTPUT_ROOT
```

进行配置。

若未指定，则默认写入：

```text
/tmp/ultrasonic_outputs
```

---

## 应用方向

本项目主要面向：

- 混凝土内部无损检测
- 超声结构检测
- 内部目标自动识别
- 工程结构智能检测
- AI 辅助无损检测
- 超声数据智能分析
- 混凝土内部结构定位

---

## 研究目标

本项目尝试将传统超声信号处理与人工智能检测模型结合，用于混凝土内部目标的自动识别和定位。

相比单纯依赖人工查看超声信号，本系统希望进一步提高：

- 检测效率
- 自动化程度
- 目标定位能力
- 结果可视化能力
- 工程部署能力

---

## Future Work

后续计划包括：

- 提高低置信度目标的检测稳定性
- 优化复杂噪声环境下的鲁棒性
- 扩展更多超声数据类型
- 增加更多混凝土内部缺陷类别
- 优化目标深度估计精度
- 优化多模型融合策略
- 增加批量文件自动检测能力
- 完善工程化 Web 界面
- 探索不同超声设备和不同材料条件下的迁移能力

---

## 注意事项

本项目目前主要用于科研、学习与工程原型验证。

检测结果不应直接替代正式工程检测规范、专业检测人员判断以及实际工程安全评估。

---

## License

This project is licensed under the MIT License.

---

## Author

**Yuuki18**

Project:

```text
UltraConcrete-Det
```

Ultrasonic AI Detection for Concrete Internal Structures.

---

## Contact

For questions, bug reports, or suggestions regarding this project, please feel free to contact me.

如在使用本项目过程中遇到任何问题、Bug，或有改进建议，欢迎通过邮箱联系。

**Email:** 2975106762@qq.com

You can also open an Issue on GitHub.
