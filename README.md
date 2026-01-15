# Video Generation Related Work

基于 Wan2.1 的快速视频生成方法对比，包含 4 种不同的蒸馏/加速方案。

## 方法对比

| 方法 | 步数 | 来源 | 说明 |
|------|------|------|------|
| FastVideo | 8-step | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | CausalWan2.2 一致性蒸馏 |
| LightX2V CausVid | 9-step | [ModelTC/lightx2v](https://github.com/ModelTC/lightx2v) | Causal Video Distillation |
| rCM | 4-step | [NVlabs/rcm](https://github.com/NVlabs/rcm) | Reduced Consistency Model |
| Krea Self-Forcing | 4-step | [krea-ai/realtime-video](https://github.com/krea-ai/realtime-video) | Self-Forcing 训练策略 |

## 安装

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/Enderfga/video-gen-related.git
cd video-gen-related
```

如果已克隆但忘记 `--recursive`：
```bash
git submodule update --init --recursive
```

### 2. 创建环境

需要 3 个 conda 环境：

```bash
# FastVideo 和 Krea 共用环境
conda create -n fastvideo python=3.10
conda activate fastvideo
pip install -r FastVideo/requirements.txt

# rCM 需要 flash_attn（使用已有的 longlive 环境或新建）
conda create -n longlive python=3.10
conda activate longlive
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation

# LightX2V
conda create -n lightx2v python=3.10
conda activate lightx2v
pip install -r lightx2v/requirements.txt
```

### 3. 下载权重

```bash
# FastVideo CausalWan2.2 (需要约 30GB)
# 放到指定目录或修改 infer_fastvideo.py 中的路径

# LightX2V CausVid 权重
# 下载 causal_model.pt 到 CausVid/ 目录

# rCM 权重
# 下载 rCM_Wan2.1_T2V_14B_480p.pt 到当前目录

# Krea 使用 LongLive 的 wan_models
# 设置 MODEL_FOLDER 环境变量指向模型目录
```

## 使用

### 批量运行所有方法（4 GPU 并行）

```bash
python run_all.py
```

GPU 分配：
- GPU 0: FastVideo
- GPU 1: rCM
- GPU 2: Krea
- GPU 3: LightX2V

### 单独运行某个方法

```bash
python run_all.py --only fastvideo
python run_all.py --only rcm
python run_all.py --only krea
python run_all.py --only lightx2v
```

### 直接运行推理脚本

```bash
# FastVideo
CUDA_VISIBLE_DEVICES=0 python infer_fastvideo.py

# rCM (需要 longlive 环境)
conda activate longlive
CUDA_VISIBLE_DEVICES=1 python infer_rcm.py

# Krea
CUDA_VISIBLE_DEVICES=2 python infer_krea.py

# LightX2V
conda activate lightx2v
CUDA_VISIBLE_DEVICES=3 python infer_lightx2v.py
```

## 配置

所有方法使用统一的参数：

```python
NUM_FRAMES = 81        # 帧数
WIDTH = 832            # 宽度
HEIGHT = 480           # 高度
SEEDS = [0, 1]         # 随机种子
```

Prompt 列表在 `prompt.json` 中，包含 16 个测试 prompt。

## 输出结构

```
outputs/
├── fastvideo_8step/
│   ├── seed_0/
│   │   ├── video_000.mp4
│   │   ├── video_001.mp4
│   │   └── ...
│   └── seed_1/
├── rcm_4step/
│   ├── seed_0/
│   └── seed_1/
├── krea_4step/
│   ├── seed_0/
│   └── seed_1/
└── lightx2v_9step/
    ├── seed_0/
    └── seed_1/
```

每个方法生成 32 个视频（16 prompts × 2 seeds）。

## License

各子项目遵循其原始 License。
