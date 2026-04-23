<p align="center">
  <img src="./logo.png" alt="MAGIC-TTS logo" width="220">
</p>

# MAGIC-TTS: Fine-Grained Controllable Speech Synthesis with Explicit Local Duration and Pause Control

中文 | [English](#english)

## 中文

- 在线 Demo: https://yongaifadian1.github.io/MAGIC-TTS/

MAGIC-TTS 是一个支持细粒度局部时序控制的语音合成系统。它既可以对指定 token 的内容时长和停顿进行毫秒级控制，也可以在不提供任何显式时长的情况下自然生成语音。

这个仓库支持两种合成模式：

- `controlled`：显式提供局部控制信号，可以直接写在命令行的 `target_text` 里，也可以整理成完整时序轨文件传入
- `spontaneous`：不提供 target-side text duration，由模型自发建模内部 duration 与停顿时长

### 仓库内容

- `f5_tts/`：最小可运行的本地 Python 模块
- `inference/`：推理入口脚本
- `vendor/f5tts_duration_ft/`：release 所需的本地化时序控制实现
- `assets/default_prompt/`：默认音色 prompt 资源
- `outputs/`：公开 demo 资产；其他本地产物默认忽略
- `checkpoints/README.md`：checkpoint 放置约定

### 环境准备

```bash
conda create -n magictts python=3.10 -y
conda activate magictts
bash scripts/setup.sh
```

`setup.sh` 默认会：

- 安装 `torch==2.3.0` 和 `torchaudio==2.3.0`
- 安装本仓库 Python 依赖
- 下载 tokenizer / vocoder 到 `pretrained/`

此时仍需要你自己准备：

- MAGIC-TTS checkpoint，建议放到 `checkpoints/`，例如 `checkpoints/magictts_36k.pt`

如果当前机器不是 CUDA 11.8 环境，可覆盖默认 PyTorch wheel 源：

```bash
export MAGICTTS_TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu121
bash scripts/setup.sh
```

如果预训练资源不想放在默认的 `pretrained/`，可以设置：

```bash
export MAGICTTS_PRETRAINED_ROOT=/path/to/pretrained_assets
```

如果受控合成需要和当前环境不同的 Python，可设置：

```bash
export MAGICTTS_SYNTH_PYTHON=/path/to/python
```

如果要使用精细控制，再额外安装：

```bash
conda install -c conda-forge ffmpeg montreal-forced-aligner -y
```

如果 MFA 环境和模型资源不放在仓库内默认位置，还可以显式指定：

```bash
export MAGICTTS_MFA_ENV=/path/to/mfa_env
export MAGICTTS_MFA_ROOT=/path/to/mfa_root
```

如果当前机器无法直接访问 Hugging Face，也可以让 `setup.sh` 从一份本地 F5 资源副本拷贝：

```bash
MAGICTTS_SETUP_LOCAL_F5R_ROOT=/path/to/F5R-TTS-master bash scripts/setup.sh
```

如需顺带准备 fallback 用的 `model_1200000.safetensors`，可额外设置：

```bash
MAGICTTS_DOWNLOAD_BASE_CKPT=1 bash scripts/setup.sh
```

### 安装为本地包

如果你已经自己准备好了 Torch 环境，也可以直接：

```bash
python -m pip install -e .
```

### 默认音色

当前默认音色已经内置在仓库中：

- Prompt 音频：[prompt.wav](./assets/default_prompt/prompt.wav)
- Prompt 文本：[prompt.txt](./assets/default_prompt/prompt.txt)
- Prompt 时序轨：[prompt_track.json](./assets/default_prompt/prompt_track.json)
- Checkpoint 目录说明见：[checkpoints/README.md](./checkpoints/README.md)

### 直接试听

- [四场景控制 demo](./outputs/controlled_demos/default_voice)
- [四场景 spontaneous demo](./outputs/spontaneous_demos)
- [完整时序轨样例](./outputs/control_track_example)

### 复现四个 demo 场景

```bash
python inference/run_paper_demos.py \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/controlled_demos/default_voice
```

### 复现四个 spontaneous 场景

```bash
python inference/run_spontaneous_suite.py \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/spontaneous_demos
```

### 用自己的 prompt 和文本

如果 `target_text` 不带任何控制标记，模型会自动进入 spontaneous 模式。`--prompt-audio` 和 `--prompt-text` 都是可选的；如果省略，就会自动回退到仓库内置默认音色。

```bash
python inference/run_magictts.py \
  --target-text "前方路口，左转。" \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/my_spontaneous_demo
```

如果 `target_text` 带控制标记，模型会自动进入 controlled 模式。这里同样可以不提供 `--prompt-audio` 和 `--prompt-text`。

```bash
python inference/run_magictts.py \
  --target-text "前方路口[260]左{300}转{300}。" \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/my_controlled_demo
```

命令行控制语法直接写在 `target_text` 里：

- `字{300}` 表示该字符目标时长为 `300 ms`
- `[260]` 表示在当前位置插入一个 `260 ms` 停顿
- 未标注的内容字默认使用 `170 ms`

### 用完整时序轨文件合成

仓库里提供了一份可直接参考的完整时序轨：

- [custom_track.json](./outputs/control_track_example/custom_track.json)

然后直接合成：

```bash
python inference/run_edit_from_json.py \
  --track-json /path/to/full_track.json \
  --checkpoint /path/to/your_checkpoint.pt \
  --output-dir outputs/release_manual_demo
```

### 单独准备 prompt-side duration

```bash
python inference/align_prompt_with_mfa.py \
  --prompt-audio /path/to/prompt.wav \
  --prompt-text "前方路口" \
  --language zh \
  --output-dir outputs/prompt_alignment
```

运行结束后，`outputs/prompt_alignment` 目录里会得到：

- `prompt_alignment_raw.json`
- `prompt_alignment_debug.json`
- `prompt_track.json`

### 致谢

MAGIC-TTS 使用了 F5-TTS 提供的 backbone 实现，并以其公开 checkpoint 作为训练初始化起点。

---

## English

- Online demo: https://yongaifadian1.github.io/MAGIC-TTS/

MAGIC-TTS is a speech synthesis system with explicit fine-grained local timing control. It supports millisecond-level control over selected token durations and pauses, and it can also generate speech naturally without any explicit duration input.

This repository supports two synthesis modes:

- `controlled`: explicit local control signals are provided either inline in `target_text` or through a full timing-track file
- `spontaneous`: no target-side duration is provided and the model predicts internal duration and pause timing on its own

### Repository Layout

- `f5_tts/`: minimal local Python module used by inference
- `inference/`: inference entry scripts
- `vendor/f5tts_duration_ft/`: vendored timing-control implementation required by the release layer
- `assets/default_prompt/`: built-in default prompt assets
- `outputs/`: public demo artifacts; other local outputs are ignored by default
- `checkpoints/README.md`: checkpoint placement note

### Environment Setup

```bash
conda create -n magictts python=3.10 -y
conda activate magictts
bash scripts/setup.sh
```

By default, `setup.sh` will:

- install `torch==2.3.0` and `torchaudio==2.3.0`
- install this repository's Python dependencies
- download tokenizer / vocoder assets into `pretrained/`

You still need to provide:

- a MAGIC-TTS checkpoint, preferably under `checkpoints/`, for example `checkpoints/magictts_36k.pt`

If your machine is not on CUDA 11.8, override the default PyTorch wheel index:

```bash
export MAGICTTS_TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu121
bash scripts/setup.sh
```

If you want pretrained assets in a non-default location, set:

```bash
export MAGICTTS_PRETRAINED_ROOT=/path/to/pretrained_assets
```

If controlled synthesis should use a different Python interpreter, set:

```bash
export MAGICTTS_SYNTH_PYTHON=/path/to/python
```

For fine-grained control, also install:

```bash
conda install -c conda-forge ffmpeg montreal-forced-aligner -y
```

If the MFA environment or its downloaded models live outside the repository defaults, set:

```bash
export MAGICTTS_MFA_ENV=/path/to/mfa_env
export MAGICTTS_MFA_ROOT=/path/to/mfa_root
```

If the machine cannot access Hugging Face directly, `setup.sh` can copy tokenizer / vocoder assets from a local F5 resource tree:

```bash
MAGICTTS_SETUP_LOCAL_F5R_ROOT=/path/to/F5R-TTS-master bash scripts/setup.sh
```

To also prepare the fallback `model_1200000.safetensors`, set:

```bash
MAGICTTS_DOWNLOAD_BASE_CKPT=1 bash scripts/setup.sh
```

### Install As A Local Package

If you already manage your own Torch environment, you can directly run:

```bash
python -m pip install -e .
```

### Built-In Default Prompt

- Prompt audio: [prompt.wav](./assets/default_prompt/prompt.wav)
- Prompt transcript: [prompt.txt](./assets/default_prompt/prompt.txt)
- Prompt timing track: [prompt_track.json](./assets/default_prompt/prompt_track.json)
- Checkpoint directory note: [checkpoints/README.md](./checkpoints/README.md)

### Listen To The Demos

- [4-scene controlled demo](./outputs/controlled_demos/default_voice)
- [4-scene spontaneous demo](./outputs/spontaneous_demos)
- [full timing-track example](./outputs/control_track_example)

### Reproduce The 4 Controlled Demo Scenes

```bash
python inference/run_paper_demos.py \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/controlled_demos/default_voice
```

### Reproduce The 4 Spontaneous Demo Scenes

```bash
python inference/run_spontaneous_suite.py \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/spontaneous_demos
```

### Use Your Own Prompt And Text

If `target_text` does not contain any control marker, the model automatically runs in spontaneous mode. `--prompt-audio` and `--prompt-text` are both optional; if omitted, MAGIC-TTS falls back to the built-in default voice.

```bash
python inference/run_magictts.py \
  --target-text "前方路口，左转。" \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/my_spontaneous_demo
```

If `target_text` contains control markers, the model automatically runs in controlled mode. `--prompt-audio` and `--prompt-text` are also optional here.

```bash
python inference/run_magictts.py \
  --target-text "前方路口[260]左{300}转{300}。" \
  --checkpoint /path/to/magictts_36k.pt \
  --output-dir outputs/my_controlled_demo
```

Inline control markers are written directly inside `target_text`:

- `char{300}` sets that character to `300 ms`
- `[260]` inserts a `260 ms` pause at that position
- unmarked content characters use the default `170 ms`

### Synthesize From A Full Timing Track

You can reference the example timing track included in the repository:

- [custom_track.json](./outputs/control_track_example/custom_track.json)

Then synthesize directly:

```bash
python inference/run_edit_from_json.py \
  --track-json /path/to/full_track.json \
  --checkpoint /path/to/your_checkpoint.pt \
  --output-dir outputs/release_manual_demo
```

### Prepare Prompt-Side Duration Only

```bash
python inference/align_prompt_with_mfa.py \
  --prompt-audio /path/to/prompt.wav \
  --prompt-text "前方路口" \
  --language zh \
  --output-dir outputs/prompt_alignment
```

After the script finishes, `outputs/prompt_alignment` will contain:

- `prompt_alignment_raw.json`
- `prompt_alignment_debug.json`
- `prompt_track.json`

### Acknowledgement

MAGIC-TTS builds on the backbone implementation provided by F5-TTS and uses its public checkpoint as the initialization starting point for training.
