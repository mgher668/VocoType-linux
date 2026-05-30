# SenseVoice 优先适配规划

本文档规划一条更保守的迁移路线：先把 SenseVoice ONNX 作为可选 ASR 后端接入当前 VoCoType 后端，暂时不改 Fcitx5 输入法形态。这样可以先验证新模型的识别质量、标点效果、启动耗时和运行内存，再决定是否继续推进“Pinyin 为主、VoCoType 插件化”的交互改造。

## 决策结论

- 先适配模型，不先改输入法模式。
- 使用预导出的 `iic/SenseVoiceSmall-onnx` 路线；项目内增加兼容 `tokens.json` 的轻量 ONNX 包装器。
- SenseVoice 作为可选后端，不替换默认 Paraformer。
- SenseVoice 模式先使用原生 `use_itn=true` 标点/文本规整，不叠加当前 CT Transformer 标点模型。
- 先覆盖 Fcitx5 当前使用链路；后端设计尽量保持通用，避免未来 IBus 复用困难。
- 安装时支持自动下载或预热 SenseVoice 模型。
- 语言策略默认 `auto`，适配中英混输。

## 为什么先做模型适配

当前存在两个独立问题：

1. 识别效果问题：当前 Paraformer 对部分真实语音输入不够理想。
2. 输入法交互问题：当前 VoCoType 作为主输入法不如 Pinyin/Rime 主输入法自然。

如果同时改模型和输入法形态，问题定位会变复杂。先把 SenseVoice 接入当前稳定链路，可以单独回答一个问题：SenseVoice 是否真的改善当前使用场景的识别质量和延迟。

## 当前基线

当前默认模型配置位于 `app/funasr_config.py`：

- ASR：`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`
- VAD：`iic/speech_fsmn_vad_zh-cn-16k-common-onnx`
- PUNC：`iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx`
- Revision：`v2.0.5`

当前加载路径位于 `app/funasr_server.py`：

- `_load_asr_model()` 只支持 ONNX 模型。
- ONNX ASR 当前通过 `funasr_onnx.paraformer_bin.Paraformer` 加载。
- `transcribe_audio()` 当前统一走 Paraformer 推理，再可选走 VAD/PUNC/文本归一化。

## 目标

- 新增 `sensevoice_onnx` ASR 后端。
- 默认仍为 `paraformer_onnx`，避免破坏已有用户。
- 配置可切换：

  ```json
  {
    "asr": {
      "backend": "sensevoice_onnx",
      "model": "iic/SenseVoiceSmall-onnx",
      "language": "auto",
      "use_itn": true,
      "quantize": false,
      "batch_size": 1
    }
  }
  ```

- SenseVoice 输出统一转换为现有 `transcribe_audio()` 返回结构。
- 保留回滚能力，配置改回 `paraformer_onnx` 后即可恢复原模型。

## 非目标

- 不改 Fcitx5 当前输入法形态。
- 不新增 F9 全局插件模式。
- 不删除当前 Paraformer / VAD / PUNC 模型。
- 不做 speaker diarization。
- 不把 SenseVoice 的情绪识别、音频事件检测暴露到输入法 UI。
- 不引入 PyTorch `funasr.AutoModel` 现场导出作为第一阶段方案。实测 `iic/SenseVoiceSmall` 现场导出的 ONNX 在当前 PyTorch/ONNXRuntime 组合下会加载失败，因此默认只走预导出 ONNX。

## 推荐架构

新增一个轻量 ASR 后端分派层：

```text
FunASRServer
  ├─ paraformer_onnx
  │    ├─ Paraformer
  │    ├─ optional FSMN VAD
  │    └─ optional CT Transformer PUNC
  │
  └─ sensevoice_onnx
       ├─ SenseVoiceOnnx
       ├─ language=auto
       ├─ use_itn=true
       └─ rich_transcription_postprocess
```

SenseVoice 模式下，第一版不叠加当前 PUNC 模型：

- 原因：SenseVoice 的 `use_itn=true` 已包含标点和逆文本规整能力。
- 好处：避免双重标点导致输出风格不稳定。
- 保留后续开关：如果实测原生标点不理想，再加 `asr.sensevoice.use_external_punc=true`。

## 可能修改的文件

- `app/funasr_config.py`
  - 增加默认 backend。
  - 增加 SenseVoice 默认模型配置。
  - 保留环境变量覆盖能力。

- `app/funasr_server.py`
  - 增加 ASR backend 初始化分派。
  - 新增 `_load_paraformer_onnx_model()` 或保留当前逻辑并重命名。
  - 新增 `_load_sensevoice_onnx_model()`。
  - 新增 `_transcribe_sensevoice_onnx()`。
  - 保证返回结构和当前调用方兼容。

- `app/sensevoice_onnx.py`
  - 新增 SenseVoice ONNX 包装器。
  - 兼容 `iic/SenseVoiceSmall-onnx` 只提供 `tokens.json`、不提供 bpe model 的模型结构。

- `app/download_models.py`
  - 支持 SenseVoiceSmall 缓存检查。
  - 不再强制以 `model.onnx` / `model_quant.onnx` 判断所有 ASR 模型完整性。
  - 安装时可触发 SenseVoice 模型下载。

- `requirements.txt` / `pyproject.toml`
  - 保持 `funasr_onnx` 版本，避免扩大改动。
  - SenseVoice 路线需要 CPU 版 `torch`，因为 `funasr_onnx==0.4.1` 包初始化会导入 SenseVoice 模块。

- `fcitx5/scripts/install-fcitx5.sh`
  - 增加安装问题：“是否启用 SenseVoice ASR 后端”。
  - 选择启用后写入 `~/.config/vocotype/fcitx5-backend.json`。
  - 选择启用后自动下载或预热 SenseVoice 模型。

- `docs/FAQ.md` / `fcitx5/README.md`
  - 增加 SenseVoice 启用、回滚、排障说明。

- `test/`
  - 增加配置解析测试。
  - 增加 SenseVoice 后处理 mock 测试。
  - 增加后端分派单元测试。

## 实施阶段

### 阶段 0：基线记录

先记录当前 Paraformer 表现，避免凭感觉判断：

- 准备 10 到 20 条真实录音样本。
- 覆盖普通话、中英混输、短句、长句、口头禅、噪声。
- 记录：
  - 首次模型加载时间
  - 单次转写耗时
  - 常驻内存
  - 原始识别文本
  - 最终提交文本

### 阶段 1：配置层

新增配置字段：

```json
{
  "asr": {
    "backend": "paraformer_onnx"
  }
}
```

支持：

- `paraformer_onnx`
- `sensevoice_onnx`

未配置时默认 `paraformer_onnx`。

### 阶段 2：SenseVoice ONNX 加载

实现：

- 导入项目内 `SenseVoiceOnnx`。
- 加载 `iic/SenseVoiceSmall-onnx`。
- 参数支持：
  - `language`
  - `use_itn`
  - `quantize`（默认 false；官方预导出仓库只有 `model_quant.onnx` 时仍会自动使用该文件）
  - `batch_size`

第一版推荐默认：

```json
{
  "language": "auto",
  "use_itn": true,
  "quantize": false,
  "batch_size": 1
}
```

### 阶段 3：SenseVoice 推理路径

实现统一返回：

```json
{
  "success": true,
  "text": "...",
  "raw_text": "...",
  "duration": 1.23,
  "language": "auto",
  "model_type": "sensevoice_onnx",
  "models": {
    "asr": "iic/SenseVoiceSmall-onnx"
  }
}
```

输出处理顺序：

```text
SenseVoice ONNX raw output
  -> rich_transcription_postprocess
  -> normalize_text
  -> final text
```

### 阶段 4：安装脚本集成

安装时提供选项：

```text
是否启用 SenseVoice ASR 后端？
[1] 否，继续使用 Paraformer（默认）
[2] 是，启用 SenseVoiceSmall ONNX
```

如果选择 SenseVoice：

- 写入配置。
- 自动下载或预热模型。
- 下载失败时不终止安装，可回退 Paraformer。

### 阶段 5：对比验证

同一批样本分别跑：

- `paraformer_onnx`
- `sensevoice_onnx`

记录：

- 错字率主观对比
- 标点质量
- 中英混输表现
- 短句延迟
- 冷启动耗时
- 常驻内存

### 阶段 6：用户试用

在当前 Fcitx5 VoCoType 输入法模式下试用 SenseVoice：

- 不改输入法模式。
- 不改 F9 交互。
- 只验证 ASR 输出质量。

如果效果明显更好，再进入下一阶段：Fcitx5 插件模式改造。

## 回滚方案

配置改回：

```json
{
  "asr": {
    "backend": "paraformer_onnx"
  }
}
```

然后重启后端：

```bash
systemctl --user restart vocotype-fcitx5-backend.service
```

## 风险

- 当前 `funasr_onnx==0.4.1` 的原始 `SenseVoiceSmall` 类不完整兼容 `iic/SenseVoiceSmall-onnx`，需要项目内包装器适配 `tokens.json`。
- 不建议安装时从 `iic/SenseVoiceSmall` 现场导出 ONNX；实测导出模型在当前环境下会出现 ONNXRuntime 类型错误。
- SenseVoice 原生标点可能和当前 CT Transformer 风格不同。
- `language=auto` 对纯中文短句是否稳定，需要样本验证。
- 安装时自动下载模型会增加安装耗时和失败点。

## 验收标准

- 默认配置下 Paraformer 行为不变。
- 配置 SenseVoice 后，后端可正常启动。
- Fcitx5 当前 VoCoType 输入法模式下，F9 录音可正常提交文本。
- SenseVoice 输出可完成标点和文本归一化。
- 失败时可清晰提示并回退 Paraformer。
- 文档包含启用、验证、回滚、排障说明。

## 后续决策点

如果 SenseVoice 验证通过，再继续讨论：

- 是否将 SenseVoice 设为默认后端。
- 是否推进 Fcitx5 插件模式。
- 是否隐藏旧 VoCoType 输入法入口。
- 是否移除内置 Rime 拼音转发逻辑。
