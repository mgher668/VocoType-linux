# SenseVoice 模型迁移规划

本文档规划将 VoCoType Linux 当前 FunASR Paraformer ONNX 识别后端，扩展为可选的 SenseVoice 后端。目标不是直接删除现有模型，而是新增一个可回退的 ASR 后端选项，让用户可以在稳定的 Paraformer 与更强的 SenseVoice 之间切换。

## 背景

当前项目的 ASR 栈：

- ASR：`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`
- VAD：`iic/speech_fsmn_vad_zh-cn-16k-common-onnx`
- 标点：`iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx`
- 推理库：`funasr_onnx==0.4.1`
- 加载方式：`app/funasr_server.py` 中通过 `funasr_onnx.paraformer_bin.Paraformer` 加载 ONNX Paraformer

SenseVoice 官方仓库：

- GitHub: https://github.com/FunAudioLLM/SenseVoice
- ModelScope: https://www.modelscope.cn/models/iic/SenseVoiceSmall
- Hugging Face: https://huggingface.co/FunAudioLLM/SenseVoiceSmall

SenseVoice 的价值：

- 支持 ASR、语种识别、情绪识别、音频事件检测。
- 支持普通话、粤语、英语、日语、韩语等多语言。
- 官方提供 `funasr.AutoModel` 和 `funasr_onnx.SenseVoiceSmall` 两条推理路线。
- 对输入法场景而言，优先考虑 ONNX 路线，避免引入过重的 PyTorch 依赖。

## 目标

- 新增 `sensevoice_onnx` ASR 后端。
- 保留当前 `paraformer_onnx` 作为默认后端和回退方案。
- 通过配置切换模型，而不是让用户改源码。
- 保持 Fcitx5 / IBus 上层输入逻辑不变。
- 支持 CPU 离线推理，优先保证短句低延迟。

## 非目标

- 本阶段不做 speaker diarization。
- 本阶段不接入 SenseVoice emotion / event 标签到 UI。
- 本阶段不改 Fcitx5 输入法形态，也不把 VoCoType 改成 Rime 插件。
- 本阶段不移除 Paraformer、FSMN VAD、CT Transformer 标点模型。

## 推荐方案

采用“可插拔 ASR 后端”设计：

```json
{
  "asr": {
    "backend": "sensevoice_onnx",
    "model": "iic/SenseVoiceSmall",
    "language": "auto",
    "use_itn": true,
    "batch_size": 1,
    "quantize": true
  }
}
```

默认配置仍保持：

```json
{
  "asr": {
    "backend": "paraformer_onnx"
  }
}
```

后端分支：

- `paraformer_onnx`：继续使用现有 `Paraformer + CT Transformer punc`。
- `sensevoice_onnx`：使用 `funasr_onnx.SenseVoiceSmall`，并用 `rich_transcription_postprocess` 清理输出。

SenseVoice 模式下建议默认关闭现有独立标点模型，优先使用 SenseVoice 的 `use_itn=true` 输出，避免双重标点。

## 可能修改的文件

- `app/funasr_config.py`
  - 增加 ASR backend 配置默认值。
  - 增加 SenseVoice 默认模型名。

- `app/funasr_server.py`
  - 拆分现有 `_load_asr_model()`，按 backend 分派。
  - 增加 `_load_sensevoice_onnx_model()`。
  - 增加 `_transcribe_sensevoice_onnx()`。
  - 在 `transcribe_audio()` 中根据 backend 选择推理路径。

- `app/download_models.py`
  - 支持 SenseVoiceSmall 的缓存路径检查。
  - 不再假设所有 ASR 模型都一定有 `model.onnx` / `model_quant.onnx`。

- `requirements.txt` / `pyproject.toml`
  - 确认 `funasr_onnx` 版本是否满足 SenseVoiceSmall。
  - 如当前 `0.4.1` 不满足，升级并验证安装脚本。

- `fcitx5/scripts/install-fcitx5.sh`
  - 安装时可询问是否启用 SenseVoice。
  - 写入 `~/.config/vocotype/fcitx5-backend.json` 的 `asr.backend`。
  - 保留 Paraformer 默认路径。

- `docs/FAQ.md` / `fcitx5/README.md` / `readme.md`
  - 增加 SenseVoice 配置说明、回滚说明、依赖说明。

- `test/`
  - 增加 ASR backend 配置解析测试。
  - 增加 SenseVoice 输出后处理测试。
  - 如果测试环境没有模型，使用 mock model 做单元测试。

## 实施步骤

1. 记录当前基线
   - 保存当前 Paraformer 模式的短句识别延迟、内存占用、典型错误样本。
   - 样本至少覆盖普通话、中英混合、命令式短句、噪声环境。

2. 加配置层
   - 在 `fcitx5-backend.json` 支持 `asr.backend`。
   - 支持值：`paraformer_onnx`、`sensevoice_onnx`。
   - 未配置时默认 `paraformer_onnx`，避免破坏现有用户。

3. 加 SenseVoice ONNX 加载器
   - 使用 `from funasr_onnx import SenseVoiceSmall`。
   - 使用 `from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess`。
   - 参数先收敛为：`model`、`language`、`use_itn`、`quantize`、`batch_size`。

4. 加 SenseVoice 推理路径
   - 将当前音频文件路径包装成单元素列表传给模型。
   - 输出统一成现有 `transcribe_audio()` 返回结构。
   - 保持字段：`success`、`text`、`raw_text`、`duration`、`language`、`model_type`、`models`。

5. 处理标点和文本归一化
   - SenseVoice 模式默认使用 `use_itn=true`。
   - SenseVoice 输出先走 `rich_transcription_postprocess`。
   - 再走现有 `normalize_text()`，保持项目现有中文数字和空白处理行为。
   - 默认不再调用 CT Transformer 标点，除非后续实测需要。

6. 更新模型下载逻辑
   - SenseVoiceSmall 不强制检查 `model.onnx`。
   - 首次加载可能触发 ONNX export，应在日志中明确提示。
   - 下载失败时提示回退到 `paraformer_onnx`。

7. 更新安装脚本
   - 增加可选问题：“是否启用 SenseVoice ASR 后端”。
   - 默认选否，保持当前稳定路径。
   - 如果用户选择启用，写入配置并下载或预热模型。

8. 增加验证命令
   - Paraformer 回归：确认默认模式仍可识别。
   - SenseVoice smoke test：用本地短音频跑一次转写。
   - Fcitx5 smoke test：后端服务重启后，F9 录音提交文本正常。

9. 做性能对比
   - 记录冷启动模型加载时间。
   - 记录 1 秒、3 秒、10 秒音频的转写延迟。
   - 记录常驻内存。
   - 对比识别准确率和标点质量。

10. 做回滚方案
    - 配置改回：

      ```json
      {
        "asr": {
          "backend": "paraformer_onnx"
        }
      }
      ```

    - 重启：

      ```bash
      systemctl --user restart vocotype-fcitx5-backend.service
      ```

## 风险

- `funasr_onnx==0.4.1` 是否完整支持 SenseVoiceSmall 需要实测确认。
- SenseVoice 首次加载可能需要导出 ONNX，冷启动可能比当前模型慢。
- `use_itn=true` 可能输出和当前标点模型不同的标点风格。
- SenseVoice 对短促口语、口头禅、命令式输入是否优于当前 Paraformer，需要用真实输入样本验证。
- 如果走 `funasr.AutoModel`，会引入更重依赖，不适合作为第一阶段方案。

## 验收标准

- 默认配置不变时，现有 Paraformer 行为不回退。
- 配置 `asr.backend=sensevoice_onnx` 后，后端可正常启动。
- F9 短句语音输入可提交文本。
- 识别失败时不会卡死输入法 UI。
- SenseVoice 模式下可以通过改配置一键回退 Paraformer。
- 文档包含安装、启用、回滚、排障说明。

## 建议顺序

第一阶段只做 `sensevoice_onnx` 可选后端，不动输入法交互形态。等模型质量和延迟验证通过后，再考虑是否把它设为默认模型。
