/*
 * VoCoType Fcitx5 Addon Implementation
 */

#include "vocotype.h"
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/text.h>
#include <fcitx/candidatelist.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/utf8.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <thread>
#include <chrono>

namespace {

std::string stopRecorderProcess(pid_t pid, int stdin_fd, FILE* stdout_file) {
    if (stdin_fd >= 0) {
        close(stdin_fd);
    }

    std::string audio_path;
    if (stdout_file) {
        char buffer[1024];
        if (fgets(buffer, sizeof(buffer), stdout_file) != nullptr) {
            audio_path = buffer;
            while (!audio_path.empty() &&
                   (audio_path.back() == '\n' || audio_path.back() == '\r')) {
                audio_path.pop_back();
            }
        }
        fclose(stdout_file);
    }

    if (pid > 0) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
    }

    return audio_path;
}

std::string clipProbeText(const std::string& text, size_t limit = 48) {
    std::string cleaned;
    cleaned.reserve(text.size());
    for (char ch : text) {
        if (ch == '\n') {
            cleaned.append("⏎");
        } else if (ch == '\t') {
            cleaned.append("⇥");
        } else {
            cleaned.push_back(ch);
        }
    }
    if (cleaned.size() <= limit) {
        return cleaned;
    }
    return cleaned.substr(0, limit) + "...";
}

std::pair<std::string, std::string> extractSentenceWindow(
    const std::string& text, int cursor_pos
) {
    if (text.empty()) {
        return {"", ""};
    }
    std::vector<std::pair<int, int>> spans;
    int start = 0;
    for (int i = 0; i < static_cast<int>(text.size()); ++i) {
        const char ch = text[static_cast<size_t>(i)];
        if (ch == '!' || ch == '?' || ch == ';' || ch == '\n' || ch == '.') {
            int end = i + 1;
            if (end > start) {
                spans.emplace_back(start, end);
            }
            start = end;
        }
    }
    if (start < static_cast<int>(text.size())) {
        spans.emplace_back(start, static_cast<int>(text.size()));
    }
    if (spans.empty()) {
        return {text, ""};
    }
    int cursor = std::max(0, std::min(cursor_pos, static_cast<int>(text.size())));
    int current_idx = static_cast<int>(spans.size()) - 1;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i].first <= cursor && cursor <= spans[i].second) {
            current_idx = static_cast<int>(i);
            break;
        }
    }
    const auto [cur_start, cur_end] = spans[static_cast<size_t>(current_idx)];
    std::string current = text.substr(static_cast<size_t>(cur_start), static_cast<size_t>(cur_end - cur_start));
    std::string previous;
    if (current_idx > 0) {
        const auto [prev_start, prev_end] = spans[static_cast<size_t>(current_idx - 1)];
        previous = text.substr(
            static_cast<size_t>(prev_start),
            static_cast<size_t>(prev_end - prev_start)
        );
    }
    return {current, previous};
}

fcitx::KeyStates keyStateFromMask(int state_mask) {
    fcitx::KeyStates states;
    if (state_mask & (1 << 0)) {
        states |= fcitx::KeyState::Shift;
    }
    if (state_mask & (1 << 2)) {
        states |= fcitx::KeyState::Ctrl;
    }
    if (state_mask & (1 << 3)) {
        states |= fcitx::KeyState::Alt;
    }
    return states;
}

} // namespace

namespace vocotype {

// F9 键（Shift+F9 使用同一个 keyval，通过 modifier 区分）
constexpr int PTT_KEYVAL = FcitxKey_F9;

VoCoTypeAddon::VoCoTypeAddon(fcitx::Instance* instance)
    : instance_(instance),
      ipc_client_(std::make_unique<IPCClient>("/tmp/vocotype-fcitx5.sock")) {

    // 获取安装路径
    const char* home = std::getenv("HOME");
    if (home) {
        python_venv_path_ = std::string(home) + "/.local/share/vocotype-fcitx5/.venv/bin/python";
        recorder_script_path_ = std::string(home) + "/.local/share/vocotype-fcitx5/backend/audio_recorder.py";
    } else {
        FCITX_ERROR() << "HOME environment variable not set";
    }

    FCITX_INFO() << "VoCoType Addon initialized";

    // 测试 Backend 连接
    if (ipc_client_->ping()) {
        FCITX_INFO() << "Backend connection OK";
    } else {
        FCITX_WARN() << "Backend not responding, please ensure fcitx5_server.py is running";
    }
}

VoCoTypeAddon::~VoCoTypeAddon() {
    if (recorder_pid_ > 0 || recorder_stdout_ || recorder_stdin_fd_ >= 0) {
        std::string audio_path =
            stopRecorderProcess(recorder_pid_, recorder_stdin_fd_, recorder_stdout_);
        if (!audio_path.empty()) {
            std::remove(audio_path.c_str());
        }
        recorder_pid_ = -1;
        recorder_stdin_fd_ = -1;
        recorder_stdout_ = nullptr;
        is_recording_ = false;
    }
    FCITX_INFO() << "VoCoType Addon destroyed";
}

std::vector<fcitx::InputMethodEntry> VoCoTypeAddon::listInputMethods() {
    std::vector<fcitx::InputMethodEntry> result;

    auto entry = fcitx::InputMethodEntry("vocotype", "VoCoType", "zh_CN", "vocotype");
    entry.setNativeName("语音输入");
    entry.setIcon("microphone");
    entry.setLabel("🎤");

    result.push_back(std::move(entry));
    return result;
}

void VoCoTypeAddon::keyEvent(const fcitx::InputMethodEntry& entry,
                              fcitx::KeyEvent& keyEvent) {
    auto ic = keyEvent.inputContext();

    // 获取按键信息
    auto key = keyEvent.key();
    int keyval = key.sym();
    bool is_release = keyEvent.isRelease();

    FCITX_DEBUG() << "Key event: keyval=" << keyval
                  << ", release=" << is_release
                  << ", F9=" << PTT_KEYVAL;

    // 处理 F9 键（PTT）
    if (keyval == PTT_KEYVAL) {
        const bool ctrl_mode = (key.states() & fcitx::KeyState::Ctrl);
        const bool shift_mode = (key.states() & fcitx::KeyState::Shift);
        if (is_release) {
            // F9 松开：停止录音并转录
            if (is_recording_) {
                stopAndTranscribe(ic);
            }
            keyEvent.filterAndAccept();
            return;
        }

        // Ctrl+Shift+F9: surrounding 探针
        if (ctrl_mode && shift_mode) {
            outputSurroundingProbe(ic);
            keyEvent.filterAndAccept();
            return;
        }

        // Ctrl+F9: 编辑模式
        if (ctrl_mode) {
            SurroundingSnapshot snapshot;
            std::string error;
            if (!captureSurroundingSnapshot(ic, snapshot, error)) {
                showError(ic, error.empty() ? "当前输入框不支持获取输入内容" : error);
                keyEvent.filterAndAccept();
                return;
            }
            edit_snapshot_ = std::move(snapshot);
            has_edit_snapshot_ = true;
            if (!is_recording_) {
                startRecording(ic, RecordingMode::Edit);
            }
        } else {
            // F9 / Shift+F9
            has_edit_snapshot_ = false;
            if (!is_recording_) {
                startRecording(
                    ic,
                    shift_mode ? RecordingMode::Long : RecordingMode::Normal
                );
            }
        }
        keyEvent.filterAndAccept();
        return;
    }

    // 其他键：转发给 Rime
    if (!is_release) {
        // 跳过输入法切换热键
        if (isIMSwitchHotkey(key)) {
            return;
        }

        // 构建 Rime modifier mask
        int mask = 0;
        if (key.states() & fcitx::KeyState::Shift) {
            mask |= (1 << 0);  // kShiftMask
        }
        if (key.states() & fcitx::KeyState::CapsLock) {
            mask |= (1 << 1);  // kLockMask
        }
        if (key.states() & fcitx::KeyState::Ctrl) {
            mask |= (1 << 2);  // kControlMask
        }
        if (key.states() & fcitx::KeyState::Alt) {
            mask |= (1 << 3);  // kAltMask
        }

        // 调用 IPC
        try {
            RimeUIState state = ipc_client_->processKey(keyval, mask);

            // 如果有提交文本，先提交
            if (!state.commit_text.empty()) {
                commitText(ic, state.commit_text);
            }

            // 更新 UI
            updateUI(ic, state);

            // 如果被 Rime 处理，则拦截此按键
            if (state.handled) {
                keyEvent.filterAndAccept();
                return;
            }

        } catch (const std::exception& e) {
            FCITX_ERROR() << "Rime key processing failed: " << e.what();
        }
    }
}

void VoCoTypeAddon::reset(const fcitx::InputMethodEntry& entry,
                           fcitx::InputContextEvent& event) {
    auto ic = event.inputContext();
    clearUI(ic);
    ipc_client_->reset();
}

void VoCoTypeAddon::activate(const fcitx::InputMethodEntry& entry,
                              fcitx::InputContextEvent& event) {
    FCITX_DEBUG() << "VoCoType activated";
}

void VoCoTypeAddon::deactivate(const fcitx::InputMethodEntry& entry,
                                fcitx::InputContextEvent& event) {
    auto ic = event.inputContext();
    clearUI(ic);

    // 如果正在录音，停止录音但不转录
    if (is_recording_) {
        stopRecording(ic, false);
    }

    FCITX_DEBUG() << "VoCoType deactivated";
}

void VoCoTypeAddon::startRecording(fcitx::InputContext* ic, RecordingMode mode) {
    if (is_recording_) {
        return;
    }

    if (python_venv_path_.empty() || recorder_script_path_.empty()) {
        showError(ic, "录音配置无效");
        return;
    }

    int stdin_pipe[2];
    int stdout_pipe[2];
    if (pipe(stdin_pipe) != 0) {
        showError(ic, "启动录音失败");
        return;
    }
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        showError(ic, "启动录音失败");
        return;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        showError(ic, "启动录音失败");
        return;
    }

    if (pid == 0) {
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);

        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        execl(python_venv_path_.c_str(),
              python_venv_path_.c_str(),
              recorder_script_path_.c_str(),
              static_cast<char*>(nullptr));
        _exit(127);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    FILE* stdout_file = fdopen(stdout_pipe[0], "r");
    if (!stdout_file) {
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        kill(pid, SIGTERM);
        waitpid(pid, nullptr, 0);
        showError(ic, "启动录音失败");
        return;
    }

    recorder_pid_ = pid;
    recorder_stdin_fd_ = stdin_pipe[1];
    recorder_stdout_ = stdout_file;
    is_recording_ = true;
    recording_mode_ = mode;
    const bool long_mode = (mode == RecordingMode::Long);
    const bool edit_mode = (mode == RecordingMode::Edit);

    // 长句模式按下时并行预加载本地 SLM，减少松键后等待
    if (long_mode) {
        std::thread([this]() {
            (void)ipc_client_->prewarmSlm();
        }).detach();
    }

    // 显示录音状态
    auto& inputPanel = ic->inputPanel();
    fcitx::Text preedit;
    if (long_mode) {
        preedit.append("🎤 录音中(长句)...");
    } else if (edit_mode) {
        preedit.append("🎤 录音中(编辑指令)...");
    } else {
        preedit.append("🎤 录音中...");
    }
    inputPanel.setClientPreedit(preedit);
    ic->updatePreedit();
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);

    FCITX_INFO() << "Recording started, mode="
                 << (edit_mode ? "edit" : (long_mode ? "long" : "normal"));
}

void VoCoTypeAddon::stopAndTranscribe(fcitx::InputContext* ic) {
    stopRecording(ic, true);
}

void VoCoTypeAddon::stopRecording(fcitx::InputContext* ic, bool transcribe) {
    if (!is_recording_) {
        return;
    }

    is_recording_ = false;
    const RecordingMode mode = recording_mode_;
    const bool long_mode = (mode == RecordingMode::Long);
    const bool edit_mode = (mode == RecordingMode::Edit);
    recording_mode_ = RecordingMode::Normal;
    const SurroundingSnapshot edit_snapshot = edit_snapshot_;
    const bool has_edit_snapshot = has_edit_snapshot_;
    has_edit_snapshot_ = false;

    if (ic) {
        if (transcribe) {
            auto& inputPanel = ic->inputPanel();
            fcitx::Text preedit;
            if (edit_mode) {
                preedit.append("⏳ 识别编辑指令中...");
            } else if (long_mode) {
                preedit.append("⏳ 识别+润色中...");
            } else {
                preedit.append("⏳ 识别中...");
            }
            inputPanel.setClientPreedit(preedit);
            ic->updatePreedit();
            ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        } else {
            clearUI(ic);
            if (long_mode || edit_mode) {
                std::thread([this]() {
                    (void)ipc_client_->releaseSlm();
                }).detach();
            }
        }
    }

    pid_t pid = recorder_pid_;
    int stdin_fd = recorder_stdin_fd_;
    FILE* stdout_file = recorder_stdout_;
    recorder_pid_ = -1;
    recorder_stdin_fd_ = -1;
    recorder_stdout_ = nullptr;

    auto ic_ref =
        ic ? ic->watch() : fcitx::TrackableObjectReference<fcitx::InputContext>();

    std::thread([
        this,
        pid,
        stdin_fd,
        stdout_file,
        transcribe,
        long_mode,
        edit_mode,
        edit_snapshot,
        has_edit_snapshot,
        ic_ref
    ]() mutable {
        std::string audio_path = stopRecorderProcess(pid, stdin_fd, stdout_file);
        if (audio_path.empty()) {
            if (transcribe) {
                instance_->eventDispatcher().scheduleWithContext(
                    ic_ref, [this, ic_ref]() {
                        auto* ic_ptr = ic_ref.get();
                        if (ic_ptr) {
                            showError(ic_ptr, "录音失败");
                        }
                    });
            }
            return;
        }

        if (!transcribe) {
            std::remove(audio_path.c_str());
            return;
        }

        TranscribeResult result;
        if (edit_mode && !has_edit_snapshot) {
            result.success = false;
            result.error = "编辑上下文获取失败，请重试";
        } else {
            result = ipc_client_->transcribeAudio(
                audio_path,
                long_mode,
                edit_mode,
                edit_mode ? &edit_snapshot : nullptr
            );
        }
        std::remove(audio_path.c_str());

        instance_->eventDispatcher().scheduleWithContext(
            ic_ref, [this, ic_ref, result, edit_mode, edit_snapshot]() {
                auto* ic_ptr = ic_ref.get();
                if (!ic_ptr) {
                    return;
                }
                if (!result.success) {
                    showError(ic_ptr,
                              result.error.empty() ? "转录失败" : result.error);
                    return;
                }

                if (edit_mode) {
                    if (result.mode == "key_events") {
                        runKeyEvents(ic_ptr, result.key_events, result.hint);
                        return;
                    }
                    if (result.mode == "commit_only") {
                        if (!result.text.empty()) {
                            commitText(ic_ptr, result.text);
                        }
                        if (!result.hint.empty()) {
                            showHint(ic_ptr, result.hint);
                        }
                        return;
                    }
                    if (result.mode == "no_replace") {
                        if (!result.hint.empty()) {
                            showHint(ic_ptr, result.hint);
                        } else {
                            clearUI(ic_ptr);
                        }
                        return;
                    }
                    const std::string target_text =
                        result.text.empty() ? edit_snapshot.text : result.text;
                    replaceSurroundingText(
                        ic_ptr,
                        target_text,
                        edit_snapshot.text,
                        edit_snapshot.cursor_pos,
                        result.hint
                    );
                    return;
                }

                if (!result.text.empty()) {
                    commitText(ic_ptr, result.text);
                } else {
                    clearUI(ic_ptr);
                }
            });
    }).detach();

    FCITX_INFO() << "Recording stopped, mode="
                 << (edit_mode ? "edit" : (long_mode ? "long" : "normal"));
}

bool VoCoTypeAddon::captureSurroundingSnapshot(
    fcitx::InputContext* ic,
    SurroundingSnapshot& snapshot,
    std::string& error
) {
    if (!ic) {
        error = "输入上下文不可用";
        return false;
    }
    ic->updateSurroundingText();
    const auto& surrounding = ic->surroundingText();
    if (!surrounding.isValid()) {
        error = "当前输入框不支持获取输入内容";
        return false;
    }
    snapshot.text = surrounding.text();
    snapshot.cursor_pos = static_cast<int>(surrounding.cursor());
    snapshot.anchor_pos = static_cast<int>(surrounding.anchor());
    snapshot.selected_text = surrounding.selectedText();
    return true;
}

void VoCoTypeAddon::outputSurroundingProbe(fcitx::InputContext* ic) {
    SurroundingSnapshot snapshot;
    std::string error;
    if (!captureSurroundingSnapshot(ic, snapshot, error)) {
        commitText(
            ic,
            "[VT-SURR cap=0 error='" + clipProbeText(error, 64) + "']"
        );
        return;
    }

    auto window = extractSentenceWindow(snapshot.text, snapshot.cursor_pos);
    const std::string probe =
        "[VT-SURR cap=1 "
        "del=" + replace_capability_state_ + " "
        "len=" + std::to_string(snapshot.text.size()) + " "
        "cursor=" + std::to_string(snapshot.cursor_pos) + " "
        "anchor=" + std::to_string(snapshot.anchor_pos) + " "
        "prev='" + clipProbeText(window.second) + "' "
        "cur='" + clipProbeText(window.first) + "' "
        "sel='" + clipProbeText(snapshot.selected_text) + "' "
        "all='" + clipProbeText(snapshot.text, 120) + "']";
    commitText(ic, probe);
}

void VoCoTypeAddon::replaceSurroundingText(
    fcitx::InputContext* ic,
    const std::string& new_text,
    const std::string& original_text,
    int cursor_pos,
    const std::string& hint
) {
    SurroundingSnapshot live_snapshot;
    std::string error;
    if (!captureSurroundingSnapshot(ic, live_snapshot, error)) {
        replace_capability_state_ = "unsupported";
        showError(ic, "当前输入框不支持替换文本");
        return;
    }

    if (live_snapshot.text != original_text || live_snapshot.cursor_pos != cursor_pos) {
        showError(ic, "输入框内容已变化，请重试");
        return;
    }

    if (new_text == original_text) {
        if (!hint.empty()) {
            showHint(ic, hint);
        } else {
            clearUI(ic);
        }
        return;
    }

    const int original_chars = static_cast<int>(fcitx::utf8::length(original_text));
    const int safe_cursor = std::max(0, std::min(cursor_pos, original_chars));
    ic->deleteSurroundingText(-safe_cursor, static_cast<unsigned int>(std::max(0, original_chars)));
    replace_capability_state_ = "supported";
    commitText(ic, new_text);
    if (!hint.empty()) {
        showHint(ic, hint);
    }
}

void VoCoTypeAddon::runKeyEvents(
    fcitx::InputContext* ic,
    const std::vector<std::pair<int, int>>& events,
    const std::string& hint
) {
    if (!ic) {
        return;
    }
    for (const auto& event : events) {
        const int keyval = event.first;
        const int state = event.second;
        fcitx::Key key(
            static_cast<fcitx::KeySym>(keyval),
            keyStateFromMask(state)
        );
        ic->forwardKey(key, false);
        ic->forwardKey(key, true);
    }
    if (!hint.empty()) {
        showHint(ic, hint);
    }
}

void VoCoTypeAddon::updateUI(fcitx::InputContext* ic, const RimeUIState& state) {
    auto& inputPanel = ic->inputPanel();

    // 更新预编辑
    if (!state.preedit_text.empty()) {
        fcitx::Text preedit;
        preedit.append(state.preedit_text, fcitx::TextFormatFlag::Underline);
        inputPanel.setClientPreedit(preedit);
        // 注意：Fcitx5 的 InputPanel 可能没有直接的 setCursor 方法
        // 光标位置通常通过 preedit 的属性设置
        ic->updatePreedit();
    } else {
        inputPanel.setClientPreedit(fcitx::Text());
        ic->updatePreedit();
    }

    // 更新候选词
    if (!state.candidates.empty()) {
        auto candidateList = std::make_unique<fcitx::CommonCandidateList>();
        candidateList->setPageSize(state.page_size);
        candidateList->setCursorPositionAfterPaging(
            fcitx::CursorPositionAfterPaging::ResetToFirst);

        // 设置候选词选择键（数字 1-0）
        candidateList->setSelectionKey({
            fcitx::Key(FcitxKey_1), fcitx::Key(FcitxKey_2), fcitx::Key(FcitxKey_3),
            fcitx::Key(FcitxKey_4), fcitx::Key(FcitxKey_5), fcitx::Key(FcitxKey_6),
            fcitx::Key(FcitxKey_7), fcitx::Key(FcitxKey_8), fcitx::Key(FcitxKey_9),
            fcitx::Key(FcitxKey_0)
        });

        for (size_t i = 0; i < state.candidates.size(); ++i) {
            const auto& [text, comment] = state.candidates[i];
            fcitx::Text candidate_text;
            candidate_text.append(text);
            if (!comment.empty()) {
                candidate_text.append(" ");
                candidate_text.append(comment);
            }
            candidateList->append<fcitx::DisplayOnlyCandidateWord>(candidate_text);
        }

        int cursor_index = state.highlighted_index;
        if (cursor_index < 0 ||
            cursor_index >= static_cast<int>(state.candidates.size())) {
            cursor_index = 0;
        }
        candidateList->setGlobalCursorIndex(cursor_index);
        inputPanel.setCandidateList(std::move(candidateList));
    } else {
        inputPanel.setCandidateList(nullptr);
    }

    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void VoCoTypeAddon::clearUI(fcitx::InputContext* ic) {
    auto& inputPanel = ic->inputPanel();
    inputPanel.reset();
    ic->updatePreedit();
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void VoCoTypeAddon::commitText(fcitx::InputContext* ic, const std::string& text) {
    clearUI(ic);
    ic->commitString(text);
    FCITX_INFO() << "Committed text: " << text;
}

void VoCoTypeAddon::showError(fcitx::InputContext* ic, const std::string& error) {
    auto& inputPanel = ic->inputPanel();
    fcitx::Text preedit;
    preedit.append("❌ " + error);
    inputPanel.setClientPreedit(preedit);
    ic->updatePreedit();
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);

    // 简化：不自动清除，等待用户下次按键
    // 2 秒自动清除在 Fcitx5 中需要更复杂的实现
}

void VoCoTypeAddon::showHint(fcitx::InputContext* ic, const std::string& hint) {
    auto& inputPanel = ic->inputPanel();
    fcitx::Text preedit;
    preedit.append(hint);
    inputPanel.setClientPreedit(preedit);
    ic->updatePreedit();
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

bool VoCoTypeAddon::isIMSwitchHotkey(const fcitx::Key& key) const {
    // 只拦截 Super+Space (输入法切换)，不拦截 Ctrl+Space (中英切换)
    if (key.sym() == FcitxKey_space) {
        if (key.states() & fcitx::KeyState::Super) {
            return true;
        }
    }

    // Ctrl+Shift 或 Alt+Shift
    if (key.sym() == FcitxKey_Shift_L || key.sym() == FcitxKey_Shift_R) {
        if (key.states() & fcitx::KeyState::Ctrl) {
            return true;
        }
        if (key.states() & fcitx::KeyState::Alt) {
            return true;
        }
    }

    return false;
}

} // namespace vocotype

// Fcitx5 插件注册
class VoCoTypeAddonFactory : public fcitx::AddonFactory {
    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
        return new vocotype::VoCoTypeAddon(manager->instance());
    }
};

FCITX_ADDON_FACTORY(VoCoTypeAddonFactory);
