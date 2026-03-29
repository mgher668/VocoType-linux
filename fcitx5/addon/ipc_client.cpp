/*
 * VoCoType Fcitx5 Addon - IPC Client Implementation
 */

#include "ipc_client.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace vocotype {

IPCClient::IPCClient(const std::string& socket_path)
    : socket_path_(socket_path) {
}

IPCClient::~IPCClient() {
}

std::string IPCClient::sendRequest(const std::string& request) {
    // 创建 Unix Socket
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    // 连接到服务器
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        throw std::runtime_error("Failed to connect to backend: " + socket_path_);
    }

    // 发送请求（确保全部写入）
    size_t total_sent = 0;
    while (total_sent < request.size()) {
        ssize_t sent = send(sock, request.data() + total_sent,
                            request.size() - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(sock);
            throw std::runtime_error("Failed to send request");
        }
        total_sent += static_cast<size_t>(sent);
    }

    shutdown(sock, SHUT_WR);

    // 接收响应（读到 EOF）
    std::string response;
    char buffer[4096];
    while (true) {
        ssize_t len = recv(sock, buffer, sizeof(buffer), 0);
        if (len < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(sock);
            throw std::runtime_error("Failed to receive response");
        }
        if (len == 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(len));
    }

    close(sock);
    return response;
}

TranscribeResult IPCClient::transcribeAudio(
    const std::string& audio_path,
    bool long_mode,
    bool edit_mode,
    const SurroundingSnapshot* surrounding
) {
    TranscribeResult result;

    try {
        // 构建请求
        json request = {
            {"type", "transcribe"},
            {"audio_path", audio_path},
            {"long_mode", long_mode},
            {"edit_mode", edit_mode},
        };
        if (edit_mode && surrounding) {
            request["surrounding"] = {
                {"text", surrounding->text},
                {"cursor", surrounding->cursor_pos},
                {"anchor", surrounding->anchor_pos},
                {"selected", surrounding->selected_text},
            };
        }

        // 发送请求
        std::string response_str = sendRequest(request.dump());

        // 解析响应
        json response = json::parse(response_str);

        result.success = response.value("success", false);
        result.mode = response.value("mode", "");
        result.hint = response.value("hint", "");
        result.record_history = response.value("record_history", false);
        if (response.contains("key_events") && response["key_events"].is_array()) {
            for (const auto& event : response["key_events"]) {
                int keyval = event.value("keyval", 0);
                int state = event.value("state", 0);
                result.key_events.emplace_back(keyval, state);
            }
        }
        if (result.success) {
            result.text = response.value("text", "");
        } else {
            result.error = response.value("error", "Unknown error");
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
    }

    return result;
}

bool IPCClient::prewarmSlm() {
    try {
        json request = {{"type", "slm_prewarm"}};
        std::string response_str = sendRequest(request.dump());
        json response = json::parse(response_str);
        return response.value("success", false);
    } catch (const std::exception& e) {
        return false;
    }
}

bool IPCClient::releaseSlm() {
    try {
        json request = {{"type", "slm_release"}};
        std::string response_str = sendRequest(request.dump());
        json response = json::parse(response_str);
        return response.value("success", false);
    } catch (const std::exception& e) {
        return false;
    }
}

RimeUIState IPCClient::processKey(int keyval, int mask) {
    RimeUIState state;

    try {
        // 构建请求
        json request = {
            {"type", "key_event"},
            {"keyval", keyval},
            {"mask", mask}
        };

        // 发送请求
        std::string response_str = sendRequest(request.dump());

        // 解析响应
        json response = json::parse(response_str);

        state.handled = response.value("handled", false);

        // 提交文本
        if (response.contains("commit")) {
            state.commit_text = response["commit"];
        }

        // 预编辑
        if (response.contains("preedit")) {
            state.preedit_text = response["preedit"]["text"];
            state.cursor_pos = response["preedit"]["cursor_pos"];
        }

        // 候选词
        if (response.contains("candidates")) {
            for (const auto& candidate : response["candidates"]) {
                std::string text = candidate["text"];
                std::string comment = candidate["comment"];
                state.candidates.push_back({text, comment});
            }
            state.highlighted_index = response.value("highlighted_index", 0);
            state.page_size = response.value("page_size", 5);
        }

    } catch (const std::exception& e) {
        // 错误时返回未处理状态
        state.handled = false;
    }

    return state;
}

void IPCClient::reset() {
    try {
        json request = {{"type", "reset"}};
        sendRequest(request.dump());
    } catch (const std::exception& e) {
        // 忽略错误
    }
}

bool IPCClient::ping() {
    try {
        json request = {{"type", "ping"}};
        std::string response_str = sendRequest(request.dump());
        json response = json::parse(response_str);
        return response.value("pong", false);
    } catch (const std::exception& e) {
        return false;
    }
}

} // namespace vocotype
