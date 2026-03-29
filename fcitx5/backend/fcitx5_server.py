#!/usr/bin/env python3
"""Fcitx 5 Python 后端服务（语音 + Rime）

此服务作为独立进程运行，通过 Unix Socket 接收来自 C++ Addon 的请求，
提供语音识别和 Rime 拼音输入功能。
"""
from __future__ import annotations

import sys
import os
import json
import socket
import logging
import re
import signal
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_CONFIG, ensure_logging_dir, load_config
from app.funasr_server import FunASRServer
from app.logging_config import setup_logging
from app.slm_polisher import SLMPolisher
from backend.rime_handler import RimeHandler

logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/vocotype-fcitx5.sock"
MAX_REQUEST_BYTES = 1024 * 1024
REQUEST_TIMEOUT_S = 2.0
DEFAULT_CONFIG_PATH = "~/.config/vocotype/fcitx5-backend.json"


@dataclass
class SurroundingSnapshot:
    text: str
    cursor_pos: int
    anchor_pos: int
    selected_text: str


@dataclass
class DirectEditResult:
    handled: bool
    new_text: Optional[str] = None
    record_history: bool = True
    hint: str = ""
    mode: str = "replace"  # replace / key_events / no_replace / commit_only
    key_events: list[dict[str, int]] | None = None


def load_backend_config() -> tuple[dict, str]:
    """Load backend config from user config file if present."""
    config_path = os.environ.get("VOCOTYPE_FCITX5_CONFIG", DEFAULT_CONFIG_PATH)
    expanded_path = os.path.expanduser(config_path)
    if not os.path.exists(expanded_path):
        return dict(DEFAULT_CONFIG), expanded_path

    try:
        return load_config(expanded_path), expanded_path
    except Exception as exc:
        print(f"Failed to load config {expanded_path}: {exc}", file=sys.stderr)
        return dict(DEFAULT_CONFIG), expanded_path


def configure_logging(config: dict, debug: bool) -> None:
    """Configure logging with optional file output."""
    logging_cfg = config.get("logging", {})
    level = "DEBUG" if debug else logging_cfg.get("level", "INFO")
    write_file = bool(logging_cfg.get("file", False))
    log_dir = ensure_logging_dir(config) if write_file else None
    setup_logging(level=level, log_dir=log_dir)


class Fcitx5Backend:
    """Fcitx 5 Python 后端服务

    职责：
    1. 接收语音识别请求，调用 FunASRServer
    2. 接收 Rime 按键请求，调用 RimeHandler
    3. 通过 IPC 返回结果给 C++ Addon
    """

    CTRL_MASK = 1 << 2
    SHIFT_MASK = 1 << 0
    ALT_MASK = 1 << 3
    EDIT_HISTORY_LIMIT = 20
    KEY_LEFT = 65361
    KEY_UP = 65362
    KEY_RIGHT = 65363
    KEY_DOWN = 65364
    KEY_HOME = 65360
    KEY_END = 65367
    KEY_A = 97
    KEY_Z = 122
    _PUNCTUATION_MAP = {
        "句号": "。",
        "逗号": "，",
        "问号": "？",
        "感叹号": "！",
        "冒号": "：",
        "分号": "；",
        "引号": "“”",
    }

    def __init__(self, config: dict | None = None):
        self.config = dict(config or DEFAULT_CONFIG)

        # 语音识别服务
        logger.info("正在初始化 FunASR 服务器...")
        self.asr_server = FunASRServer()
        asr_result = self.asr_server.initialize()
        if not asr_result['success']:
            logger.error("FunASR 初始化失败: %s", asr_result.get('error'))
            sys.exit(1)
        logger.info("FunASR 服务器初始化成功")

        self._asr_options = dict(self.config.get("asr", {}))
        self._slm_polisher = SLMPolisher(self.config.get("slm", {}))
        logger.info("SLM 长句润色: enabled=%s", self._slm_polisher.enabled)

        # Rime 处理器
        self.rime_handler = RimeHandler()
        if self.rime_handler.available:
            logger.info("Rime 集成已启用")
        else:
            logger.info("Rime 集成未启用（纯语音模式）")

        # 标记运行状态
        self.running = True
        self._asr_lock = threading.Lock()
        self._rime_lock = threading.Lock()
        self._edit_lock = threading.Lock()
        self._edit_undo_stack: list[str] = []
        self._edit_redo_stack: list[str] = []
        self._voice_clipboard = ""
        self._last_text_change_source = "none"  # none / voice_edit / app_commit
        self._last_internal_edit_text: Optional[str] = None

        # 注册信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _cleanup_socket_path(self, path: str) -> None:
        """安全删除旧 socket 文件（避免误删普通文件）"""
        if not os.path.exists(path):
            return

        try:
            st = os.lstat(path)
        except OSError as exc:
            logger.warning("检查旧 socket 失败: %s", exc)
            return

        if stat.S_ISSOCK(st.st_mode) or stat.S_ISLNK(st.st_mode):
            try:
                os.remove(path)
                logger.info("已移除旧 socket: %s", path)
            except OSError as exc:
                logger.warning("移除旧 socket 失败: %s", exc)
        else:
            raise RuntimeError(f"socket 路径已存在且不是 socket: {path}")

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info("收到信号 %d，准备退出...", signum)
        self.running = False

    @staticmethod
    def _snapshot_from_request(payload: Any) -> Optional[SurroundingSnapshot]:
        if not isinstance(payload, dict):
            return None
        text = str(payload.get("text", ""))
        try:
            cursor = int(payload.get("cursor", 0))
            anchor = int(payload.get("anchor", cursor))
        except Exception:
            return None
        selected = str(payload.get("selected", ""))
        if not selected:
            start, end = sorted((max(0, cursor), max(0, anchor)))
            selected = text[start:end] if end > start else ""
        return SurroundingSnapshot(
            text=text,
            cursor_pos=cursor,
            anchor_pos=anchor,
            selected_text=selected,
        )

    @staticmethod
    def _normalize_voice_command(command: str) -> str:
        cmd = " ".join((command or "").strip().split())
        if not cmd:
            return ""
        cmd = re.sub(r"^(?:请|麻烦|帮我|帮忙)\s*", "", cmd)
        cmd = re.sub(r"(一下子?|吧)$", "", cmd)
        cmd = re.sub(r"[。！？!?，,；;：:]+$", "", cmd)
        return cmd.strip()

    def _rewrite_insert_generation_instruction(self, command: str) -> str:
        cmd = self._normalize_voice_command(command)
        if not cmd:
            return ""

        match = re.match(
            r"^(?:输入|写|写一段|生成|生成一段|来一段)\s*(.+)\s*$",
            cmd,
        )
        if not match:
            return ""

        request = self._strip_command_quotes(match.group(1))
        if not request:
            return ""

        return (
            "请按以下要求生成并插入文本："
            f"{request}。"
            "将生成结果插入到当前光标位置；如果当前有选中文本，则替换选中内容。"
            "除插入/替换位置外，不要改动任何其他文本。"
            "只输出编辑后的完整输入框文本。"
        )

    @staticmethod
    def _strip_command_quotes(text: str) -> str:
        return str(text or "").strip().strip("“”\"'")

    @staticmethod
    def _parse_count_from_command(cmd: str) -> int:
        digit_match = re.search(r"(\d+)", cmd)
        if digit_match:
            return max(1, min(20, int(digit_match.group(1))))

        cn_map = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        for ch, value in cn_map.items():
            if ch in cmd:
                return value
        return 1

    @staticmethod
    def _key_events(
        keyval: int,
        *,
        state: int = 0,
        repeat: int = 1,
    ) -> list[dict[str, int]]:
        count = max(1, min(20, int(repeat)))
        return [{"keyval": int(keyval), "state": int(state)} for _ in range(count)]

    def _push_undo_state(self, text: str) -> None:
        if self._edit_undo_stack and self._edit_undo_stack[-1] == text:
            return
        self._edit_undo_stack.append(text)
        if len(self._edit_undo_stack) > self.EDIT_HISTORY_LIMIT:
            self._edit_undo_stack.pop(0)
        self._edit_redo_stack.clear()

    @staticmethod
    def _predict_commit_result(snapshot: SurroundingSnapshot, payload: str) -> str:
        text = snapshot.text or ""
        cursor = max(0, min(int(snapshot.cursor_pos), len(text)))
        anchor = max(0, min(int(snapshot.anchor_pos), len(text)))
        sel_start, sel_end = sorted((anchor, cursor))
        if sel_end > sel_start:
            return text[:sel_start] + payload + text[sel_end:]
        return text[:cursor] + payload + text[cursor:]

    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        delimiters = set("。！？!?；;.\n")
        spans: list[tuple[int, int]] = []
        start = 0
        for idx, ch in enumerate(text):
            if ch in delimiters:
                end = idx + 1
                if end > start:
                    spans.append((start, end))
                start = end
        if start < len(text):
            spans.append((start, len(text)))
        return spans

    @staticmethod
    def _locate_sentence_index(spans: list[tuple[int, int]], cursor_pos: int) -> int:
        if not spans:
            return -1
        cursor = max(0, cursor_pos)
        for idx, (seg_start, seg_end) in enumerate(spans):
            if seg_start <= cursor <= seg_end:
                return idx
        return len(spans) - 1

    @staticmethod
    def _clip_probe_text(text: str, limit: int = 48) -> str:
        cleaned = (text or "").replace("\n", "⏎").replace("\t", "⇥")
        cleaned = " ".join(cleaned.split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit]}..."

    @staticmethod
    def _extract_sentence_window(text: str, cursor_pos: int) -> tuple[str, str]:
        if not text:
            return "", ""
        delimiters = set("。！？!?；;.\n")
        spans: list[tuple[int, int]] = []
        start = 0
        for idx, ch in enumerate(text):
            if ch in delimiters:
                end = idx + 1
                if end > start:
                    spans.append((start, end))
                start = end
        if start < len(text):
            spans.append((start, len(text)))
        if not spans:
            return text.strip(), ""
        cursor = max(0, min(cursor_pos, len(text)))
        current_idx = len(spans) - 1
        for i, (seg_start, seg_end) in enumerate(spans):
            if seg_start <= cursor <= seg_end:
                current_idx = i
                break
        cur_start, cur_end = spans[current_idx]
        current = text[cur_start:cur_end].strip()
        previous = ""
        if current_idx > 0:
            prev_start, prev_end = spans[current_idx - 1]
            previous = text[prev_start:prev_end].strip()
        return current, previous

    def _apply_direct_edit_command(
        self,
        snapshot: SurroundingSnapshot,
        instruction: str,
    ) -> DirectEditResult:
        cmd = self._normalize_voice_command(instruction)
        if not cmd:
            return DirectEditResult(False)

        text = snapshot.text
        cursor = snapshot.cursor_pos
        anchor = snapshot.anchor_pos
        lower_cmd = cmd.lower()

        if lower_cmd in {
            "显示上下文",
            "显示上下文信息",
            "输出上下文",
            "输出上下文信息",
            "显示surrounding信息",
            "输出surrounding信息",
            "surrounding info",
            "context info",
        }:
            current_sentence, previous_sentence = self._extract_sentence_window(text, cursor)
            report = (
                "[VT-SURR "
                "cap=1 "
                f"len={len(text)} cursor={cursor} anchor={anchor} "
                f"prev='{self._clip_probe_text(previous_sentence)}' "
                f"cur='{self._clip_probe_text(current_sentence)}' "
                f"sel='{self._clip_probe_text(snapshot.selected_text)}' "
                f"all='{self._clip_probe_text(text, 120)}'"
                "]"
            )
            return DirectEditResult(
                handled=True,
                mode="commit_only",
                new_text=report,
                record_history=True,
                hint="已输出上下文信息",
            )

        sel_start, sel_end = sorted((anchor, cursor))
        selected_text = text[sel_start:sel_end] if sel_end > sel_start else ""

        if lower_cmd in {"撤销", "撤回", "撤销修改", "撤销上一步", "undo"}:
            can_internal_undo = (
                bool(self._edit_undo_stack)
                and self._last_text_change_source == "voice_edit"
                and self._last_internal_edit_text == text
            )
            if can_internal_undo:
                previous = self._edit_undo_stack.pop()
                self._edit_redo_stack.append(text)
                if len(self._edit_redo_stack) > self.EDIT_HISTORY_LIMIT:
                    self._edit_redo_stack.pop(0)
                return DirectEditResult(True, previous, record_history=False, hint="已撤销语音编辑")

            self._last_text_change_source = "app_commit"
            self._last_internal_edit_text = None
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_Z, state=self.CTRL_MASK),
                record_history=False,
                hint="已发送应用撤销",
            )

        if lower_cmd in {"重做", "恢复", "redo"}:
            can_internal_redo = (
                bool(self._edit_redo_stack)
                and self._last_text_change_source == "voice_edit"
                and self._last_internal_edit_text == text
            )
            if can_internal_redo:
                recovered = self._edit_redo_stack.pop()
                self._edit_undo_stack.append(text)
                if len(self._edit_undo_stack) > self.EDIT_HISTORY_LIMIT:
                    self._edit_undo_stack.pop(0)
                return DirectEditResult(True, recovered, record_history=False, hint="已重做语音编辑")

            self._last_text_change_source = "app_commit"
            self._last_internal_edit_text = None
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(
                    self.KEY_Z,
                    state=self.CTRL_MASK | self.SHIFT_MASK,
                ),
                record_history=False,
                hint="已发送应用重做",
            )

        if lower_cmd in {"复制全部", "复制全文", "copy all"}:
            self._voice_clipboard = text
            return DirectEditResult(True, text, record_history=False, hint="已复制全文")

        if lower_cmd in {"复制选中", "复制选中内容", "copy that"}:
            if not selected_text:
                return DirectEditResult(True, text, record_history=False, hint="当前没有选中内容")
            self._voice_clipboard = selected_text
            return DirectEditResult(True, text, record_history=False, hint="已复制选中内容")

        if lower_cmd in {"剪切全部", "剪切全文", "cut all"}:
            self._voice_clipboard = text
            return DirectEditResult(True, "", record_history=True, hint="已剪切全文")

        if lower_cmd in {"剪切选中", "剪切选中内容", "cut that"}:
            if not selected_text:
                return DirectEditResult(True, text, record_history=False, hint="当前没有选中内容")
            self._voice_clipboard = selected_text
            return DirectEditResult(
                True,
                text[:sel_start] + text[sel_end:],
                record_history=True,
                hint="已剪切选中内容",
            )

        if lower_cmd in {"粘贴", "贴上", "paste"}:
            if not self._voice_clipboard:
                return DirectEditResult(True, text, record_history=False, hint="剪贴板为空")
            if sel_end > sel_start:
                merged = text[:sel_start] + self._voice_clipboard + text[sel_end:]
            else:
                merged = text[:cursor] + self._voice_clipboard + text[cursor:]
            return DirectEditResult(True, merged, record_history=True, hint="已粘贴")

        if lower_cmd in {"清空", "清空输入框", "删除全部", "删掉全部", "全选删除"}:
            return DirectEditResult(True, "", record_history=True, hint="已清空")

        if lower_cmd in {"删除选中", "删除选中内容"}:
            if not selected_text:
                return DirectEditResult(True, text, record_history=False, hint="当前没有选中内容")
            return DirectEditResult(
                True,
                text[:sel_start] + text[sel_end:],
                record_history=True,
                hint="已删除选中内容",
            )

        if lower_cmd in {"删除当前句", "删掉当前句"}:
            spans = self._sentence_spans(text)
            idx = self._locate_sentence_index(spans, cursor)
            if idx < 0:
                return DirectEditResult(True, text, record_history=False, hint="未找到当前句")
            start, end = spans[idx]
            return DirectEditResult(True, text[:start] + text[end:], record_history=True, hint="已删除当前句")

        if lower_cmd in {"删除上一句", "删掉上一句"}:
            spans = self._sentence_spans(text)
            idx = self._locate_sentence_index(spans, cursor)
            if idx <= 0:
                return DirectEditResult(True, text, record_history=False, hint="没有上一句可删除")
            start, end = spans[idx - 1]
            return DirectEditResult(True, text[:start] + text[end:], record_history=True, hint="已删除上一句")

        replace_match = re.match(
            r"^(?:把|将)\s*(.+?)\s*(?:改成|改为|替换成|替换为)\s*(.+)\s*$",
            cmd,
        )
        if replace_match:
            old = self._strip_command_quotes(replace_match.group(1))
            new = self._strip_command_quotes(replace_match.group(2))
            if not old:
                return DirectEditResult(True, text, record_history=False, hint="替换目标为空")
            if old not in text:
                return DirectEditResult(True, text, record_history=False, hint=f"未找到“{old}”")
            return DirectEditResult(
                True,
                text.replace(old, new, 1),
                record_history=True,
                hint="已替换",
            )

        insert_before_match = re.match(r"^在\s*(.+?)\s*(?:前面|前)\s*插入\s*(.+)\s*$", cmd)
        if insert_before_match:
            marker = self._strip_command_quotes(insert_before_match.group(1))
            payload = self._strip_command_quotes(insert_before_match.group(2))
            idx = text.find(marker)
            if idx < 0:
                return DirectEditResult(True, text, record_history=False, hint=f"未找到“{marker}”")
            return DirectEditResult(True, text[:idx] + payload + text[idx:], record_history=True, hint="已插入")

        insert_after_match = re.match(r"^在\s*(.+?)\s*(?:后面|后)\s*插入\s*(.+)\s*$", cmd)
        if insert_after_match:
            marker = self._strip_command_quotes(insert_after_match.group(1))
            payload = self._strip_command_quotes(insert_after_match.group(2))
            idx = text.find(marker)
            if idx < 0:
                return DirectEditResult(True, text, record_history=False, hint=f"未找到“{marker}”")
            end = idx + len(marker)
            return DirectEditResult(True, text[:end] + payload + text[end:], record_history=True, hint="已插入")

        prepend_match = re.match(r"^(?:在)?(?:开头|最前面)(?:插入|添加|加上)\s*(.+)\s*$", cmd)
        if prepend_match:
            payload = self._strip_command_quotes(prepend_match.group(1))
            return DirectEditResult(True, payload + text, record_history=True, hint="已在开头插入")

        append_match = re.match(r"^(?:在)?(?:结尾|末尾|最后)(?:插入|添加|加上|追加)\s*(.+)\s*$", cmd)
        if append_match:
            payload = self._strip_command_quotes(append_match.group(1))
            return DirectEditResult(True, text + payload, record_history=True, hint="已在结尾插入")

        append_simple_match = re.match(r"^(?:追加|添加|加上)\s*(.+)\s*$", cmd)
        if append_simple_match:
            payload = self._strip_command_quotes(append_simple_match.group(1))
            return DirectEditResult(True, text + payload, record_history=True, hint="已追加")

        punct_match = re.match(r"^(?:加|插入)\s*(句号|逗号|问号|感叹号|冒号|分号|引号)\s*$", cmd)
        if punct_match:
            punct = self._PUNCTUATION_MAP.get(punct_match.group(1), "")
            if punct:
                return DirectEditResult(True, text + punct, record_history=True, hint="已添加标点")

        if lower_cmd in {"全部大写", "全大写", "uppercase"}:
            return DirectEditResult(True, text.upper(), record_history=True, hint="已转为大写")
        if lower_cmd in {"全部小写", "全小写", "lowercase"}:
            return DirectEditResult(True, text.lower(), record_history=True, hint="已转为小写")
        if lower_cmd in {"首字母大写", "标题格式", "title case"}:
            return DirectEditResult(True, text.title(), record_history=True, hint="已转为首字母大写")
        if lower_cmd in {"加粗", "加粗选中", "bold", "bold that"}:
            if sel_end > sel_start:
                styled = text[:sel_start] + f"**{selected_text}**" + text[sel_end:]
            else:
                styled = f"**{text}**"
            return DirectEditResult(True, styled, record_history=True, hint="已加粗")
        if lower_cmd in {"斜体", "斜体选中", "italic", "italicize"}:
            if sel_end > sel_start:
                styled = text[:sel_start] + f"*{selected_text}*" + text[sel_end:]
            else:
                styled = f"*{text}*"
            return DirectEditResult(True, styled, record_history=True, hint="已设为斜体")

        delete_match = re.match(r"^(?:删除|删掉|去掉)\s*(.+)\s*$", cmd)
        if delete_match:
            target = self._strip_command_quotes(delete_match.group(1))
            if target in {"当前句", "上一句", "全部", "选中内容", "选中"}:
                return DirectEditResult(False)
            if target and target in text:
                return DirectEditResult(
                    True,
                    text.replace(target, "", 1),
                    record_history=True,
                    hint="已删除",
                )
            return DirectEditResult(True, text, record_history=False, hint=f"未找到“{target}”")

        count = self._parse_count_from_command(cmd)
        if lower_cmd in {"全选", "选中全部", "select all"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_A, state=self.CTRL_MASK),
                record_history=False,
                hint="已全选",
            )
        if lower_cmd in {"移动到开头", "跳到开头", "到开头", "行首", "到行首", "移动到行首"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_HOME),
                record_history=False,
                hint="已移动到开头",
            )
        if lower_cmd in {"移动到结尾", "跳到结尾", "到结尾", "行尾", "到行尾", "移动到行尾"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_END),
                record_history=False,
                hint="已移动到结尾",
            )
        if lower_cmd in {"段首", "到段首", "移动到段首"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_UP, state=self.CTRL_MASK),
                record_history=False,
                hint="已尝试移动到段首",
            )
        if lower_cmd in {"段尾", "到段尾", "移动到段尾"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_DOWN, state=self.CTRL_MASK),
                record_history=False,
                hint="已尝试移动到段尾",
            )

        if re.match(r"^(?:向|往)?左(?:移|移动)?(?:\s*\d+|\s*[一二两三四五六七八九十])?(?:次|个字|个字符)?$", cmd) or lower_cmd in {"左移", "向左"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_LEFT, repeat=count),
                record_history=False,
                hint=f"已左移{count}次",
            )
        if re.match(r"^(?:向|往)?右(?:移|移动)?(?:\s*\d+|\s*[一二两三四五六七八九十])?(?:次|个字|个字符)?$", cmd) or lower_cmd in {"右移", "向右"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_RIGHT, repeat=count),
                record_history=False,
                hint=f"已右移{count}次",
            )

        if lower_cmd in {"下一个词", "到下一个词", "移动到下一个词", "next word"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_RIGHT, state=self.CTRL_MASK, repeat=count),
                record_history=False,
                hint="已移动到下一个词",
            )
        if lower_cmd in {"上一个词", "到上一个词", "移动到上一个词", "previous word"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(self.KEY_LEFT, state=self.CTRL_MASK, repeat=count),
                record_history=False,
                hint="已移动到上一个词",
            )
        if lower_cmd in {"选中下一个词", "选择下一个词"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(
                    self.KEY_RIGHT,
                    state=self.CTRL_MASK | self.SHIFT_MASK,
                    repeat=count,
                ),
                record_history=False,
                hint="已尝试选中下一个词",
            )
        if lower_cmd in {"选中上一个词", "选择上一个词"}:
            return DirectEditResult(
                handled=True,
                mode="key_events",
                key_events=self._key_events(
                    self.KEY_LEFT,
                    state=self.CTRL_MASK | self.SHIFT_MASK,
                    repeat=count,
                ),
                record_history=False,
                hint="已尝试选中上一个词",
            )

        return DirectEditResult(False)

    def run(self):
        """运行 IPC 服务器"""
        # 删除旧的 socket 文件
        self._cleanup_socket_path(SOCKET_PATH)

        # 创建 Unix Socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)
        sock.listen(5)
        sock.settimeout(1.0)  # 设置超时以便处理信号

        logger.info("Fcitx5 Backend 已启动，监听: %s", SOCKET_PATH)

        try:
            while self.running:
                try:
                    conn, _ = sock.accept()
                    threading.Thread(
                        target=self.handle_client,
                        args=(conn,),
                        daemon=True,
                        name="Fcitx5BackendClient",
                    ).start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self.running:
                        logger.error("接受连接失败: %s", exc)
        finally:
            sock.close()
            try:
                self._cleanup_socket_path(SOCKET_PATH)
            except RuntimeError as exc:
                logger.warning("清理 socket 失败: %s", exc)
            logger.info("Fcitx5 Backend 已停止")

    def handle_client(self, conn: socket.socket):
        """处理客户端请求

        IPC 协议：
        - 请求格式：JSON 字符串
        - 响应格式：JSON 字符串

        请求类型：
        1. transcribe: 语音识别
           {"type": "transcribe", "audio_path": "/tmp/xxx.wav", "long_mode": false}
           -> {"success": true, "text": "识别结果"}

        2. slm_prewarm: 预加载 SLM（长句模式按下时调用）
           {"type": "slm_prewarm"}
           -> {"success": true}

        3. slm_release: 释放 SLM（长句流程结束时调用）
           {"type": "slm_release"}
           -> {"success": true}

        4. key_event: Rime 按键处理
           {"type": "key_event", "keyval": 97, "mask": 0}
           -> {"handled": true, "commit": "...", "preedit": {...}, ...}

        5. reset: 重置 Rime 状态
           {"type": "reset"}
           -> {"success": true}

        6. ping: 健康检查
           {"type": "ping"}
           -> {"pong": true}
        """
        try:
            conn.settimeout(REQUEST_TIMEOUT_S)
            # 接收请求（读到 EOF）
            chunks = []
            total_bytes = 0
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes > MAX_REQUEST_BYTES:
                    response_str = json.dumps({"error": "Request too large"}, ensure_ascii=False)
                    conn.sendall(response_str.encode('utf-8'))
                    return
            if not chunks:
                return
            data = b''.join(chunks).decode('utf-8')

            request = json.loads(data)
            req_type = request.get('type')

            logger.debug("收到请求: type=%s", req_type)

            # 处理请求
            if req_type == 'transcribe':
                # 语音识别
                audio_path = request.get('audio_path')
                long_mode = bool(request.get('long_mode', False))
                edit_mode = bool(request.get('edit_mode', False))
                edit_snapshot = (
                    self._snapshot_from_request(request.get("surrounding"))
                    if edit_mode
                    else None
                )
                if not audio_path:
                    response = {"success": False, "error": "缺少 audio_path 参数"}
                elif edit_mode and edit_snapshot is None:
                    response = {"success": False, "error": "编辑上下文无效"}
                else:
                    try:
                        asr_start = time.perf_counter()
                        with self._asr_lock:
                            result = self.asr_server.transcribe_audio(
                                audio_path,
                                options=self._asr_options,
                            )
                        asr_ms = (time.perf_counter() - asr_start) * 1000.0

                        slm_used = False
                        slm_ms = 0.0
                        slm_reason = "not_used"
                        if result.get("success"):
                            text = str(result.get("text", "")).strip()
                            if text:
                                if edit_mode and edit_snapshot is not None:
                                    with self._edit_lock:
                                        direct_result = self._apply_direct_edit_command(
                                            edit_snapshot,
                                            text,
                                        )
                                        if direct_result.handled:
                                            if (
                                                direct_result.mode == "commit_only"
                                                and direct_result.record_history
                                                and direct_result.new_text
                                            ):
                                                self._push_undo_state(edit_snapshot.text)
                                                self._last_internal_edit_text = self._predict_commit_result(
                                                    edit_snapshot,
                                                    direct_result.new_text,
                                                )
                                                self._last_text_change_source = "voice_edit"
                                            elif direct_result.mode == "replace" and direct_result.new_text is not None:
                                                if direct_result.record_history:
                                                    self._push_undo_state(edit_snapshot.text)
                                                self._last_internal_edit_text = direct_result.new_text
                                                self._last_text_change_source = "voice_edit"

                                            response = {
                                                "success": True,
                                                "mode": direct_result.mode,
                                                "hint": direct_result.hint,
                                                "record_history": bool(direct_result.record_history),
                                            }
                                            if direct_result.new_text is not None:
                                                response["text"] = direct_result.new_text
                                            if direct_result.key_events:
                                                response["key_events"] = direct_result.key_events
                                            slm_reason = "direct_command"

                                    if not direct_result.handled:
                                        slm_instruction = (
                                            self._rewrite_insert_generation_instruction(text)
                                            or text
                                        )
                                        edited_text, metrics = self._slm_polisher.edit_with_instruction(
                                            context_text=edit_snapshot.text,
                                            instruction=slm_instruction,
                                            cursor_pos=edit_snapshot.cursor_pos,
                                            anchor_pos=edit_snapshot.anchor_pos,
                                            selected_text=edit_snapshot.selected_text,
                                        )
                                        slm_ms = metrics.latency_ms
                                        slm_reason = metrics.reason
                                        slm_used = metrics.used
                                        if self._slm_polisher.is_failure_reason(metrics.reason):
                                            logger.warning(
                                                "编辑模式 SLM 调用失败: reason=%s",
                                                metrics.reason,
                                            )
                                            response = {
                                                "success": False,
                                                "error": self._slm_polisher.format_failure_message(
                                                    metrics.reason
                                                ),
                                            }
                                        else:
                                            with self._edit_lock:
                                                self._push_undo_state(edit_snapshot.text)
                                                self._last_internal_edit_text = edited_text
                                                self._last_text_change_source = "voice_edit"
                                            response = {
                                                "success": True,
                                                "mode": "replace",
                                                "text": edited_text,
                                                "hint": "",
                                                "record_history": True,
                                            }
                                else:
                                    should_polish = (
                                        long_mode
                                        and self._slm_polisher.should_polish(
                                            text,
                                            long_mode=True,
                                        )
                                    )
                                    if should_polish:
                                        polished_text, metrics = self._slm_polisher.polish(
                                            text,
                                            long_mode=long_mode,
                                        )
                                        slm_used = metrics.used
                                        slm_ms = metrics.latency_ms
                                        slm_reason = metrics.reason
                                        if self._slm_polisher.is_failure_reason(metrics.reason):
                                            logger.warning(
                                                "长句 SLM 调用失败: reason=%s",
                                                metrics.reason,
                                            )
                                            result = {
                                                "success": False,
                                                "error": self._slm_polisher.format_failure_message(
                                                    metrics.reason
                                                ),
                                            }
                                        else:
                                            result["text"] = polished_text
                                    elif long_mode:
                                        slm_reason = (
                                            "disabled"
                                            if not self._slm_polisher.enabled
                                            else "too_short"
                                        )
                            else:
                                slm_reason = "empty_asr_text"
                                if edit_mode:
                                    response = {
                                        "success": True,
                                        "mode": "no_replace",
                                        "hint": "未识别到编辑指令",
                                    }
                        elif edit_mode:
                            response = result

                        if not edit_mode:
                            response = result
                            if response.get("success") and str(response.get("text", "")).strip():
                                with self._edit_lock:
                                    self._last_text_change_source = "app_commit"
                                    self._last_internal_edit_text = None

                        logger.info(
                            "Fcitx 转录流水线 mode=%s asr_ms=%.2f slm_used=%s slm_ms=%.2f fallback_reason=%s",
                            "edit" if edit_mode else ("long" if long_mode else "normal"),
                            asr_ms,
                            slm_used,
                            slm_ms,
                            slm_reason,
                        )
                    finally:
                        if long_mode or edit_mode:
                            self._slm_polisher.release()

            elif req_type == 'slm_prewarm':
                self._slm_polisher.prewarm(long_mode=True)
                response = {"success": True}

            elif req_type == 'slm_release':
                self._slm_polisher.release()
                response = {"success": True}

            elif req_type == 'key_event':
                # Rime 按键处理
                keyval = request.get('keyval')
                mask = request.get('mask', 0)
                if keyval is None:
                    response = {"handled": False, "error": "缺少 keyval 参数"}
                else:
                    with self._rime_lock:
                        result = self.rime_handler.process_key(keyval, mask)
                    response = result

            elif req_type == 'reset':
                # 重置 Rime
                with self._rime_lock:
                    self.rime_handler.reset()
                response = {"success": True}

            elif req_type == 'ping':
                # 健康检查
                response = {"pong": True}

            else:
                response = {"error": f"未知的请求类型: {req_type}"}

            # 发送响应
            response_str = json.dumps(response, ensure_ascii=False)
            conn.sendall(response_str.encode('utf-8'))

            logger.debug("已发送响应: %d 字节", len(response_str))

        except json.JSONDecodeError as exc:
            logger.error("JSON 解析失败: %s", exc)
            try:
                error_response = json.dumps({"error": "Invalid JSON"})
                conn.sendall(error_response.encode('utf-8'))
            except Exception:
                pass

        except socket.timeout:
            logger.warning("IPC 请求读取超时")
            try:
                error_response = json.dumps({"error": "Request timeout"})
                conn.sendall(error_response.encode('utf-8'))
            except Exception:
                pass

        except Exception as exc:
            logger.error("处理请求失败: %s", exc)
            import traceback
            traceback.print_exc()
            try:
                error_response = json.dumps({"error": str(exc)})
                conn.sendall(error_response.encode('utf-8'))
            except Exception:
                pass

        finally:
            conn.close()

    def cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")
        try:
            self.asr_server.cleanup()
            self.rime_handler.cleanup()
        except Exception as exc:
            logger.error("清理资源失败: %s", exc)


def main():
    """主入口"""
    global SOCKET_PATH
    import argparse

    parser = argparse.ArgumentParser(
        description='VoCoType Fcitx5 Backend Server'
    )
    parser.add_argument(
        '--socket',
        default=SOCKET_PATH,
        help=f'Unix socket path (default: {SOCKET_PATH})'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    config, config_path = load_backend_config()
    configure_logging(config, args.debug)
    logger.info("配置文件路径: %s", config_path)

    SOCKET_PATH = args.socket

    backend = Fcitx5Backend(config=config)
    try:
        backend.run()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出...")
    finally:
        backend.cleanup()


if __name__ == '__main__':
    main()
