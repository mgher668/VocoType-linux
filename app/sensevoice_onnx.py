"""SenseVoice ONNX runtime wrapper compatible with ModelScope ONNX exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import librosa
import numpy as np

from funasr_onnx.utils.frontend import WavFrontend
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr_onnx.utils.utils import OrtInferSession, read_yaml


class TokenListTokenizer:
    """Decode token IDs from SenseVoice ONNX repos that ship tokens.json only."""

    def __init__(self, token_file: Union[str, Path]):
        with open(token_file, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        if not isinstance(tokens, list):
            raise ValueError(f"tokens.json must contain a token list: {token_file}")
        self.tokens = [str(token) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces = []
        for token_id in token_ids:
            idx = int(token_id)
            if 0 <= idx < len(self.tokens):
                pieces.append(self.tokens[idx])

        text = "".join(pieces)
        # tokens.json uses sentencepiece-style "▁" as a word-boundary marker.
        text = text.replace("▁", " ")
        return text.strip()


class SenseVoiceOnnx:
    """Small wrapper mirroring funasr_onnx.SenseVoiceSmall with token-list fallback."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
    ):
        model_dir = Path(model_dir)
        model_file = model_dir / ("model_quant.onnx" if quantize else "model.onnx")
        if not model_file.exists() and not quantize:
            model_file = model_dir / "model_quant.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"SenseVoice ONNX model not found in {model_dir}")

        config = read_yaml(str(model_dir / "config.yaml"))
        cmvn_file = model_dir / "am.mvn"
        bpe_model = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        token_file = model_dir / "tokens.json"
        if bpe_model.exists():
            self.tokenizer = SentencepiecesTokenizer(bpemodel=str(bpe_model))
        elif token_file.exists():
            self.tokenizer = TokenListTokenizer(token_file)
        else:
            raise FileNotFoundError(
                f"SenseVoice tokenizer not found: expected {bpe_model.name} or {token_file.name}"
            )

        config["frontend_conf"]["cmvn_file"] = str(cmvn_file)
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            str(model_file),
            device_id,
            intra_op_num_threads=intra_op_num_threads,
        )
        self.batch_size = batch_size
        self.blank_id = 0
        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }
        self.textnorm_dict = {"withitn": 14, "woitn": 15}

    def _get_lid(self, lid: str) -> int:
        if lid not in self.lid_dict:
            raise ValueError(f"The language {lid} is not in {list(self.lid_dict.keys())}")
        return self.lid_dict[lid]

    def _get_tnid(self, tnid: str) -> int:
        if tnid not in self.textnorm_dict:
            raise ValueError(f"The textnorm {tnid} is not in {list(self.textnorm_dict.keys())}")
        return self.textnorm_dict[tnid]

    def read_tags(self, language_input, textnorm_input):
        if isinstance(language_input, list):
            language_list = [self._get_lid(l) for l in language_input]
        elif isinstance(language_input, str):
            if os.path.exists(language_input):
                with open(language_input, "r", encoding="utf-8") as f:
                    language_list = [self._get_lid(line.strip()) for line in f]
            else:
                language_list = [self._get_lid(language_input)]
        else:
            raise ValueError(f"Unsupported type {type(language_input)} for language_input")

        if isinstance(textnorm_input, list):
            textnorm_list = [self._get_tnid(tn) for tn in textnorm_input]
        elif isinstance(textnorm_input, str):
            if os.path.exists(textnorm_input):
                with open(textnorm_input, "r", encoding="utf-8") as f:
                    textnorm_list = [self._get_tnid(line.strip()) for line in f]
            else:
                textnorm_list = [self._get_tnid(textnorm_input)]
        else:
            raise ValueError(f"Unsupported type {type(textnorm_input)} for textnorm_input")
        return language_list, textnorm_list

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs):
        language_list, textnorm_list = self.read_tags(
            kwargs.get("language", "auto"),
            kwargs.get("textnorm", "woitn"),
        )
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)

        if len(language_list) not in (1, waveform_nums):
            raise ValueError("language list length must be 1 or equal to waveform count")
        if len(textnorm_list) not in (1, waveform_nums):
            raise ValueError("textnorm list length must be 1 or equal to waveform count")

        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            lang = language_list[beg_idx:end_idx]
            textnorm = textnorm_list[beg_idx:end_idx]
            batch_size = feats.shape[0]
            if len(lang) == 1 and batch_size != 1:
                lang = lang * batch_size
            if len(textnorm) == 1 and batch_size != 1:
                textnorm = textnorm * batch_size

            ctc_logits, encoder_out_lens = self.infer(
                feats,
                feats_len,
                np.array(lang, dtype=np.int32),
                np.array(textnorm, dtype=np.int32),
            )
            for idx in range(batch_size):
                length = int(encoder_out_lens[idx])
                yseq = np.argmax(ctc_logits[idx, :length, :], axis=-1)
                if yseq.size:
                    keep = np.concatenate(([True], yseq[1:] != yseq[:-1]))
                    yseq = yseq[keep]
                token_int = yseq[yseq != self.blank_id].tolist()
                asr_res.append(self.tokenizer.decode(token_int))

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]
        if isinstance(wav_content, str):
            return [load_wav(wav_content)]
        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]
        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        feat_res = []
        for feat in feats:
            pad_width = ((0, max_feat_len - feat.shape[0]), (0, 0))
            feat_res.append(np.pad(feat, pad_width, "constant", constant_values=0))
        return np.array(feat_res).astype(np.float32)

    def infer(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.ort_infer([feats, feats_len, language, textnorm])
