import json
import random
from importlib.resources import files
import os
from pathlib import Path

import jieba
import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from pypinyin import Style, lazy_pinyin
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default

TARGET_SAMPLE_RATE = 24_000
HOP_LENGTH = 256
PUNCTUATION_CHARS = set("，。？！；：、,.!?;:()[]{}<>\"'`“”‘’")


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    zh_quote_trans = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    custom_trans = str.maketrans({";": ","})
    for text in text_list:
        char_list = []
        text = text.translate(zh_quote_trans)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for item in seg:
                    if item not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(item)
            else:
                for char in seg:
                    if ord(char) < 256:
                        char_list.extend(char)
                    else:
                        if char not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(char, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(char)
        final_text_list.append(char_list)
    return final_text_list


def tokenize_word_for_dataset(word_text: str):
    if contains_cjk(word_text):
        tokens = convert_char_to_pinyin([word_text], polyphone=True)[0]
        return [str(token) for token in tokens if str(token).strip() and str(token) not in PUNCTUATION_CHARS]
    return [ch.lower() for ch in str(word_text) if ch.isalnum()]


def load_sidecar_words(sidecar_path: Path):
    with sidecar_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    words = []
    for segment in obj.get("segments", []):
        for word_info in segment.get("words", []):
            word = str(word_info.get("word", "")).strip()
            if not word:
                continue
            start = float(word_info.get("start", 0.0) or 0.0)
            end = float(word_info.get("end", start) or start)
            if end <= start:
                continue
            words.append({"word": word, "start": start, "end": end})
    return words


def choose_boundary_word_index(words, total_duration_sec, prompt_frac, min_prompt_sec, min_target_sec):
    if len(words) < 2:
        return None
    target_prompt_sec = total_duration_sec * prompt_frac
    best_index = None
    best_score = None
    for boundary_word_index in range(1, len(words)):
        prompt_sec = float(words[boundary_word_index]["start"])
        target_sec = total_duration_sec - prompt_sec
        if prompt_sec < min_prompt_sec or target_sec < min_target_sec:
            continue
        score = abs(prompt_sec - target_prompt_sec)
        if best_score is None or score < best_score:
            best_score = score
            best_index = boundary_word_index
    return best_index


def infer_sidecar_path(audio_path: str) -> Path | None:
    sidecar_root = Path(
        os.environ.get(
            "F5TTS_FT_PROMPT_BOUNDARY_SIDECAR_ROOT",
            str(Path(__file__).resolve().parents[2] / "data" / "prompt_sidecars"),
        )
    )
    audio = Path(audio_path)
    if len(audio.parts) < 3:
        return None
    candidate = sidecar_root / audio.parts[-3] / audio.parts[-2] / f"{audio.stem}.words.json"
    if candidate.exists():
        return candidate
    return None


def compute_prompt_boundary_index(audio_path: str, text_tokens, token_durations, duration_sec: float) -> int:
    prompt_frac_min = float(os.environ.get("F5TTS_FT_PROMPT_FRAC_MIN", "0.3"))
    prompt_frac_max = float(os.environ.get("F5TTS_FT_PROMPT_FRAC_MAX", str(prompt_frac_min)))
    min_prompt_sec = float(os.environ.get("F5TTS_FT_PROMPT_MIN_SEC", "0.5"))
    min_target_sec = float(os.environ.get("F5TTS_FT_TARGET_MIN_SEC", "0.5"))
    prompt_frac = random.uniform(prompt_frac_min, prompt_frac_max)
    if token_durations is None:
        raise RuntimeError(f"missing token durations for audio: {audio_path}")

    normalized_text = [str(token).lower() for token in text_tokens]
    if len(token_durations) != len(normalized_text):
        raise RuntimeError(
            f"text/token_duration length mismatch for audio: {audio_path}; "
            f"text_len={len(normalized_text)}, duration_len={len(token_durations)}"
        )
    if len(normalized_text) < 2:
        raise RuntimeError(f"not enough tokens for prompt boundary on audio: {audio_path}")

    target_prompt_sec = duration_sec * prompt_frac
    cumulative_sec = 0.0
    best_index = None
    best_score = None
    frame_to_sec = HOP_LENGTH / TARGET_SAMPLE_RATE

    for boundary_index in range(1, len(token_durations)):
        content_frames, pause_frames = token_durations[boundary_index - 1]
        cumulative_sec += max(0.0, float(content_frames)) * frame_to_sec
        cumulative_sec += max(0.0, float(pause_frames)) * frame_to_sec
        prompt_sec = cumulative_sec
        target_sec = duration_sec - prompt_sec
        if prompt_sec < min_prompt_sec or target_sec < min_target_sec:
            continue
        score = abs(prompt_sec - target_prompt_sec)
        if best_score is None or score < best_score:
            best_score = score
            best_index = boundary_index

    if best_index is None:
        raise RuntimeError(f"cannot find valid token-duration boundary for audio: {audio_path}")
    return best_index


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        dataset_root: str | None = None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.enable_prompt_boundary = any(
            os.environ.get(env_name, "0") not in {"0", "0.0", ""}
            for env_name in (
                "F5TTS_FT_PROMPT_DURATION_MASK_PROB",
                "F5TTS_FT_PROMPT_TEXT_MASK_PROB",
                "F5TTS_FT_PROMPT_TEXT_TARGET_ONLY_PROB",
            )
        )

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        if self.dataset_root is not None and not os.path.isabs(audio_path):
            audio_path = str((self.dataset_root / audio_path).resolve())
        text = row["text"]
        raw_text = row["raw_text"] if "raw_text" in row else None
        duration = row["duration"]
        token_durations = row.get("token_durations") if hasattr(row, "get") else None

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])

        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if duration > 15 or duration < 1.0:
                return self.__getitem__((index + 1) % len(self.data))

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

        prompt_boundary_index = row.get("prompt_boundary_index") if hasattr(row, "get") else None
        if prompt_boundary_index is not None:
            prompt_boundary_index = int(prompt_boundary_index)
        elif self.enable_prompt_boundary and token_durations is not None:
            try:
                prompt_boundary_index = compute_prompt_boundary_index(
                    audio_path=audio_path,
                    text_tokens=text,
                    token_durations=token_durations,
                    duration_sec=duration,
                )
            except FileNotFoundError:
                raise
            except RuntimeError:
                return self.__getitem__((index + 1) % len(self.data))

        return dict(
            mel_spec=mel_spec,
            text=text,
            raw_text=raw_text,
            audio_path=audio_path,
            token_durations=token_durations,
            prompt_boundary_index=prompt_boundary_index,
        )


# Dynamic Batch Sampler


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False,
        repeat_count=1, mini_repeat_count=1, drop_residual=False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.repeat_count = repeat_count
        self.mini_repeat_count = mini_repeat_count
        self.drop_residual = drop_residual

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        # repeat
        self.batches = []
        for chunk in batches:
            for _ in range(self.repeat_count):
                batch_sub = []
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        batch_sub.append(index)
                self.batches.append(batch_sub)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            dataset_root=f"{dataset_name}/raw",
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        preprocessed_mel = False
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            dataset_root=f"{dataset_name}/raw",
            preprocessed_mel=preprocessed_mel,
            **mel_spec_kwargs,
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    raw_text = [item.get("raw_text") for item in batch]
    audio_path = [item.get("audio_path") for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    token_durations = None
    token_duration_mask = None
    prompt_boundary_index = None
    if any(item.get("token_durations") is not None for item in batch):
        max_text_length = max(len(item) for item in text)
        padded_token_durations = []
        padded_token_duration_masks = []
        for item in batch:
            values = item.get("token_durations")
            if values is None:
                cur = torch.zeros(max_text_length, 2, dtype=torch.float32)
                cur_mask = torch.zeros(max_text_length, 2, dtype=torch.float32)
            else:
                cur = torch.tensor(values, dtype=torch.float32)
                if cur.ndim == 1:
                    cur = cur.unsqueeze(-1)
                if cur.shape[-1] == 1:
                    cur = torch.cat([cur, torch.zeros_like(cur)], dim=-1)
                elif cur.shape[-1] > 2:
                    cur = cur[:, :2]
                if cur.shape[0] > max_text_length:
                    cur = cur[:max_text_length]
                elif cur.shape[0] < max_text_length:
                    cur = F.pad(cur, (0, 0, 0, max_text_length - cur.shape[0]), value=0.0)
                cur_mask = torch.zeros(max_text_length, 2, dtype=torch.float32)
                cur_mask[: min(len(values), max_text_length), :] = 1.0
            padded_token_durations.append(cur)
            padded_token_duration_masks.append(cur_mask)
        token_durations = torch.stack(padded_token_durations, dim=0)
        token_duration_mask = torch.stack(padded_token_duration_masks, dim=0)
    if any(item.get("prompt_boundary_index") is not None for item in batch):
        prompt_boundary_index = torch.LongTensor(
            [
                int(item.get("prompt_boundary_index", 0) or 0)
                for item in batch
            ]
        )

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        raw_text=raw_text,
        audio_path=audio_path,
        text_lengths=text_lengths,
        token_durations=token_durations,
        token_duration_mask=token_duration_mask,
        prompt_boundary_index=prompt_boundary_index,
    )
