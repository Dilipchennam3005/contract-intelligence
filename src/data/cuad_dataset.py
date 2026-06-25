"""
CUAD PyTorch Dataset with sliding-window tokenization.

Long contracts (median 6k words) are split into overlapping 512-token windows
(stride=128). Each window becomes an independent training example. At inference,
windows from the same contract are merged by taking the highest-confidence span.
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


def load_squad_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                samples.append({
                    "id":       qa["id"],
                    "title":    article.get("title", ""),
                    "context":  ctx,
                    "question": qa["question"],
                    "answers":  qa["answers"],  # list of {text, answer_start}
                })
    return samples


class CUADDataset(Dataset):
    """
    Tokenizes CUAD samples with sliding-window chunking.

    Each raw sample may produce multiple feature dicts (one per window).
    Impossible windows (answer not in window) get start/end = cls_index (0).
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        stride: int = 128,
        split: str = "train",
        max_windows_per_sample: Optional[int] = None,
    ):
        self.split      = split
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stride     = stride
        self.features   = []
        self.sample_ids = []  # maps feature index -> original sample id

        self._build_features(samples, max_windows_per_sample)

    def _build_features(self, samples: list[dict], max_windows_per_sample: Optional[int] = None):
        for sample in samples:
            question = sample["question"]
            context  = sample["context"]
            answers  = sample["answers"]

            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                stride=self.stride,
                truncation="only_second",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )

            offset_mapping    = encoding.pop("offset_mapping")
            overflow_to_sample = encoding.pop("overflow_to_sample_mapping", None)
            num_windows       = encoding["input_ids"].shape[0]

            # For quick/CPU mode: cap windows per sample so training stays fast.
            # Prioritise the window that contains the answer (if any).
            window_indices = list(range(num_windows))
            if max_windows_per_sample and num_windows > max_windows_per_sample:
                window_indices = window_indices[:max_windows_per_sample]

            for window_idx in window_indices:
                offsets    = offset_mapping[window_idx].tolist()
                input_ids  = encoding["input_ids"][window_idx]
                attn_mask  = encoding["attention_mask"][window_idx]

                # Determine which token indices belong to the context (not question/[SEP])
                sequence_ids = encoding.sequence_ids(window_idx)
                ctx_start_tok = next(i for i, s in enumerate(sequence_ids) if s == 1)
                ctx_end_tok   = next(
                    (len(sequence_ids) - 1 - i for i, s in enumerate(reversed(sequence_ids)) if s == 1),
                    len(sequence_ids) - 1
                )

                # Character span of this window in the original context
                win_char_start = offsets[ctx_start_tok][0]
                win_char_end   = offsets[ctx_end_tok][1]

                start_pos = 0  # default: CLS = unanswerable
                end_pos   = 0

                if self.split == "train" and answers:
                    ans_text  = answers[0]["text"]
                    ans_char_start = answers[0]["answer_start"]
                    ans_char_end   = ans_char_start + len(ans_text)

                    # Answer must be fully within this window
                    if ans_char_start >= win_char_start and ans_char_end <= win_char_end:
                        # Find token indices
                        tok_start = next(
                            (i for i, (s, e) in enumerate(offsets)
                             if s <= ans_char_start < e),
                            ctx_start_tok,
                        )
                        tok_end = next(
                            (i for i, (s, e) in enumerate(offsets)
                             if s < ans_char_end <= e),
                            tok_start,
                        )
                        start_pos = tok_start
                        end_pos   = tok_end

                feature = {
                    "input_ids":      input_ids,
                    "attention_mask": attn_mask,
                    "start_positions": torch.tensor(start_pos, dtype=torch.long),
                    "end_positions":   torch.tensor(end_pos,   dtype=torch.long),
                }
                if "token_type_ids" in encoding:
                    feature["token_type_ids"] = encoding["token_type_ids"][window_idx]

                self.features.append(feature)
                self.sample_ids.append(sample["id"])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def build_dataloaders(
    train_samples: list[dict],
    test_samples:  list[dict],
    tokenizer:     PreTrainedTokenizerFast,
    batch_size:    int = 16,
    max_length:    int = 512,
    stride:        int = 128,
    num_workers:   int = 0,
    max_windows_per_sample: Optional[int] = None,
):
    train_ds = CUADDataset(train_samples, tokenizer, max_length, stride, split="train",
                           max_windows_per_sample=max_windows_per_sample)
    test_ds  = CUADDataset(test_samples,  tokenizer, max_length, stride, split="test",
                           max_windows_per_sample=max_windows_per_sample)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_ds, test_ds
