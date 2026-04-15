#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import re
from typing import List

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_tokens(text: str) -> List[str]:
    return text.strip().split()


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


def aug_random_deletion(text: str, p: float = 0.08) -> str:
    tokens = split_tokens(text)
    if len(tokens) <= 3:
        return text
    kept = [token for token in tokens if random.random() > p]
    if len(kept) < 2:
        return text
    return join_tokens(kept)


def aug_random_swap(text: str, n_swaps: int = 1) -> str:
    tokens = split_tokens(text)
    if len(tokens) < 3:
        return text
    tokens = tokens[:]
    for _ in range(n_swaps):
        i, j = random.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return join_tokens(tokens)


def aug_char_noise(text: str, p: float = 0.02) -> str:
    chars = list(text)
    out = []
    for ch in chars:
        rnd = random.random()
        if rnd < p / 4:
            continue
        if rnd < p / 2:
            out.extend([ch, ch])
            continue
        if rnd < p:
            out.append(random.choice([" ", ".", ","]))
            continue
        out.append(ch)
    result = re.sub(r"\s{2,}", " ", "".join(out)).strip()
    return result or text


def augment_text(text: str, methods: List[str], max_try_per_method: int = 1) -> List[str]:
    outputs = []
    for method in methods:
        for _ in range(max_try_per_method):
            if method == "del":
                augmented = aug_random_deletion(text)
            elif method == "swap":
                augmented = aug_random_swap(text)
            elif method == "noise":
                augmented = aug_char_noise(text)
            else:
                raise ValueError(f"Unknown method: {method}")
            if augmented and augmented != text:
                outputs.append(augmented)

    unique = []
    seen = set()
    for item in outputs:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
