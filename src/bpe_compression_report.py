"""Reproduce the BPE vs. GPT compression-ratio experiments and figures.

Run from the repository root:

    python src/bpe_compression_report.py

The script learns a from-scratch BPE tokenizer for each language, compares it
against the GPT-2/3.5/4 tokenizers via ``tiktoken``, and regenerates every
figure in ``figures/``.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken

# Paths are resolved relative to the repository root so the script works
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DATA_FILES = {
    "English": DATA_DIR / "English Data.txt",
    "German": DATA_DIR / "German Data.txt",
    "Spanish": DATA_DIR / "Spanish Data.txt",
    "French": DATA_DIR / "French Data.txt",
}

PLOT_FILES = {
    "bpe_lang": FIGURES_DIR / "01_bpe_compression_by_language.png",
    "bpe_vs_gpt": FIGURES_DIR / "02_bpe_vs_gpt_tokenizers.png",
    "vocab_vs_ratio": FIGURES_DIR / "03_vocab_size_vs_compression.png",
    "size_vs_ratio": FIGURES_DIR / "04_file_size_vs_compression.png",
}


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_vocab(words: list[str]) -> Counter:
    vocab = Counter()
    for w in words:
        vocab[" ".join(list(w) + ["</w>"])] += 1
    return vocab


def get_stats(vocab: Counter) -> Counter:
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple[str, str], vocab: Counter) -> Counter:
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    return Counter({
        pattern.sub("".join(pair), word): freq
        for word, freq in vocab.items()
    })


def build_initial_token_vocab(text: str) -> dict[str, int]:
    words = re.findall(r"\S+", text)

    symbols = set()
    for w in words:
        symbols.update(list(w))

    symbols.add("</w>")

    vocab_list = sorted(symbols)
    return {tok: i for i, tok in enumerate(vocab_list)}


def build_final_vocab(initial_vocab: dict[str, int], merges: list[tuple[str, str]]) -> dict[str, int]:
    vocab = set(initial_vocab.keys())
    for pair in merges:
        vocab.add("".join(pair))
    vocab_list = sorted(vocab)
    return {tok: i for i, tok in enumerate(vocab_list)}


def learn_bpe(text: str, extra_tokens: int, verbose: bool = False) -> tuple[list[tuple[str, str]], int]:
    words = re.findall(r"\S+", text)
    vocab = build_vocab(words)

    symbols = set("".join(words))
    symbols.add("</w>")
    target_size = len(symbols) + extra_tokens

    merges: list[tuple[str, str]] = []
    while len(symbols) < target_size:
        pairs = get_stats(vocab)
        if not pairs:
            break

        best = pairs.most_common(1)[0][0]
        vocab = merge_vocab(best, vocab)

        new_symbol = "".join(best)
        symbols.add(new_symbol)
        merges.append(best)

        if verbose and len(merges) % 50 == 0:
            print(f"merges={len(merges)}, symbols={len(symbols)}/{target_size}")

    return merges, len(symbols)


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def bpe_encode_word(word: list[str], ranks: dict[tuple[str, str], int]) -> tuple[str, ...]:
    word_tuple = tuple(word)

    while True:
        pairs = get_pairs(word_tuple)
        if not pairs:
            break

        bigram = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if bigram not in ranks:
            break

        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == bigram:
                new_word.append(word_tuple[i] + word_tuple[i + 1])
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1

        word_tuple = tuple(new_word)
        if len(word_tuple) == 1:
            break

    return word_tuple


def bpe_token_count(text: str, merges: list[tuple[str, str]]) -> int:
    ranks = {m: i for i, m in enumerate(merges)}
    count = 0
    for w in re.findall(r"\S+", text):
        count += len(bpe_encode_word(list(w) + ["</w>"], ranks))
    return count


def compression_ratio(text: str, token_count: int) -> float:
    return len(text) / max(token_count, 1)


def tiktoken_ratio(text: str, model: str) -> float:
    enc = tiktoken.encoding_for_model(model)
    return compression_ratio(text, len(enc.encode(text)))


def main() -> None:
    texts = {lang: read_text(path) for lang, path in DATA_FILES.items()}

    # Task 1: BPE compression ratio per language
    extra_tokens = 200
    results = []
    for lang, text in texts.items():
        initial_vocab = build_initial_token_vocab(text)
        merges, _ = learn_bpe(text, extra_tokens, verbose=False)
        final_vocab = build_final_vocab(initial_vocab, merges)
        token_count = bpe_token_count(text, merges)
        results.append({
            "Language": lang,
            "Original Vocab Size": len(initial_vocab),
            "Final Vocab Size": len(final_vocab),
            "BPE Tokens": token_count,
            "Compression Ratio": compression_ratio(text, token_count),
        })

    bpe_df = pd.DataFrame(results).sort_values("Language").reset_index(drop=True)
    print("\nBPE results:\n", bpe_df)

    # Task 2: Bar plot for BPE compression ratio across languages
    plt.figure()
    plt.bar(bpe_df["Language"], bpe_df["Compression Ratio"])
    plt.xlabel("Language")
    plt.ylabel("Compression Ratio")
    plt.title("BPE Compression Ratio Across Languages")
    plt.tight_layout()
    plt.savefig(PLOT_FILES["bpe_lang"])
    plt.close()

    # Task 3: BPE vs GPT tokenizers
    model_map = {
        "GPT-2": "gpt2",
        "GPT-3.5": "gpt-3.5-turbo",
        "GPT-4": "gpt-4",
    }
    rows = []
    for lang, text in texts.items():
        for name, model in model_map.items():
            rows.append({
                "Language": lang,
                "Tokenizer": name,
                "Compression Ratio": tiktoken_ratio(text, model),
            })

    bpe_plot = bpe_df[["Language", "Compression Ratio"]].assign(Tokenizer="BPE")
    comp_df = pd.concat([bpe_plot, pd.DataFrame(rows)], ignore_index=True)
    comp_pivot = (
        comp_df
        .pivot(index="Language", columns="Tokenizer", values="Compression Ratio")
        .loc[["English", "French", "German", "Spanish"]]
    )
    print("\nBPE vs GPT compression ratios:\n", comp_pivot)

    comp_pivot.plot(kind="bar")
    plt.xlabel("Language")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio: BPE vs GPT Tokenizers")
    plt.tight_layout()
    plt.savefig(PLOT_FILES["bpe_vs_gpt"])
    plt.close()

    # Task 4: Vocabulary size vs compression ratio (BPE)
    extra_tokens_list = [200, 500, 800]
    rows = []
    for lang, text in texts.items():
        for extra in extra_tokens_list:
            merges, vocab_size = learn_bpe(text, extra)
            rows.append({
                "Language": lang,
                "Final Vocab Size": vocab_size,
                "Compression Ratio": compression_ratio(text, bpe_token_count(text, merges)),
            })

    vocab_df = pd.DataFrame(rows)
    print("\nVocab size vs compression ratio:\n", vocab_df)

    for lang in vocab_df["Language"].unique():
        d = vocab_df[vocab_df["Language"] == lang]
        plt.plot(d["Final Vocab Size"], d["Compression Ratio"], marker="o", label=lang)

    plt.xlabel("Final Vocabulary Size")
    plt.ylabel("Compression Ratio")
    plt.title("Vocabulary Size vs Compression Ratio (BPE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILES["vocab_vs_ratio"])
    plt.close()

    # Task 5: File size vs compression ratio (English, BPE)
    english = texts["English"]
    factors = [10, 8, 6]
    rows = []

    for f in factors:
        text = english[: len(english) // f]
        extra = int(0.05 * len(text))
        merges, vocab_size = learn_bpe(text, extra)

        rows.append({
            "Text Length": len(text),
            "Final Vocab Size": vocab_size,
            "Compression Ratio": compression_ratio(text, bpe_token_count(text, merges)),
        })

    size_df = pd.DataFrame(rows).sort_values("Text Length")
    print("\nFile size vs compression ratio:\n", size_df)

    plt.plot(size_df["Text Length"], size_df["Compression Ratio"], marker="o")
    plt.xlabel("File Size (Characters)")
    plt.ylabel("Compression Ratio")
    plt.title("File Size vs Compression Ratio (English, BPE)")
    plt.tight_layout()
    plt.savefig(PLOT_FILES["size_vs_ratio"])
    plt.close()


if __name__ == "__main__":
    main()
