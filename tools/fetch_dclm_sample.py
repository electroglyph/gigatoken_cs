#!/usr/bin/env python3
"""Fetch a curated DCLM sample: edge-case-heavy documents under byte quotas.

Mirrors the selection rules of upstream gigatoken's tests/dclm_fixture.py
(same shard, same revision, same category quotas and classifiers), but writes
the selected documents to a local file instead of keeping them in memory, so
the C# hostile-parity sweep can consume them without Python:

  uv run --with huggingface_hub --with zstandard \\
      python tools/fetch_dclm_sample.py --out /tmp/dclm_sample.jsonl

Each output line is {"category": ..., "text": ...}. Every machine downloads
the same pinned shard and applies the same deterministic rules, so the
selection is identical everywhere.

For offline smoke tests, --shard accepts a local .jsonl.zst file and
--max-docs caps the scan (selection itself needs no network).
"""

import argparse
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

HF_REPO = "mlfoundations/dclm-baseline-1.0"
HF_REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
HF_SHARD = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"

TARGET_BYTES = 20_000_000  # total selection size (UTF-8 bytes of text)
MAX_DOC_BYTES = 262_144  # skip anything larger outright

_CJK_RE = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002a6df\U0002a600-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceB0-\U0002ebef\U00030000-\U0003134f\U00031350-\U000323af]")
_OTHER_SCRIPT_RE = re.compile("[\u0370-\u03ff\u0400-\u04ff\u0500-\u052f\u0530-\u058f\u0590-\u05ff\u0600-\u06ff\u0700-\u074f\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
_EMOJI_RE = re.compile("[←-\u21ff\u2600-\u27bf\u2b00-\u2bff\U0001f000-\U0001faff]")
_COMBINING_RE = re.compile("[̀-ͯ᪰-᫿⃐-⃿︠-︯]")
_ODD_WS_RE = re.compile("[\t\r\x0b\f\x85\xa0​-‍  　﻿]| {8,}|\n{4,}")
_LONG_TOKEN_RE = re.compile(r"\S{80,}")
_CODE_MARK_RE = re.compile(r"[{};]|\n[ \t]+\S|</\w|&\w+;")
_DIGIT_RE = re.compile(r"[0-9]")


def _count(pattern: re.Pattern, text: str, at_least: int) -> bool:
    n = 0
    for _ in pattern.finditer(text):
        n += 1
        if n >= at_least:
            return True
    return False


# (category, byte quota). A document lands in the first category it matches
# that still has quota left; order therefore puts the rarest classes first.
QUOTAS: list[tuple[str, int]] = [
    ("nfc_divergent", 1_500_000),
    ("other_script", 2_000_000),
    ("cjk", 2_000_000),
    ("combining", 1_000_000),
    ("emoji", 1_500_000),
    ("odd_whitespace", 2_000_000),
    ("long_token", 1_500_000),
    ("code", 2_500_000),
    ("digit_heavy", 1_500_000),
    ("huge", 1_500_000),
    ("tiny", 500_000),
    ("general", 2_500_000),
]


def classify(text: str, nbytes: int) -> list[str]:
    """All categories a document qualifies for, in QUOTAS priority order."""
    cats = []
    if not text.isascii():
        if not unicodedata.is_normalized("NFC", text):
            cats.append("nfc_divergent")
        if _count(_OTHER_SCRIPT_RE, text, 50):
            cats.append("other_script")
        if _count(_CJK_RE, text, 50):
            cats.append("cjk")
        if _count(_COMBINING_RE, text, 10):
            cats.append("combining")
        if _count(_EMOJI_RE, text, 5):
            cats.append("emoji")
    if _count(_ODD_WS_RE, text, 5):
        cats.append("odd_whitespace")
    if _LONG_TOKEN_RE.search(text):
        cats.append("long_token")
    if _count(_CODE_MARK_RE, text, max(20, nbytes // 200)):
        cats.append("code")
    if len(_DIGIT_RE.findall(text)) * 6 > len(text):
        cats.append("digit_heavy")
    if nbytes >= 131_072:
        cats.append("huge")
    if nbytes < 200:
        cats.append("tiny")
    cats.append("general")
    return cats


def select(texts, target_bytes: int = TARGET_BYTES) -> list[tuple[str, str]]:
    """Curate (category, text) from an iterable of document texts."""
    remaining = dict(QUOTAS)
    selected: list[tuple[str, str]] = []
    spill: list[str] = []
    spill_bytes = 0
    seen: set[bytes] = set()
    total = 0

    for text in texts:
        if not text:
            continue
        raw = text.encode("utf-8")
        nbytes = len(raw)
        if total >= target_bytes:
            break
        if nbytes > MAX_DOC_BYTES:
            continue
        h = hashlib.blake2b(raw, digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        for cat in classify(text, nbytes):
            if remaining[cat] > 0:
                selected.append((cat, text))
                remaining[cat] -= nbytes
                total += nbytes
                break
        else:
            if spill_bytes < 8_000_000:
                spill.append(text)
                spill_bytes += nbytes

    for text in spill:
        if total >= target_bytes:
            break
        selected.append(("general", text))
        total += len(text.encode("utf-8"))
    return selected


def download_shard() -> Path:
    """Fetch the pinned DCLM shard into the HuggingFace cache (no-op when cached)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(HF_REPO, HF_SHARD, repo_type="dataset", revision=HF_REVISION)
    )


def iter_shard_texts(shard: Path):
    import zstandard

    with open(shard, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                text = json.loads(line).get("text")
                if text:
                    yield text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    ap.add_argument("--max-docs", type=int, default=0,
                    help="stop scanning after N docs (0 = no cap; for smoke tests)")
    ap.add_argument("--shard", default="",
                    help="local .jsonl.zst shard (default: download the pinned shard)")
    args = ap.parse_args()

    shard = Path(args.shard) if args.shard else download_shard()
    texts = iter_shard_texts(shard)
    if args.max_docs:
        texts = (t for _, t in zip(range(args.max_docs), texts))
    selected = select(texts, args.target_bytes)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        for cat, text in selected:
            fh.write(json.dumps({"category": cat, "text": text}, ensure_ascii=False) + "\n")

    total = sum(len(t.encode("utf-8")) for _, t in selected)
    print(f"selected {len(selected)} docs, {total / 1e6:.2f} MB -> {out}")
    by_cat: dict[str, int] = {}
    for cat, _ in selected:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    for cat, _ in QUOTAS:
        if cat in by_cat:
            print(f"  {cat}: {by_cat[cat]} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
