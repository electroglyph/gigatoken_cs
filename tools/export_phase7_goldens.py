#!/usr/bin/env python3
"""Phase 7 parity-sweep exporter (checked in; data + goldens are NOT).

Builds, under --out (default ~/data/gigatoken_cs_phase7):
  owt_docs.bin / dclm_docs.bin            docs: u64 n, then {u64 byte_len, utf8}*
  ids.{gpt2,qwen2_5,tinyllama}.bin        per corpus: u64 n, then {u64 count, u32[count] LE}*
  manifest.json                           tokenizer paths, revisions, counts

Oracles: tiktoken r50k_base (gpt2), HF tokenizers lib (qwen2.5 snapshot,
TinyLlama tokenizer.json download). Docs containing special-token-shaped
substrings are dropped so oracle semantics (tiktoken default encode raises on
specials; HF splits added specials like gigatoken) stay aligned; the drop
count is recorded in the manifest.

DCLM selection mirrors tests/dclm_fixture.py quotas/classifiers exactly.
"""
import argparse
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# --- DCLM selection (mirrors upstream/tests/dclm_fixture.py) ---
DCLM_REPO = "mlfoundations/dclm-baseline-1.0"
DCLM_REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
DCLM_SHARD = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"
DCLM_TARGET = 20_000_000
DCLM_MAX_DOC = 262_144

_CJK_RE = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")
_OTHER_SCRIPT_RE = re.compile(r"[Ͱ-ϿЀ-ӿ԰-֏֐-׿؀-ۿऀ-ॿ฀-๿]")
_EMOJI_RE = re.compile(r"[←-⇿☀-➿⬀-⯿\U0001f000-\U0001faff]")
_COMBINING_RE = re.compile(r"[̀-ͯ᪰-᫿⃐-⃿︠-︯]")
_ODD_WS_RE = re.compile(r"[\t\r\x0b\f\x85\xa0​-‍  　﻿]| {8,}|\n{4,}")
_LONG_TOKEN_RE = re.compile(r"\S{80,}")
_CODE_MARK_RE = re.compile(r"[{};]|\n[ \t]+\S|</\w|&\w+;")
_DIGIT_RE = re.compile(r"[0-9]")


def _count(pat, text, at_least):
    n = 0
    for _ in pat.finditer(text):
        n += 1
        if n >= at_least:
            return True
    return False


QUOTAS = [
    ("nfc_divergent", 1_500_000), ("other_script", 2_000_000),
    ("cjk", 2_000_000), ("combining", 1_000_000), ("emoji", 1_500_000),
    ("odd_whitespace", 2_000_000), ("long_token", 1_500_000),
    ("code", 2_500_000), ("digit_heavy", 1_500_000), ("huge", 1_500_000),
    ("tiny", 500_000), ("general", 2_500_000),
]


def _classify(text, nbytes):
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


_SPECIAL_SHAPE = re.compile(r"<\|[^<>]*\|>|<s>|</s>|<unk>|<pad>|<0x[0-9A-Fa-f]{2}>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/gigatoken_cs_phase7")
    ap.add_argument("--owt-bytes", type=int, default=100_000_000)
    ap.add_argument("--only-dclm", action="store_true",
                    help="reuse existing OWT files; only (re)collect DCLM + its goldens")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    manifest_path = out / "manifest.json"
    if args.only_dclm and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"revisions": {}, "dropped_specials": 0}

    # --- tokenizer assets ---
    qwen_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    assert (qwen_dir / "tokenizer.json").exists(), f"missing {qwen_dir}"
    tiny_path = hf_hub_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.json")
    api = HfApi()
    manifest["revisions"]["tinyllama"] = api.model_info("TinyLlama/TinyLlama-1.1B-Chat-v1.0").sha
    manifest["tokenizers"] = {
        "gpt2": {"kind": "bpe-r50k", "oracle": "tiktoken r50k_base"},
        "qwen2_5": {"kind": "bpe", "path": str(qwen_dir), "oracle": "HF tokenizers"},
        "tinyllama": {"kind": "sentencepiece", "path": tiny_path, "oracle": "HF tokenizers"},
    }

    # --- OWT subset (streaming, sequential = deterministic per revision) ---
    import datasets
    owt_docs, owt_bytes = [], 0
    if args.only_dclm:
        print("owt: reused from previous run", flush=True)
    else:
        manifest["revisions"]["openwebtext"] = api.dataset_info("Skylion007/openwebtext").sha
        ds = datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        owt_bytes, seen = 0, set()
        for row in ds:
            t = row["text"]
            if not t:
                continue
            raw = t.encode("utf-8")
            if len(raw) > DCLM_MAX_DOC:
                continue
            if _SPECIAL_SHAPE.search(t):
                manifest["dropped_specials"] += 1
                continue
            h = hashlib.blake2b(raw, digest_size=16).digest()
            if h in seen:
                continue
            seen.add(h)
            owt_docs.append(t)
            owt_bytes += len(raw)
            if owt_bytes >= args.owt_bytes:
                break
        print(f"owt: {len(owt_docs)} docs, {owt_bytes/1e6:.1f} MB", flush=True)

    # --- DCLM hostile sample (pinned shard -> HF cache, zstd stream) ---
    # NOTE: hf_hub_download returns a symlink into blobs/; the `zstd` CLI
    # refuses symlinks, so resolve to the real blob first.
    shard = Path(hf_hub_download(DCLM_REPO, DCLM_SHARD, repo_type="dataset", revision=DCLM_REVISION)).resolve()
    proc = subprocess.Popen(["zstd", "-d", "-c", str(shard)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    remaining = dict(QUOTAS)
    selected, spill, spill_bytes, seen2 = [], [], 0, set()
    total = scanned = 0
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text")
        except ValueError:
            continue
        if not text:
            continue
        raw = text.encode("utf-8")
        nbytes = len(raw)
        scanned += nbytes
        if total >= DCLM_TARGET:
            break
        if nbytes > DCLM_MAX_DOC or not text.strip():
            continue
        if _SPECIAL_SHAPE.search(text):
            manifest["dropped_specials"] += 1
            continue
        h = hashlib.blake2b(raw, digest_size=16).digest()
        if h in seen2:
            continue
        seen2.add(h)
        for cat in _classify(text, nbytes):
            if remaining[cat] > 0:
                selected.append(text)
                remaining[cat] -= nbytes
                total += nbytes
                break
        else:
            if spill_bytes < 8_000_000:
                spill.append(text)
                spill_bytes += nbytes
    proc.wait()
    for text in spill:
        if total >= DCLM_TARGET:
            break
        selected.append(text)
        total += len(text.encode("utf-8"))
    manifest["revisions"]["dclm"] = f"{DCLM_REPO}@{DCLM_REVISION}:{DCLM_SHARD}"
    print(f"dclm: {len(selected)} docs, {total/1e6:.1f} MB (scanned {scanned/1e6:.0f} MB)", flush=True)
    for cat, quota in QUOTAS:
        print(f"  {cat}: {(quota - remaining[cat])/1e6:.2f}/{quota/1e6:.1f} MB", flush=True)

    corpora = {"dclm": selected} if args.only_dclm else {"owt": owt_docs, "dclm": selected}
    for name, docs in corpora.items():
        write_docs(out / f"{name}_docs.bin", docs)

    # --- oracles ---
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(out / "tiktoken_cache"))
    r50k = tiktoken.get_encoding("r50k_base")
    from tokenizers import Tokenizer as HfTok
    qwen = HfTok.from_file(str(qwen_dir / "tokenizer.json"))
    tiny = HfTok.from_file(tiny_path)

    backends = {
        "gpt2": lambda t: r50k.encode(t),
        "qwen2_5": lambda t: qwen.encode(t, add_special_tokens=False).ids,
        "tinyllama": lambda t: tiny.encode(t, add_special_tokens=False).ids,
    }
    manifest.setdefault("corpora", {})
    for name, docs in corpora.items():
        manifest["corpora"][name] = {"docs": len(docs), "bytes": sum(len(d.encode()) for d in docs)}
        for bname, fn in backends.items():
            t0 = time.perf_counter()
            all_ids = [fn(t) for t in docs]
            dt = time.perf_counter() - t0
            ntok = sum(len(x) for x in all_ids)
            write_ids(out / f"ids.{name}.{bname}.bin", all_ids)
            print(f"{name}/{bname}: {ntok/1e6:.2f}M tokens in {dt:.1f}s ({ntok/dt/1e6:.2f} Mtoks/s)", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest, indent=2))


def write_docs(path, docs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(docs)))
        for t in docs:
            raw = t.encode("utf-8")
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)


def write_ids(path, all_ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(all_ids)))
        for ids in all_ids:
            f.write(struct.pack("<Q", len(ids)))
            f.write(struct.pack(f"<{len(ids)}I", *ids) if ids else b"")


if __name__ == "__main__":
    main()
