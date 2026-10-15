#!/usr/bin/env python3
"""Regenerate the checked-in golden fixtures for gigatoken_cs tests.

Regenerates (deterministically) into tests/Gigatoken.Tests/Fixtures/:
  tiny_sp.model  - toy SentencePiece BPE model (vocab 512, byte fallback, NFKC)
  tiny.parquet   - tiny pyarrow table (string col "text" incl. a null, int col "n")
and prints the C# golden literals baked into SentencePieceTests /
GoldenVectorTests / ParquetFileSourceTests.

Note: retraining tiny_sp.model is functionally deterministic (the printed
golden IDs are stable across runs) but not byte-identical: zero-frequency
tail merges order nondeterministically. The golden IDs are the source of
truth; the checked-in .model bytes are just one valid training output.

Golden IDs always come from an independent oracle (the `sentencepiece` /
`tiktoken` packages), never from gigatoken itself.

Requires (only for the generators you run):
  uv run --with sentencepiece --with pyarrow --with tiktoken python tools/export_goldens.py --all

Usage:
  python tools/export_goldens.py --all [--out tests/Gigatoken.Tests/Fixtures]
  python tools/export_goldens.py --sp | --gpt2 | --parquet [--out ...]
"""

import argparse
import sys
from pathlib import Path

SP_VOCAB_SIZE = 512

SP_CORPUS = """\
the quick brown fox jumps over the lazy dog
hello world hello world hello again
leading space and trailing space
double  space and\ttabs\ttoo
numbers 1234567890 and 3.14159
Unicode: caf\u00e9 na\u00efve r\u00e9sum\u00e9
code: def f(x): return x * 2
aaaa bbbb cccc dddd eeee ffff
short line
another line here for merges
""" * 8

SP_TEXTS = [
    "Hello, world!",
    "double  space",
    "\ttab and\nnewline",
    " leading",
    "trailing ",
    "NFKC: \ufb01 \u00bd \uff57\uff49\uff44\uff45",
    "CJK \u65e5\u672c\u8a9e",
    "null\x00byte",
    "code:\n    indented block",
    "",
    " ",
]

GPT2_TEXTS = [
    "Hello, world!",
    "",
    " ",
    "Hello",
    "The quick brown fox jumps over the lazy dog.",
    "Unicode: caf\u00e9 \u65e5\u672c\u8a9e \U0001f389",
    "'quotes' \"double\" `backticks`",
    " Say 'hi' & <bye>\n",
]

PARQUET_ROWS = ["hello world", "second doc with unicode caf\u00e9", None, "line1\nline2", ""]


def _csharp_ids(ids: list[int]) -> str:
    return "[]" if not ids else "[" + ", ".join(f"{i}u" for i in ids) + "]"


def _csharp_uint_array(ids: list[int]) -> str:
    return "Array.Empty<uint>()" if not ids else "new uint[] { " + ", ".join(f"{i}u" for i in ids) + " }"


def _csharp_str(text: str) -> str:
    out = ['"']
    for ch in text:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\0":
            out.append("\\0")
        elif 32 <= o < 127:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\u{o:04X}")
        else:
            out.append(f"\\U{o:08X}")
    out.append('"')
    return "".join(out)


def gen_sp(out: Path) -> None:
    import sentencepiece as spm

    train_txt = out / "_sp_train.txt"
    train_txt.write_text(SP_CORPUS, encoding="utf-8")
    prefix = str(out / "_sp")
    # Silence the trainer's stdout chatter; errors still raise.
    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()):
        spm.SentencePieceTrainer.train(
            input=str(train_txt),
            model_prefix=prefix,
            vocab_size=SP_VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,
            shuffle_input_sentence=False,
        )
    (out / "tiny_sp.model").write_bytes((out / "_sp.model").read_bytes())
    for ext in ("_sp.model", "_sp.vocab", "_sp_train.txt"):
        (out / ext).unlink(missing_ok=True)

    proc = spm.SentencePieceProcessor(str(out / "tiny_sp.model"))
    print("// SentencePieceTests.Golden (sentencepiece oracle):")
    for t in SP_TEXTS:
        print(f"    ({_csharp_str(t)}, {_csharp_ids(proc.encode(t, out_type=int))}),")


def gen_gpt2() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    print("// GoldenVectorTests.Gpt2Goldens (tiktoken gpt2 oracle):")
    for t in GPT2_TEXTS:
        print(f"    new object[] {{ {_csharp_str(t)}, {_csharp_uint_array(enc.encode(t))} }},")


def gen_parquet(out: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "text": pa.array(PARQUET_ROWS, type=pa.string()),
            "n": list(range(len(PARQUET_ROWS))),
        }
    )
    pq.write_table(table, out / "tiny.parquet")
    print(f"wrote {out / 'tiny.parquet'} ({len(PARQUET_ROWS)} rows, 1 null)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/Gigatoken.Tests/Fixtures")
    ap.add_argument("--sp", action="store_true", help="train tiny_sp.model + print SP goldens")
    ap.add_argument("--gpt2", action="store_true", help="print GPT-2 goldens (tiktoken oracle)")
    ap.add_argument("--parquet", action="store_true", help="write tiny.parquet")
    ap.add_argument("--all", action="store_true", help="all of the above")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_sp = args.sp or args.all
    run_gpt2 = args.gpt2 or args.all
    run_parquet = args.parquet or args.all
    if not (run_sp or run_gpt2 or run_parquet):
        ap.error("nothing to do: pass --sp, --gpt2, --parquet, or --all")
    if run_sp:
        gen_sp(out)
    if run_gpt2:
        gen_gpt2()
    if run_parquet:
        gen_parquet(out)


if __name__ == "__main__":
    sys.exit(main())
