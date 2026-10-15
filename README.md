# Gigatoken for .NET

Fast multilingual tokenizers for .NET: a C# binding over the gigatoken Rust
core (`cabi/`), mirroring the Python `gigatoken` package API (v0.10.0).
BPE and SentencePiece backends, file/bytes sources, BPE training, a
HuggingFace `tokenizers`-compatible surface, a `tiktoken`-compatible surface,
and a `gigatoken bench` CLI.

- Same Rust engine as upstream Python `gigatoken`, exposed to C# via a
  51-function C ABI (`gt_*`) and `LibraryImport` P/Invoke.
- Bit-identical output to HuggingFace `tokenizers` on the same
  `tokenizer.json` (spot-verified against `Tokenizers.DotNet`).
- ~18x faster than the HF `tokenizers` C# binding on short sequences
  (see [Benchmarks](#benchmarks)).

## Install / build

Prerequisites: .NET 10 SDK, a Rust toolchain (`cargo`).

```sh
git clone --recurse-submodules https://github.com/<you>/gigatoken_cs.git
./build.sh            # host RID: cargo build + stage runtimes/ + dotnet build
TARGETS="x86_64-unknown-linux-gnu aarch64-apple-darwin" ./build.sh  # cross
./build.sh --pack     # ... then dotnet pack (produces Gigatoken.0.10.0.nupkg)
```

Already cloned without submodules? `git submodule update --init`.

Upstream Python repo:
[`marcelroed/gigatoken`](https://github.com/marcelroed/gigatoken),
checked in as the `upstream/` git submodule (pinned to commit `fac0114`).
The Rust core is compiled from those exact sources via `#[path]` includes
in `cabi/lib.rs` — no fork, no copied files.

If you build the Rust side yourself, point the binding at it without
`build.sh`:

```sh
GIGATOKEN_NATIVE=/path/to/libgigatoken_cabi.so dotnet run --project MyApp
```

## Quickstart

```csharp
using Gigatoken;

using var tok = Tokenizer.FromPretrained("gpt2"); // repo-id, dir, or tokenizer.json
uint[] ids = tok.Encode("Hello, world!");          // [15496, 11, 995, 0]
string text = tok.DecodeToString(ids);             // "Hello, world!"
```

More runnable snippets live in [`examples/`](examples) — each one is compiled
into the test assembly and executed by `ExamplesTests`, so they cannot rot:

| Example | Upstream analogue |
|---|---|
| `examples/Quickstart.cs` | `examples/quickstart.py` |
| `examples/TiktokenDropIn.cs` | `examples/tiktoken_drop_in.py` |
| `examples/HfDropIn.cs` | `examples/hf_drop_in.py` |
| `examples/EncodeFilesExample.cs` | `examples/encode_files.py` |

## API surfaces

- **Core** (`Tokenizer`): `FromPretrained` (repo-id / dir / `tokenizer.json` /
  `.tiktoken` / `.model`), `FromJson`, `FromTiktoken`, `FromSentencePiece`,
  `Encode`, `EncodeBatch`, `EncodeFiles`, `DecodeToString`, `EncodeBatchPadded`,
  `TrainBpe` (via `BpeTrainer`).
- **HuggingFace-compat** (`tok.AsHf()` → `HfCompat`): `Call` (padding /
  truncation / tensors / attention mask), `Decode`, `BatchDecode`,
  `ConvertIdsToTokens`, `ConvertTokensToIds`, `AllSpecialTokens`.
- **Tiktoken-compat** (`tok.AsTiktoken()` → `TiktokenCompat`):
  `EncodeOrdinary`, `EncodeWithAllowedSpecial`, `Decode`, `EncodeBatch`.
- **Sources**: `BytesSource`, `TextFileSource` (separator snapshot, streaming
  from disk inside Rust).
- **Training**: `BpeTrainer.Train(corpus, vocabSize, specials, tieBreaking)`.

```csharp
// Tiktoken-style drop-in
using var tok = Tokenizer.FromTiktoken("cl100k_base.tiktoken", pretokenizer: "cl100k");
var tik = tok.AsTiktoken();
uint[] ids = tik.EncodeOrdinary("Hello, world!");

// HF-style drop-in
using var hfTok = Tokenizer.FromPretrained("Qwen/Qwen3-0.6B");
var hf = hfTok.AsHf();
BatchEncoding batch = hf.Call(["Hello, world!"], padding: true);
```

## CLI

```sh
dotnet run --project tools/Gigatoken.Cli -- bench <tokenizer> <files...> [options]

  --stream-from-disk     encode via file sources in Rust (disk IO inside timing)
  --compare-to <tok>     second Gigatoken tokenizer spec for parity + speedup
  --comparison-limit <n> max bytes compared (default 100MB, decimal SI)
  --validate             fail if any compared doc diverges (needs --compare-to)
  --doc-separator <sep>  split files on this separator
  --pretokenizer <name>  scheme for .tiktoken inputs (gpt2, cl100k, ...)
```

Note: upstream compares against HuggingFace `tokenizers`; there is no live HF
path in .NET, so `--compare-to` takes a second Gigatoken tokenizer instead.
Golden-vector validation lives in the test suite (`GoldenVectorTests`).

## Benchmarks

Short-sequence latency, GPT-2 `tokenizer.json`, this repo's fixture —
this library (release native build) vs
[`Tokenizers.DotNet`](https://github.com/sappho192/Tokenizers.DotNet) 1.4.1
(the standard HF `tokenizers` Rust-core C# binding). Two API paths of
*this library* are measured: raw `Tokenizer.Encode`, and `HfCompat.Call`
— same library, same engine, just the HuggingFace-style calling
convention (padding / truncation / attention-mask handling). Outputs
spot-verified bit-identical. Best-of-3 rounds, single
`Encode` calls from C#.

| scenario | this lib, raw `Encode` | this lib, `HfCompat.Call` | HF Tokenizers.DotNet |
|---|---:|---:|---:|
| hot, 4-token string | 0.17 µs | 1.64 µs | 2.07 µs (~12x / ~1.3x) |
| hot, 31-token string | 0.45 µs | 1.00 µs | 8.23 µs (~18x / ~8x) |
| 10k varied short strings (~9 tokens avg) | 0.22 µs | 0.81 µs | 4.0 µs (~18x / ~5x) |
| 10k unique random strings, pretoken cache defeated | 0.52 µs | 1.14 µs | 9.4 µs (~18x / ~8x) |

Reading: raw `Encode` is ~18x faster across the board; the ~18x persists
with the pretoken cache defeated, so it is engine speed, not caching.
The `HfCompat.Call` path — again, same library, just the HF-style API —
adds ~0.6 µs of managed overhead (arg validation, attention
mask) and still lands ~5–8x ahead. HF-side latency scales ~linearly with
length (~0.25 µs/token); this library's curve is much flatter.

For bulk throughput, upstream reports ~100–1000x vs HF `tokenizers` on
multi-GB corpora (see upstream README benchmarks — multithreaded Rust on both
sides; the gap is algorithmic plus parallelism, and it applies here too
since it is the same core). To reproduce bulk numbers locally:

```sh
dotnet run --project tools/Gigatoken.Cli -- bench <tokenizer> <files...> [--stream-from-disk]
dotnet run -c Release --project benches/Gigatoken.Benchmarks -- <tokenizer> [corpusFile]
```

## Tests

```sh
dotnet test Gigatoken.slnx -c Release
```

132 tests, all self-contained except two external-data sweeps that skip when
data is absent (same policy as upstream):

- `Phase7ParityTests` — OWT/DCLM hostile sweep; fetch data with
  `tools/fetch_dclm_sample.py`, export vectors with
  `tools/export_phase7_goldens.py`.
- `ProcessIsolationTests` — needs the CLI built (a test-project build
  dependency, so plain `dotnet test` already covers it).

Checked-in fixtures (`tests/Gigatoken.Tests/Fixtures/`) are regenerated with
independent oracles, never from gigatoken itself:

```sh
uv run --with sentencepiece --with pyarrow --with tiktoken \
    python tools/export_goldens.py --all
```

## Layout

| Path | Contents |
|---|---|
| `cabi/` | Rust C ABI (`gigatoken-cabi` cdylib): 51 `gt_*` functions |
| `src/Gigatoken/` | C# library (`net10.0`): facade, backends, sources, compat layers |
| `src/Gigatoken/Interop/` | `LibraryImport` P/Invoke, SafeHandles, native loader |
| `tests/Gigatoken.Tests/` | xUnit suite + checked-in fixtures |
| `tools/Gigatoken.Cli/` | `gigatoken bench` CLI |
| `tools/export_goldens.py` | regenerates fixtures + prints golden literals |
| `tools/fetch_dclm_sample.py` | curated DCLM sample downloader (pinned shard + quotas) |
| `benches/Gigatoken.Benchmarks/` | BenchmarkDotNet harness |
| `examples/` | runnable snippets (executed by `ExamplesTests`) |

## Port notes (adaptations, not gaps)

- `Tokenizer.EncodeNoNormalize` exists on the facade (upstream reaches the
  backend extra via `__getattr__`; C# has none, so it throws
  `InvalidOperationException` on BPE backends).
- `TokenizerLoadException : IOException` — upstream names
  `InvalidDataException`, which is `sealed` in .NET and cannot be inherited.
- `SubstringMatcher` is `public` — `Options.Forbid` exposes it, so `internal`
  is impossible.
- `TiktokenCompat.IsSpecialToken` mirrors upstream `is_special_token`;
  `IsSpecial` exists as an alias.
- `--compare-to` compares two Gigatoken tokenizers; validation against
  checked-in golden ids is `GoldenVectorTests`, not a CLI mode.
- UTF-8 marshaling is manual pinning (`fixed` + `Utf8.GetBytes`), not
  `MarshalUsing` custom marshallers — same bytes on the wire.
- `ProcessIsolationTests` spawns the CLI as a subprocess (no `fork` on .NET).

## License

MIT — see upstream `LICENSE`.
