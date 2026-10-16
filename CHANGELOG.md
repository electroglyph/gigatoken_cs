# Changelog

All notable changes to this project will be documented in this file.

## [0.11.0] - 2026-10-16

Correctness release: no design changes, only bug fixes with regression
tests.

### Fixed

- Pretoken iterator (`cabi/lib.rs`): hold the position lock across the
  whole read-advance step, so concurrent `next` calls can no longer deliver
  the same span twice or skip one. The by-design behavior that a successful
  `next` clears pending forbid hits is now documented on the function.

- `gt_vocab_entry` (`cabi/lib.rs`): validate all four out-pointers up
  front and return an error instead of writing through a null pointer; on
  the error path `*out_id` is only written after the `out_vec` payload
  succeeds, so a heap failure leaves every out-param untouched.

- `gt_tokenizer_encode_regions` (`cabi/lib.rs`): run the forbid scan
  before encoding, so a forbidden span errors without paying for a wasted
  encode.

- SentencePiece model conversion (`cabi/sp_model.rs`): deduplicate
  identical `(left, right)` merge pairs, keeping the first (highest-rank)
  occurrence, so duplicate piece entries no longer emit the same merge
  twice. Malformed inner fields are handled strictly: missing piece
  content / score / type, trainer unk piece, and normalizer charmap fall
  back to fixed defaults, while present-but-wrong-wire-type fields are
  errors instead of silent defaults.

- Padded batch encoding (`src/Gigatoken/Tokenizer.cs`): throw
  `ArgumentException` when `MaxLength` leaves no room for the affix
  tokens, instead of clamping the cap to zero and overflowing the matrix
  (or returning a silently over-wide row). A lone `MaxLength` is now
  inert unless `Truncate` / `PadToMaxLength` is set, matching the native
  padding path.

- `Tokenizer.FromTiktoken` (`src/Gigatoken/Tokenizer.cs`): infer the
  pretokenizer with an ordinal (case-sensitive) lookup, so a wrongly
  cased file stem no longer silently loads another scheme.

- `Tokenizer.FromPretrained` (`src/Gigatoken/Tokenizer.cs`): resolve
  local directories and files before consulting the hub, so a local path
  whose name also looks like a repo id never triggers a network fetch.

- Null-argument names: batch element failures now report the indexed
  `texts[i]` / `paths[i]` name (`Tokenizer`, `Backends`,
  `TiktokenCompat.EncodeBatch`, `Sources.FileSourceBase`), and
  `Backends.FromTiktoken`, `Hub.GetFile`, `Hub.LooksLikeRepoId`, and
  `Tokenizer.FromPretrained` guard their null inputs.

- Empty padded batches (`src/Gigatoken/HfCompat.cs`): keep the pad width,
  producing a `(0, max)` tensor when `padding="max_length"` with
  `max_length` and `(0, 0)` otherwise.

- `HfCompat.ConvertTokensToString` (`src/Gigatoken/HfCompat.cs`):
  iterate by rune instead of splitting UTF-16 code units, so
  astral-plane characters round-trip instead of becoming lone
  surrogates.

- Tiktoken allowed/disallowed sets
  (`src/Gigatoken/TiktokenCompat.cs`): an explicitly passed set now
  always wins over the `"all"` / `"none"` string, in both directions.

- Forbid-matcher cache (`src/Gigatoken/TiktokenCompat.cs`): the cache
  key uses length-prefixed framing instead of a bare separator join, so
  distinct scans (even with NUL-bearing special tokens) never share a
  matcher. Eviction no longer disposes entries that concurrent encodes
  may still borrow (evicted matchers are left for `SafeHandle`
  finalization), and an out-of-range native matcher index maps to
  `GigatokenException` instead of leaking `IndexOutOfRangeException`.

- `DecodeSingleTokenBytes`, `DecodeTokensBytes`, and `TokenByteValues`
  (`src/Gigatoken/TiktokenCompat.cs`): return array copies, so mutating
  a result can no longer poison the shared caches.

- `NativeBridge.ReadSpecialHits`
  (`src/Gigatoken/Interop/NativeBridge.cs`): check the
  `gt_special_hit_get` return code instead of silently yielding `0`.

- Native handle lifetime (`src/Gigatoken/Interop/NativeBridge.cs`):
  `wrap` / `matcher` / `source` handles are pinned with
  `DangerousAddRef` / `DangerousRelease` instead of
  `DangerousGetHandle` + `GC.KeepAlive`, closing the use-after-free
  window on concurrent dispose.

- Native buffer cleanup (`src/Gigatoken/Interop/NativeBridge.cs`):
  multi-output calls (`EncodeBatch`, `EncodeRegions`, `EncodeFiles`,
  `EncodeBatchPadded`, `TrainBpe`) copy outputs inside a joint
  `try`/`finally`, so a failed copy can no longer leak the
  already-allocated native buffers.

- Native library loading (`src/Gigatoken/Interop/NativeLib.cs`): a bad
  `GIGATOKEN_NATIVE` path is a fail-fast error that echoes the offending
  value, with no silent fallback probing. Loader startup is
  lock-serialized and publishes the native handle before the loaded
  flag. The CLI prints the full inner-exception chain so the underlying
  loader error stays visible.

- `EncodeCache.SetMaxBytes` (`src/Gigatoken/HubCacheTrain.cs`):
  convert with overflow checking and report a too-large budget as
  `ArgumentOutOfRangeException` instead of silently truncating on
  32-bit.

- File-based training (`cabi/lib.rs`): honor the caller-provided
  separator when the source format carries none (previously silently
  dropped).

- `BatchPadded` benchmark (`benches/Gigatoken.Benchmarks/Program.cs`):
  set `Truncate: true` alongside `PadToMaxLength`, so it exercises the
  intended path instead of throwing on overlong rows.
