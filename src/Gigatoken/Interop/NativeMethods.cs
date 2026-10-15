// Lifetime rules for native handles.
// - Every handle is an opaque heap Box owned by its SafeHandle. The ONLY way
//   to destroy one is Dispose() (or the finalizer). Never call gt_*_free
//   directly; never DangerousGetHandle + free.
// - P/Invoke methods take the SafeHandle-derived type directly, so the runtime
//   issues DangerousAddRef/Release around each call. The handle cannot be
//   closed concurrently with a call.
// - Out-buffers (ids, bytes, strings, pointer arrays) are libc-malloc'd by
//   Rust and MUST be released with gt_free after copying. NativeBridge does
//   this in finally blocks; public API never sees raw pointers.
// - Byte slices returned by gt_pretokenize_iter_next borrow the iterator's
//   internal buffer: copy them before advancing or freeing the iterator.
// - gt_free(NULL) is a no-op; null-ptr + len-0 is accepted as empty input.
using System.Runtime.InteropServices;

namespace Gigatoken.Interop;

/// <summary>Raw P/Invoke surface. Unsafe, pointer-based; use <see cref="NativeBridge"/> instead.</summary>
internal static unsafe partial class NativeMethods
{
    internal const string LibraryName = "gigatoken_cabi";

    static NativeMethods() => NativeLib.EnsureLoaded();

    [LibraryImport(LibraryName)]
    internal static partial int gt_version_major();

    [LibraryImport(LibraryName)]
    internal static partial int gt_version_minor();

    [LibraryImport(LibraryName)]
    internal static partial int gt_version_patch();

    [LibraryImport(LibraryName)]
    internal static partial void gt_free(nint ptr);

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_last_error_len();

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_last_error_write(byte* buf, nuint len);

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_special_hit_count();

    [LibraryImport(LibraryName)]
    internal static partial int gt_special_hit_get(nuint index, nuint* @out);

    [LibraryImport(LibraryName)]
    internal static partial void gt_tokenizer_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_kind(TokenizerHandle handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_load_hf_json(byte* data, nuint len, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_bpe_from_tiktoken(
        byte* path, nuint pathLen, byte* pretokenizer, nuint pretokenizerLen,
        byte** specialsKeys, nuint* specialsKeysLens, uint* specialsVals, nuint nSpecials,
        out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_bpe_from_hf(byte* path, nuint pathLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_sp_from_hf(byte* path, nuint pathLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_sentencepiece_from_bytes(byte* json, nuint jsonLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_sentencepiece_model_to_json(
        byte* model, nuint modelLen, byte** outJson, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_encode(
        TokenizerHandle handle, byte* input, nuint inputLen, uint** outIds, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial int gt_sp_encode_no_normalize(
        TokenizerHandle handle, byte* strPtr, nuint strLen, uint** outIds, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_encode_batch(
        TokenizerHandle handle, byte** inputs, nuint* inputsLens, nuint n, int parallel,
        nint wrapOrNull, uint** outFlat, nuint* outFlatLen, long** outCounts, nuint* outNrows);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_encode_regions(
        TokenizerHandle handle, byte** buffers, nuint* buffersLens, nuint n,
        byte* sep, nuint sepLen, int hasSep, int parallel,
        nint wrapOrNull, uint** outFlat, nuint* outFlatLen, long** outCounts, nuint* outNrows);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_encode_batch_padded(
        TokenizerHandle handle, byte** inputs, nuint* inputsLens, nuint n, int parallel,
        PadOptsHandle padOpts, uint** outMatrix, nuint* outNrows, nuint* outNcols, long** outLengths);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_encode_files(
        TokenizerHandle handle, SourceHandle source, int parallel,
        uint** outFlat, nuint* outFlatLen, long** outCounts, nuint* outNrows);

    [LibraryImport(LibraryName)]
    internal static partial int gt_tokenizer_decode(
        TokenizerHandle handle, uint* ids, nuint idsLen, byte** outBytes, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_vocab_size(TokenizerHandle handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_vocab_entry(
        TokenizerHandle handle, nuint index, uint* outId, byte** outBytes, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_merge_count(TokenizerHandle handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_merge_entry(
        TokenizerHandle handle, nuint index,
        byte** outLeft, nuint* outLeftLen, byte** outRight, nuint* outRightLen);

    [LibraryImport(LibraryName)]
    internal static partial nuint gt_tokenizer_cache_entries(TokenizerHandle handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_text_source_new(
        byte** paths, nuint* pathsLens, nuint nPaths,
        byte* sep, nuint sepLen, int hasSep, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_jsonl_source_new(
        byte** paths, nuint* pathsLens, nuint nPaths,
        byte* field, nuint fieldLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_parquet_source_new(
        byte** paths, nuint* pathsLens, nuint nPaths,
        byte* column, nuint columnLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_path_source_new(
        byte** paths, nuint* pathsLens, nuint nPaths, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_bytes_source_new(
        byte** buffers, nuint* buffersLens, nuint nBuffers,
        byte* sep, nuint sepLen, int hasSep, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial void gt_source_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_matcher_new(
        byte** patterns, nuint* patternsLens, nuint n, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial void gt_matcher_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_matcher_present(
        MatcherHandle handle, byte* text, nuint textLen, nuint** outIndices, nuint* outN);

    [LibraryImport(LibraryName)]
    internal static partial int gt_wrap_opts_new(
        uint* prefix, nuint prefixLen, uint* suffix, nuint suffixLen,
        int hasMax, nuint maxTokens, int truncateLeft, nint matcherOrNull,
        out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial void gt_wrap_opts_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_pad_opts_new(
        uint padId, int hasMax, nuint maxLength, int padToMaxLength, int truncate,
        int padLeft, int truncateLeft, uint* prefix, nuint prefixLen,
        uint* suffix, nuint suffixLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial void gt_pad_opts_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_train_bpe(
        int inputTag, byte* input, nuint inputLen, nint sourceOrNull,
        nuint vocabSize, byte** specials, nuint* specialsLens, nuint nSpecials,
        int tieBreaking, byte* sep, nuint sepLen, int hasSep,
        uint** outVocabIds, byte*** outVocabPtrs, nuint** outVocabLens, nuint* outVocabN,
        byte*** outMergeLeft, nuint** outMergeLeftLens,
        byte*** outMergeRight, nuint** outMergeRightLens, nuint* outMergeN);

    [LibraryImport(LibraryName)]
    internal static partial int gt_pretokenize_iter_new(byte* bytes, nuint bytesLen, out nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_pretokenize_iter_next(
        PretokenIterHandle iter, byte** outPtr, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial void gt_pretokenize_iter_free(nint handle);

    [LibraryImport(LibraryName)]
    internal static partial int gt_pretokenized_counts(
        byte* bytes, nuint bytesLen, byte* sep, nuint sepLen, int hasSep,
        byte*** outKeys, nuint** outKeysLens, nuint** outCounts, nuint* outN);

    [LibraryImport(LibraryName)]
    internal static partial int gt_hub_file(
        byte* repo, nuint repoLen, byte* filename, nuint filenameLen, int repoType,
        byte* revision, nuint revisionLen, byte** outPath, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial int gt_looks_like_repo_id(byte* str, nuint strLen);

    [LibraryImport(LibraryName)]
    internal static partial int gt_get_hf_token(byte** outBuf, nuint* outLen);

    [LibraryImport(LibraryName)]
    internal static partial void gt_set_max_cache_bytes(int hasValue, nuint value);

    [LibraryImport(LibraryName)]
    internal static partial void gt_get_max_cache_bytes(int* outHas, nuint* outValue);
}
