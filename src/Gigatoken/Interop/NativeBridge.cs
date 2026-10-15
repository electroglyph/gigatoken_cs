using System.Runtime.InteropServices;
using System.Text;

namespace Gigatoken.Interop;

/// <summary>Native error codes (port.md §1 error model).</summary>
internal static class GtCode
{
    internal const int Ok = 0;
    internal const int InvalidArg = 1;
    internal const int Utf8 = 2;
    internal const int IoNotFound = 3;
    internal const int Unauthorized = 4;
    internal const int LoadParse = 5;
    internal const int SpecialTokenFound = 6;
}

/// <summary>Pins N managed byte buffers and builds native pointer+len arrays for one call.</summary>
internal sealed class PinnedBuffers : IDisposable
{
    private readonly GCHandle[] _pins;
    internal unsafe byte** Ptrs { get; private set; }
    internal unsafe nuint* Lens { get; private set; }
    internal nuint Count { get; }

    internal unsafe PinnedBuffers(IReadOnlyList<byte[]> buffers)
    {
        _pins = new GCHandle[buffers.Count];
        try
        {
            for (var i = 0; i < buffers.Count; i++)
                _pins[i] = GCHandle.Alloc(buffers[i], GCHandleType.Pinned);
            var n = (nuint)buffers.Count;
            Count = n;
            Ptrs = (byte**)NativeMemory.Alloc(n, (nuint)sizeof(byte*));
            Lens = (nuint*)NativeMemory.Alloc(n, (nuint)sizeof(nuint));
            for (var i = 0; i < buffers.Count; i++)
            {
                Ptrs[i] = (byte*)_pins[i].AddrOfPinnedObject();
                Lens[i] = (nuint)buffers[i].Length;
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        unsafe
        {
            if (Ptrs is not null) NativeMemory.Free(Ptrs);
            if (Lens is not null) NativeMemory.Free(Lens);
            Ptrs = null;
            Lens = null;
        }
        if (_pins is not null)
        {
            foreach (var p in _pins)
                if (p.IsAllocated) p.Free();
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>Safe (no raw pointers leak out) wrappers over <see cref="NativeMethods"/>.</summary>
internal static unsafe class NativeBridge
{
    internal static readonly Encoding Utf8 = Encoding.UTF8;

    internal static byte[] Utf8Bytes(string s) => Utf8.GetBytes(s);

    internal static string ReadLastError()
    {
        var ulen = NativeMethods.gt_last_error_len();
        if (ulen == 0) return "unknown native error";
        var len = checked((int)ulen);
        var buf = new byte[len];
        fixed (byte* p = buf)
        {
            var n = checked((int)NativeMethods.gt_last_error_write(p, (nuint)len));
            if (n < 0 || n > len) return Utf8.GetString(buf, 0, len);
            return Utf8.GetString(buf, 0, n);
        }
    }

    internal static int[] ReadSpecialHits()
    {
        var n = checked((int)NativeMethods.gt_special_hit_count());
        var hits = new int[n];
        for (var i = 0; i < n; i++)
        {
            nuint v = 0;
            NativeMethods.gt_special_hit_get((nuint)i, &v);
            hits[i] = checked((int)v);
        }
        return hits;
    }

    internal static void Check(int rc)
    {
        if (rc == GtCode.Ok) return;
        var msg = ReadLastError();
        throw rc switch
        {
            GtCode.InvalidArg => new ArgumentException(msg),
            GtCode.Utf8 => new ArgumentException(msg),
            GtCode.IoNotFound => new FileNotFoundException(msg),
            GtCode.Unauthorized => new UnauthorizedAccessException(msg),
            GtCode.LoadParse => new TokenizerLoadException(msg),
            GtCode.SpecialTokenFound => new SpecialTokenFoundException(ReadSpecialHits(), msg),
            _ => new GigatokenException($"native error {rc}: {msg}"),
        };
    }

    private static uint[] TakeU32(uint* ptr, nuint len)
    {
        try
        {
            var n = checked((int)len);
            var out_ = new uint[n];
            new ReadOnlySpan<uint>(ptr, n).CopyTo(out_);
            return out_;
        }
        finally { NativeMethods.gt_free((nint)ptr); }
    }

    private static byte[] TakeBytes(byte* ptr, nuint len)
    {
        try
        {
            var n = checked((int)len);
            var out_ = new byte[n];
            new ReadOnlySpan<byte>(ptr, n).CopyTo(out_);
            return out_;
        }
        finally { NativeMethods.gt_free((nint)ptr); }
    }

    private static long[] TakeI64(long* ptr, nuint len)
    {
        try
        {
            var n = checked((int)len);
            var out_ = new long[n];
            new ReadOnlySpan<long>(ptr, n).CopyTo(out_);
            return out_;
        }
        finally { NativeMethods.gt_free((nint)ptr); }
    }

    private static nuint[] TakeUSize(nuint* ptr, nuint len)
    {
        try
        {
            var n = checked((int)len);
            var out_ = new nuint[n];
            new ReadOnlySpan<nuint>(ptr, n).CopyTo(out_);
            return out_;
        }
        finally { NativeMethods.gt_free((nint)ptr); }
    }

    private static TokenizerHandle WrapTokenizer(nint raw) => TokenizerHandle.FromRaw(raw);
    private static SourceHandle WrapSource(nint raw) => SourceHandle.FromRaw(raw);
    private static MatcherHandle WrapMatcher(nint raw) => MatcherHandle.FromRaw(raw);
    private static WrapOptsHandle WrapWrapOpts(nint raw) => WrapOptsHandle.FromRaw(raw);
    private static PadOptsHandle WrapPadOpts(nint raw) => PadOptsHandle.FromRaw(raw);
    private static PretokenIterHandle WrapIter(nint raw) => PretokenIterHandle.FromRaw(raw);

    // -- Loaders -----------------------------------------------------------

    internal static TokenizerHandle LoadHfJson(byte[] data)
    {
        nint h;
        fixed (byte* p = data)
            Check(NativeMethods.gt_load_hf_json(p, (nuint)data.Length, out h));
        return WrapTokenizer(h);
    }

    internal static TokenizerHandle BpeFromTiktoken(
        byte[] path, byte[] pretokenizer, IReadOnlyList<byte[]> specialKeys, uint[] specialVals)
    {
        if (specialKeys.Count != specialVals.Length)
            throw new ArgumentException("specialKeys/specialVals length mismatch");
        using var keys = new PinnedBuffers(specialKeys);
        nint h;
        fixed (byte* pp = path)
        fixed (byte* tp = pretokenizer)
        fixed (uint* vp = specialVals)
            Check(NativeMethods.gt_bpe_from_tiktoken(
                pp, (nuint)path.Length, tp, (nuint)pretokenizer.Length,
                keys.Ptrs, keys.Lens, vp, (nuint)specialVals.Length, out h));
        return WrapTokenizer(h);
    }

    internal static TokenizerHandle BpeFromHf(byte[] path)
    {
        nint h;
        fixed (byte* p = path)
            Check(NativeMethods.gt_bpe_from_hf(p, (nuint)path.Length, out h));
        return WrapTokenizer(h);
    }

    internal static TokenizerHandle SpFromHf(byte[] path)
    {
        nint h;
        fixed (byte* p = path)
            Check(NativeMethods.gt_sp_from_hf(p, (nuint)path.Length, out h));
        return WrapTokenizer(h);
    }

    internal static TokenizerHandle SentencePieceFromBytes(byte[] json)
    {
        nint h;
        fixed (byte* p = json)
            Check(NativeMethods.gt_sentencepiece_from_bytes(p, (nuint)json.Length, out h));
        return WrapTokenizer(h);
    }

    internal static byte[] SentencePieceModelToJson(byte[] model)
    {
        byte* jp = null; nuint jl = 0;
        fixed (byte* p = model)
            Check(NativeMethods.gt_sentencepiece_model_to_json(p, (nuint)model.Length, &jp, &jl));
        return TakeBytes(jp, jl);
    }

    // -- Encode / decode ---------------------------------------------------

    internal static uint[] Encode(TokenizerHandle h, byte[] input)
    {
        uint* ids = null; nuint n = 0;
        fixed (byte* p = input)
            Check(NativeMethods.gt_tokenizer_encode(h, p, (nuint)input.Length, &ids, &n));
        return TakeU32(ids, n);
    }

    internal static uint[] SpEncodeNoNormalize(TokenizerHandle h, byte[] text)
    {
        uint* ids = null; nuint n = 0;
        fixed (byte* p = text)
            Check(NativeMethods.gt_sp_encode_no_normalize(h, p, (nuint)text.Length, &ids, &n));
        return TakeU32(ids, n);
    }

    internal static uint[][] Unflatten(uint[] flat, long[] counts)
    {
        var rows = new uint[counts.Length][];
        var off = 0;
        for (var i = 0; i < counts.Length; i++)
        {
            var c = checked((int)counts[i]);
            if (c < 0) throw new GigatokenException($"native returned negative count {counts[i]}");
            if ((long)off + c > flat.Length)
                throw new GigatokenException("native counts exceed flat buffer");
            rows[i] = flat.AsSpan(off, c).ToArray();
            off += c;
        }
        if (off != flat.Length)
            throw new GigatokenException("native counts do not sum to flat buffer length");
        return rows;
    }

    internal static uint[][] EncodeBatch(
        TokenizerHandle h, IReadOnlyList<byte[]> inputs, bool parallel, WrapOptsHandle? wrap)
    {
        using var bufs = new PinnedBuffers(inputs);
        uint* flat = null; nuint flatLen = 0; long* counts = null; nuint nrows = 0;
        nint wrapRaw = wrap is null ? 0 : wrap.DangerousGetHandle();
        Check(NativeMethods.gt_tokenizer_encode_batch(
            h, bufs.Ptrs, bufs.Lens, bufs.Count, parallel ? 1 : 0, wrapRaw,
            &flat, &flatLen, &counts, &nrows));
        GC.KeepAlive(wrap);
        return Unflatten(TakeU32(flat, flatLen), TakeI64(counts, nrows));
    }

    internal static uint[][] EncodeRegions(
        TokenizerHandle h, IReadOnlyList<byte[]> buffers, byte[]? sep, bool parallel, WrapOptsHandle? wrap)
    {
        using var bufs = new PinnedBuffers(buffers);
        uint* flat = null; nuint flatLen = 0; long* counts = null; nuint nrows = 0;
        nint wrapRaw = wrap is null ? 0 : wrap.DangerousGetHandle();
        int rc;
        if (sep is null)
        {
            rc = NativeMethods.gt_tokenizer_encode_regions(
                h, bufs.Ptrs, bufs.Lens, bufs.Count, null, 0, 0, parallel ? 1 : 0, wrapRaw,
                &flat, &flatLen, &counts, &nrows);
        }
        else
        {
            fixed (byte* sp = sep)
                rc = NativeMethods.gt_tokenizer_encode_regions(
                    h, bufs.Ptrs, bufs.Lens, bufs.Count, sp, (nuint)sep.Length, 1, parallel ? 1 : 0, wrapRaw,
                    &flat, &flatLen, &counts, &nrows);
        }
        GC.KeepAlive(wrap);
        Check(rc);
        return Unflatten(TakeU32(flat, flatLen), TakeI64(counts, nrows));
    }

    internal static (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(
        TokenizerHandle h, IReadOnlyList<byte[]> inputs, bool parallel, PadOptsHandle pad)
    {
        using var bufs = new PinnedBuffers(inputs);
        uint* mat = null; nuint nrows = 0, ncols = 0; long* lens = null;
        Check(NativeMethods.gt_tokenizer_encode_batch_padded(
            h, bufs.Ptrs, bufs.Lens, bufs.Count, parallel ? 1 : 0, pad,
            &mat, &nrows, &ncols, &lens));
        uint[,] matrix;
        try
        {
            var rows = checked((int)nrows); var cols = checked((int)ncols);
            var total = checked(rows * cols);
            matrix = new uint[rows, cols];
            if (total != 0 && mat is null)
                throw new GigatokenException("native returned null matrix with nonzero size");
            var span = new ReadOnlySpan<uint>(mat, total);
            for (var r = 0; r < rows; r++)
                for (var c = 0; c < cols; c++)
                    matrix[r, c] = span[r * cols + c];
        }
        finally { NativeMethods.gt_free((nint)mat); }
        return (matrix, TakeI64(lens, nrows));
    }

    internal static uint[][] EncodeFiles(TokenizerHandle h, SourceHandle source, bool parallel)
    {
        uint* flat = null; nuint flatLen = 0; long* counts = null; nuint nrows = 0;
        Check(NativeMethods.gt_tokenizer_encode_files(h, source, parallel ? 1 : 0, &flat, &flatLen, &counts, &nrows));
        return Unflatten(TakeU32(flat, flatLen), TakeI64(counts, nrows));
    }

    internal static byte[] Decode(TokenizerHandle h, uint[] ids)
    {
        byte* bp = null; nuint bl = 0;
        fixed (uint* p = ids)
            Check(NativeMethods.gt_tokenizer_decode(h, p, (nuint)ids.Length, &bp, &bl));
        return TakeBytes(bp, bl);
    }

    // -- Vocab / merges / cache --------------------------------------------

    internal static int VocabSize(TokenizerHandle h) => checked((int)NativeMethods.gt_vocab_size(h));

    internal static IReadOnlyList<(uint Id, byte[] Bytes)> VocabEntries(TokenizerHandle h)
    {
        var n = VocabSize(h);
        var list = new List<(uint, byte[])>(n);
        for (nuint i = 0; i < (nuint)n; i++)
        {
            uint id = 0; byte* bp = null; nuint bl = 0;
            Check(NativeMethods.gt_vocab_entry(h, i, &id, &bp, &bl));
            list.Add((id, TakeBytes(bp, bl)));
        }
        return list;
    }

    internal static int MergeCount(TokenizerHandle h) => checked((int)NativeMethods.gt_merge_count(h));

    internal static IReadOnlyList<(byte[] Left, byte[] Right)> MergeEntries(TokenizerHandle h)
    {
        var n = MergeCount(h);
        var list = new List<(byte[], byte[])>(n);
        for (nuint i = 0; i < (nuint)n; i++)
        {
            byte* lp = null; nuint ll = 0; byte* rp = null; nuint rr = 0;
            Check(NativeMethods.gt_merge_entry(h, i, &lp, &ll, &rp, &rr));
            list.Add((TakeBytes(lp, ll), TakeBytes(rp, rr)));
        }
        return list;
    }

    internal static int CacheEntries(TokenizerHandle h) =>
        checked((int)NativeMethods.gt_tokenizer_cache_entries(h));

    // -- Sources ------------------------------------------------------------

    private delegate int SourceFn(byte* extra, nuint extraLen, int hasExtra, out nint handle);

    private static SourceHandle NewSource(PinnedBuffers paths, byte[]? extra, SourceFn fn)
    {
        nint h;
        int rc;
        if (extra is null)
        {
            rc = fn(null, 0, 0, out h);
        }
        else
        {
            fixed (byte* ep = extra)
                rc = fn(ep, (nuint)extra.Length, 1, out h);
        }
        Check(rc);
        return WrapSource(h);
    }

    internal static SourceHandle TextSource(IReadOnlyList<byte[]> paths, byte[]? sep)
    {
        using var bufs = new PinnedBuffers(paths);
        return NewSource(bufs, sep, (byte* sp, nuint sl, int hs, out nint h) =>
            NativeMethods.gt_text_source_new(bufs.Ptrs, bufs.Lens, bufs.Count, sp, sl, hs, out h));
    }

    internal static SourceHandle JsonlSource(IReadOnlyList<byte[]> paths, byte[] field)
    {
        using var bufs = new PinnedBuffers(paths);
        return NewSource(bufs, field, (byte* fp, nuint fl, int _, out nint h) =>
            NativeMethods.gt_jsonl_source_new(bufs.Ptrs, bufs.Lens, bufs.Count, fp, fl, out h));
    }

    internal static SourceHandle ParquetSource(IReadOnlyList<byte[]> paths, byte[] column)
    {
        using var bufs = new PinnedBuffers(paths);
        return NewSource(bufs, column, (byte* cp, nuint cl, int _, out nint h) =>
            NativeMethods.gt_parquet_source_new(bufs.Ptrs, bufs.Lens, bufs.Count, cp, cl, out h));
    }

    internal static SourceHandle PathSource(IReadOnlyList<byte[]> paths)
    {
        using var bufs = new PinnedBuffers(paths);
        Check(NativeMethods.gt_path_source_new(bufs.Ptrs, bufs.Lens, bufs.Count, out var h));
        return WrapSource(h);
    }

    internal static SourceHandle BytesSource(IReadOnlyList<byte[]> buffers, byte[]? sep)
    {
        using var bufs = new PinnedBuffers(buffers);
        return NewSource(bufs, sep, (byte* sp, nuint sl, int hs, out nint h) =>
            NativeMethods.gt_bytes_source_new(bufs.Ptrs, bufs.Lens, bufs.Count, sp, sl, hs, out h));
    }

    // -- Matcher / opts -----------------------------------------------------

    internal static MatcherHandle MatcherNew(IReadOnlyList<byte[]> patterns)
    {
        using var bufs = new PinnedBuffers(patterns);
        Check(NativeMethods.gt_matcher_new(bufs.Ptrs, bufs.Lens, bufs.Count, out var h));
        return WrapMatcher(h);
    }

    internal static int[] MatcherPresent(MatcherHandle h, byte[] text)
    {
        nuint* idx = null; nuint n = 0;
        fixed (byte* p = text)
            Check(NativeMethods.gt_matcher_present(h, p, (nuint)text.Length, &idx, &n));
        var raw = TakeUSize(idx, n);
        var out_ = new int[raw.Length];
        for (var i = 0; i < raw.Length; i++) out_[i] = checked((int)raw[i]);
        return out_;
    }

    internal static WrapOptsHandle WrapOptsNew(
        uint[] prefix, uint[] suffix, nuint? maxTokens, bool truncateLeft, MatcherHandle? matcher)
    {
        nint h;
        nint matcherRaw = matcher is null ? 0 : matcher.DangerousGetHandle();
        fixed (uint* pp = prefix)
        fixed (uint* sp = suffix)
            Check(NativeMethods.gt_wrap_opts_new(
                pp, (nuint)prefix.Length, sp, (nuint)suffix.Length,
                maxTokens.HasValue ? 1 : 0, maxTokens ?? 0, truncateLeft ? 1 : 0, matcherRaw,
                out h));
        GC.KeepAlive(matcher);
        return WrapWrapOpts(h);
    }

    internal static PadOptsHandle PadOptsNew(
        uint padId, nuint? maxLength, bool padToMaxLength, bool truncate,
        bool padLeft, bool truncateLeft, uint[] prefix, uint[] suffix)
    {
        nint h;
        fixed (uint* pp = prefix)
        fixed (uint* sp = suffix)
            Check(NativeMethods.gt_pad_opts_new(
                padId, maxLength.HasValue ? 1 : 0, maxLength ?? 0,
                padToMaxLength ? 1 : 0, truncate ? 1 : 0, padLeft ? 1 : 0, truncateLeft ? 1 : 0,
                pp, (nuint)prefix.Length, sp, (nuint)suffix.Length, out h));
        return WrapPadOpts(h);
    }

    // -- Train --------------------------------------------------------------

    internal static (Dictionary<uint, byte[]> Vocab, List<(byte[] Left, byte[] Right)> Merges) TrainBpe(
        int inputTag, byte[]? input, SourceHandle? source,
        int vocabSize, IReadOnlyList<byte[]> specials, int tieBreaking, byte[]? sep)
    {
        if (specials is null) throw new ArgumentNullException(nameof(specials));
        if (vocabSize < 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (inputTag == 0 || inputTag == 1)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
        }
        else if (inputTag == 2)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
        }
        var vocabSizeNative = (nuint)vocabSize;
        using var sp = new PinnedBuffers(specials);
        uint* vids = null; byte** vptrs = null; nuint* vlens = null; nuint vocabN = 0;
        byte** mleft = null; nuint* mleftLens = null; byte** mright = null; nuint* mrightLens = null; nuint mergeN = 0;
        nint srcRaw = source is null ? 0 : source.DangerousGetHandle();
        int rc;
        if (sep is null)
        {
            if (input is null)
            {
                rc = NativeMethods.gt_train_bpe(inputTag, null, 0, srcRaw, vocabSizeNative,
                    sp.Ptrs, sp.Lens, sp.Count, tieBreaking, null, 0, 0,
                    &vids, &vptrs, &vlens, &vocabN, &mleft, &mleftLens, &mright, &mrightLens, &mergeN);
            }
            else
            {
                fixed (byte* ip = input)
                    rc = NativeMethods.gt_train_bpe(inputTag, ip, (nuint)input.Length, srcRaw, vocabSizeNative,
                        sp.Ptrs, sp.Lens, sp.Count, tieBreaking, null, 0, 0,
                        &vids, &vptrs, &vlens, &vocabN, &mleft, &mleftLens, &mright, &mrightLens, &mergeN);
            }
        }
        else
        {
            fixed (byte* sxp = sep)
            {
                if (input is null)
                {
                    rc = NativeMethods.gt_train_bpe(inputTag, null, 0, srcRaw, vocabSizeNative,
                        sp.Ptrs, sp.Lens, sp.Count, tieBreaking, sxp, (nuint)sep.Length, 1,
                        &vids, &vptrs, &vlens, &vocabN, &mleft, &mleftLens, &mright, &mrightLens, &mergeN);
                }
                else
                {
                    fixed (byte* ip = input)
                        rc = NativeMethods.gt_train_bpe(inputTag, ip, (nuint)input.Length, srcRaw, vocabSizeNative,
                            sp.Ptrs, sp.Lens, sp.Count, tieBreaking, sxp, (nuint)sep.Length, 1,
                            &vids, &vptrs, &vlens, &vocabN, &mleft, &mleftLens, &mright, &mrightLens, &mergeN);
                }
            }
        }
        GC.KeepAlive(source);
        Check(rc);
        var ids = TakeU32(vids, vocabN);
        var nVocab = checked((int)vocabN);
        var vocab = new Dictionary<uint, byte[]>(nVocab);
        var ptrSpan = new ReadOnlySpan<nint>(vptrs, nVocab);
        var lenSpan = new ReadOnlySpan<nuint>(vlens, nVocab);
        var vocabDone = 0;
        try
        {
            for (; vocabDone < nVocab; vocabDone++)
            {
                var blen = checked((int)lenSpan[vocabDone]);
                var bytes = new byte[blen];
                new ReadOnlySpan<byte>((byte*)ptrSpan[vocabDone], blen).CopyTo(bytes);
                vocab[ids[vocabDone]] = bytes;
                NativeMethods.gt_free(ptrSpan[vocabDone]);
            }
        }
        catch
        {
            // Mid-loop managed OOM: free the failing entry plus all remaining
            // native vocab buffers. Entries before vocabDone were already freed
            // in-loop, so [vocabDone, nVocab) is exactly the owned remainder.
            for (var j = vocabDone; j < nVocab; j++)
                NativeMethods.gt_free(ptrSpan[j]);
            throw;
        }
        finally
        {
            NativeMethods.gt_free((nint)vptrs);
            NativeMethods.gt_free((nint)vlens);
        }
        var nMerge = checked((int)mergeN);
        var merges = new List<(byte[], byte[])>(nMerge);
        var ll = new ReadOnlySpan<nint>(mleft, nMerge);
        var lls = new ReadOnlySpan<nuint>(mleftLens, nMerge);
        var rl = new ReadOnlySpan<nint>(mright, nMerge);
        var rls = new ReadOnlySpan<nuint>(mrightLens, nMerge);
        var mergeDone = 0;
        try
        {
            for (; mergeDone < nMerge; mergeDone++)
            {
                var llen = checked((int)lls[mergeDone]);
                var rlen = checked((int)rls[mergeDone]);
                var l = new byte[llen];
                var r = new byte[rlen];
                new ReadOnlySpan<byte>((byte*)ll[mergeDone], llen).CopyTo(l);
                new ReadOnlySpan<byte>((byte*)rl[mergeDone], rlen).CopyTo(r);
                merges.Add((l, r));
                NativeMethods.gt_free(ll[mergeDone]);
                NativeMethods.gt_free(rl[mergeDone]);
            }
        }
        catch
        {
            // Mid-loop managed OOM: free both halves of the failing entry plus
            // all remaining native merge buffers. Entries before mergeDone were
            // already freed in-loop.
            for (var j = mergeDone; j < nMerge; j++)
            {
                NativeMethods.gt_free(ll[j]);
                NativeMethods.gt_free(rl[j]);
            }
            throw;
        }
        finally
        {
            NativeMethods.gt_free((nint)mleft);
            NativeMethods.gt_free((nint)mleftLens);
            NativeMethods.gt_free((nint)mright);
            NativeMethods.gt_free((nint)mrightLens);
        }
        return (vocab, merges);
    }

    // -- Pretokenizer --------------------------------------------------------

    internal static PretokenIterHandle PretokenIterNew(byte[] bytes)
    {
        nint h;
        fixed (byte* p = bytes)
            Check(NativeMethods.gt_pretokenize_iter_new(p, (nuint)bytes.Length, out h));
        return WrapIter(h);
    }

    /// <returns>Next piece, or null at end.</returns>
    internal static byte[]? PretokenIterNext(PretokenIterHandle iter)
    {
        byte* p = null; nuint len = 0;
        var rc = NativeMethods.gt_pretokenize_iter_next(iter, &p, &len);
        if (rc == 0) return null; // exhausted
        if (rc != 1) Check(rc); // 1 = piece available; -1/7 = error codes via Check
        var n = checked((int)len);
        var out_ = new byte[n];
        new ReadOnlySpan<byte>(p, n).CopyTo(out_);
        return out_;
    }

    internal static IReadOnlyList<(byte[] Pretoken, long Count)> PretokenizedCounts(byte[] bytes, byte[]? sep)
    {
        byte** keys = null; nuint* keysLens = null; nuint* counts = null; nuint n = 0;
        int rc;
        fixed (byte* p = bytes)
        {
            if (sep is null)
            {
                rc = NativeMethods.gt_pretokenized_counts(
                    p, (nuint)bytes.Length, null, 0, 0, &keys, &keysLens, &counts, &n);
            }
            else
            {
                fixed (byte* sp2 = sep)
                    rc = NativeMethods.gt_pretokenized_counts(
                        p, (nuint)bytes.Length, sp2, (nuint)sep.Length, 1, &keys, &keysLens, &counts, &n);
            }
        }
        Check(rc);
        var nn = checked((int)n);
        var list = new List<(byte[], long)>(nn);
        var k = new ReadOnlySpan<nint>(keys, nn);
        var kl = new ReadOnlySpan<nuint>(keysLens, nn);
        var c = new ReadOnlySpan<nuint>(counts, nn);
        var done = 0;
        try
        {
            for (; done < nn; done++)
            {
                var klen = checked((int)kl[done]);
                var bytesCopy = new byte[klen];
                new ReadOnlySpan<byte>((byte*)k[done], klen).CopyTo(bytesCopy);
                var cnt = checked((long)c[done]);
                list.Add((bytesCopy, cnt));
                NativeMethods.gt_free(k[done]);
            }
        }
        catch
        {
            // Mid-loop managed OOM: free the failing key plus all remaining
            // native keys. Entries before done were already freed in-loop.
            for (var j = done; j < nn; j++)
                NativeMethods.gt_free(k[j]);
            throw;
        }
        finally
        {
            NativeMethods.gt_free((nint)keys);
            NativeMethods.gt_free((nint)keysLens);
            NativeMethods.gt_free((nint)counts);
        }
        return list;
    }

    // -- Hub / cache ----------------------------------------------------------

    internal static byte[] HubFile(byte[] repo, byte[] filename, int repoType, byte[] revision)
    {
        byte* op = null; nuint ol = 0;
        fixed (byte* rp = repo)
        fixed (byte* fp = filename)
        fixed (byte* vp = revision)
            Check(NativeMethods.gt_hub_file(
                rp, (nuint)repo.Length, fp, (nuint)filename.Length, repoType,
                vp, (nuint)revision.Length, &op, &ol));
        return TakeBytes(op, ol);
    }

    internal static bool LooksLikeRepoId(byte[] s)
    {
        fixed (byte* p = s)
            return NativeMethods.gt_looks_like_repo_id(p, (nuint)s.Length) != 0;
    }

    internal static byte[]? GetHfToken()
    {
        byte* bp = null; nuint bl = 0;
        var rc = NativeMethods.gt_get_hf_token(&bp, &bl);
        // Native protocol: 1 + malloc'd bytes when a token is configured,
        // 0 with null/0 outputs when absent. Anything else is an error code.
        if (rc == 0) return null;
        if (rc == 1)
        {
            if (bp is null || bl == 0) return null;
            return TakeBytes(bp, bl);
        }
        Check(rc);
        return null;
    }

    internal static void SetMaxCacheBytes(nuint? value)
    {
        if (value.HasValue) NativeMethods.gt_set_max_cache_bytes(1, value.Value);
        else NativeMethods.gt_set_max_cache_bytes(0, 0);
    }

    internal static nuint? GetMaxCacheBytes()
    {
        int has = 0; nuint value = 0;
        NativeMethods.gt_get_max_cache_bytes(&has, &value);
        return has != 0 ? value : null;
    }
}
