using System.Collections.Frozen;
using Gigatoken.Interop;

namespace Gigatoken;

internal static class BackendOps
{
    internal static bool ResolveParallel(bool? parallel) =>
        parallel ?? GigatokenOptions.DefaultParallel ?? true;

    internal static uint[] Encode(TokenizerHandle h, byte[] input) =>
        NativeBridge.Encode(h, input);

    internal static uint[][] EncodeBatch(TokenizerHandle h, IReadOnlyList<byte[]> inputs, bool? parallel, WrapOptsHandle? wrap = null) =>
        NativeBridge.EncodeBatch(h, inputs, ResolveParallel(parallel), wrap);

    internal static uint[][] EncodeRegions(TokenizerHandle h, IReadOnlyList<byte[]> buffers, byte[]? sep, bool? parallel, WrapOptsHandle? wrap = null) =>
        NativeBridge.EncodeRegions(h, buffers, sep, ResolveParallel(parallel), wrap);

    internal static (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(
        TokenizerHandle h, IReadOnlyList<byte[]> inputs, bool? parallel, PadTruncateOptions options)
    {
        if (options.MaxLength.HasValue && options.MaxLength.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxLength must be non-negative.");
        if (options.PadToMaxLength && !options.MaxLength.HasValue)
            throw new ArgumentException("PadToMaxLength requires MaxLength.", nameof(options));
        if (options.Truncate && !options.MaxLength.HasValue)
            throw new ArgumentException("Truncate requires MaxLength.", nameof(options));
        // Snapshot prefix/suffix so caller mutation cannot race the pinned native call.
        var prefix = options.Prefix is null ? [] : options.Prefix.ToArray();
        var suffix = options.Suffix is null ? [] : options.Suffix.ToArray();
        using var pad = NativeBridge.PadOptsNew(
            options.PadId,
            options.MaxLength.HasValue ? (nuint)options.MaxLength.Value : null,
            options.PadToMaxLength, options.Truncate, options.PadLeft, options.TruncateLeft,
            prefix, suffix);
        return NativeBridge.EncodeBatchPadded(h, inputs, ResolveParallel(parallel), pad);
    }

    internal static uint[][] EncodeFiles(TokenizerHandle h, SourceHandle source, bool? parallel) =>
        NativeBridge.EncodeFiles(h, source, ResolveParallel(parallel));

    internal static int VocabSize(TokenizerHandle h) => NativeBridge.VocabSize(h);

    internal static FrozenDictionary<uint, byte[]> Vocab(TokenizerHandle h)
    {
        var entries = NativeBridge.VocabEntries(h);
        var dict = new Dictionary<uint, byte[]>(entries.Count);
        foreach (var (id, bytes) in entries) dict[id] = bytes;
        return dict.ToFrozenDictionary();
    }

    internal static IReadOnlyList<(uint Id, byte[] Bytes)> VocabEntries(TokenizerHandle h) =>
        NativeBridge.VocabEntries(h);

    internal static IReadOnlyList<(byte[] Left, byte[] Right)> Merges(TokenizerHandle h) =>
        NativeBridge.MergeEntries(h);

    internal static int CacheEntries(TokenizerHandle h) => NativeBridge.CacheEntries(h);
}

/// <summary>Low-level BPE backend (mirrors <c>BPETokenizer</c>).</summary>
public sealed class BpeTokenizer : IDisposable
{
    internal TokenizerHandle Native { get; private set; }
    private int _disposed;

    internal BpeTokenizer(TokenizerHandle native) => Native = native;

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Relinquishes native ownership (for <see cref="Tokenizer"/> wrapping).</summary>
    internal TokenizerHandle TakeNative()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        Interlocked.Exchange(ref _disposed, 1);
        GC.SuppressFinalize(this);
        return Native;
    }

    /// <summary>Loads a <c>.tiktoken</c> rank file.</summary>
    public static BpeTokenizer FromTiktoken(string path, string pretokenizer, IReadOnlyDictionary<string, uint>? specialTokens = null)
    {
        var keys = (specialTokens ?? new Dictionary<string, uint>())
            .Select(kv => NativeBridge.Utf8Bytes(kv.Key)).ToArray();
        var vals = (specialTokens ?? new Dictionary<string, uint>()).Values.ToArray();
        return new BpeTokenizer(NativeBridge.BpeFromTiktoken(
            NativeBridge.Utf8Bytes(Path.GetFullPath(path)),
            NativeBridge.Utf8Bytes(pretokenizer), keys, vals));
    }

    /// <summary>Loads a BPE <c>tokenizer.json</c> file.</summary>
    public static BpeTokenizer FromHf(string path) =>
        new(NativeBridge.BpeFromHf(NativeBridge.Utf8Bytes(Path.GetFullPath(path))));

    /// <summary>Encodes UTF-8 text.</summary>
    public uint[] Encode(string text)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);
        return BackendOps.Encode(Native, NativeBridge.Utf8Bytes(text));
    }

    /// <summary>Encodes raw UTF-8 bytes.</summary>
    public uint[] Encode(ReadOnlySpan<byte> utf8)
    {
        ThrowIfDisposed();
        return BackendOps.Encode(Native, utf8.ToArray());
    }

    /// <summary>Encodes a batch of strings.</summary>
    public uint[][] EncodeBatch(IEnumerable<string> texts, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(texts);
        return BackendOps.EncodeBatch(Native, texts.Select(NativeBridge.Utf8Bytes).ToArray(), parallel);
    }

    /// <summary>Encodes a batch of raw UTF-8 documents.</summary>
    public uint[][] EncodeBatch(IEnumerable<byte[]> byteDocs, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatch(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel);
    }

    /// <summary>Encodes in-memory buffers (Rust splits on the separator).</summary>
    public uint[][] EncodeBatch(BytesSource source, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeRegions(Native, source.Buffers, source.Separator, parallel);
    }

    /// <summary>Encodes files described by <paramref name="source"/>.</summary>
    public uint[][] EncodeFiles(IFileSource source, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeFiles(Native, source.NativeHandle, parallel);
    }

    /// <summary>Encodes a single path (format defaults by extension).</summary>
    public uint[][] EncodeFiles(string path, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(path);
        var bufs = new[] { NativeBridge.Utf8Bytes(Path.GetFullPath(path)) };
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Encodes several paths.</summary>
    public uint[][] EncodeFiles(IEnumerable<string> paths, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(paths);
        var bufs = paths.Select(p => NativeBridge.Utf8Bytes(Path.GetFullPath(p))).ToArray();
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Encodes with padding/truncation into a dense matrix plus row lengths.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<string> texts, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(texts);
        return BackendOps.EncodeBatchPadded(Native, texts.Select(NativeBridge.Utf8Bytes).ToArray(), parallel, options);
    }

    /// <summary>Encodes byte documents with padding/truncation.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<byte[]> byteDocs, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatchPadded(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel, options);
    }

    /// <summary>Encodes in-memory buffers with padding/truncation (managed fallback: encode then pad).</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(BytesSource source, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        var rows = EncodeBatch(source, parallel);
        return Tokenizer.PadRows(rows, options);
    }

    /// <summary>Decodes ids to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<uint> ids)
    {
        ThrowIfDisposed();
        return NativeBridge.Decode(Native, ids.ToArray());
    }

    /// <summary>Decodes ids (range-validated) to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<int> ids)
    {
        ThrowIfDisposed();
        var conv = new uint[ids.Length];
        for (var i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0) throw new OverflowException($"Negative token id at index {i}: {ids[i]}.");
            conv[i] = (uint)ids[i];
        }
        return NativeBridge.Decode(Native, conv);
    }

    /// <summary>Decodes ids (range-validated) to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<long> ids)
    {
        ThrowIfDisposed();
        var conv = new uint[ids.Length];
        for (var i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] > uint.MaxValue)
                throw new OverflowException($"Token id out of range at index {i}: {ids[i]}.");
            conv[i] = (uint)ids[i];
        }
        return NativeBridge.Decode(Native, conv);
    }

    /// <summary>Decodes ids to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<uint> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<int> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<long> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>BPE backend kind.</summary>
    public TokenizerKind Kind => TokenizerKind.Bpe;

    /// <summary>Number of vocabulary entries (including added tokens).</summary>
    public int VocabSize { get { ThrowIfDisposed(); return BackendOps.VocabSize(Native); } }
    /// <summary>Fresh id→bytes lookup.</summary>
    public FrozenDictionary<uint, byte[]> Vocab { get { ThrowIfDisposed(); return BackendOps.Vocab(Native); } }
    /// <summary>Vocabulary entries in ID order.</summary>
    public IReadOnlyList<(uint Id, byte[] Bytes)> VocabEntries { get { ThrowIfDisposed(); return BackendOps.VocabEntries(Native); } }
    /// <summary>Merges in merge-priority order.</summary>
    public IReadOnlyList<(byte[] Left, byte[] Right)> Merges { get { ThrowIfDisposed(); return BackendOps.Merges(Native); } }
    /// <summary>Single-doc encode-path cache entry count.</summary>
    public int CacheEntries { get { ThrowIfDisposed(); return BackendOps.CacheEntries(Native); } }

    /// <summary>Releases the native tokenizer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) Native.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Low-level SentencePiece backend (mirrors <c>SentencePieceTokenizer</c>).</summary>
public sealed class SentencePieceTokenizer : IDisposable
{
    internal TokenizerHandle Native { get; private set; }
    private int _disposed;

    internal SentencePieceTokenizer(TokenizerHandle native) => Native = native;

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Relinquishes native ownership (for <see cref="Tokenizer"/> wrapping).</summary>
    internal TokenizerHandle TakeNative()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        Interlocked.Exchange(ref _disposed, 1);
        GC.SuppressFinalize(this);
        return Native;
    }

    /// <summary>Loads a SentencePiece <c>tokenizer.json</c> file.</summary>
    public static SentencePieceTokenizer FromHf(string path) =>
        new(NativeBridge.SpFromHf(NativeBridge.Utf8Bytes(Path.GetFullPath(path))));

    /// <summary>Encodes UTF-8 text.</summary>
    public uint[] Encode(string text)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);
        return BackendOps.Encode(Native, NativeBridge.Utf8Bytes(text));
    }

    /// <summary>Encodes raw UTF-8 bytes.</summary>
    public uint[] Encode(ReadOnlySpan<byte> utf8)
    {
        ThrowIfDisposed();
        return BackendOps.Encode(Native, utf8.ToArray());
    }

    /// <summary>Encodes without normalization (SentencePiece-only extra).</summary>
    public uint[] EncodeNoNormalize(string text)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);
        return NativeBridge.SpEncodeNoNormalize(Native, NativeBridge.Utf8Bytes(text));
    }

    /// <summary>Encodes a batch of strings.</summary>
    public uint[][] EncodeBatch(IEnumerable<string> texts, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(texts);
        return BackendOps.EncodeBatch(Native, texts.Select(NativeBridge.Utf8Bytes).ToArray(), parallel);
    }

    /// <summary>Encodes a batch of raw UTF-8 documents.</summary>
    public uint[][] EncodeBatch(IEnumerable<byte[]> byteDocs, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatch(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel);
    }

    /// <summary>Encodes in-memory buffers (Rust splits on the separator).</summary>
    public uint[][] EncodeBatch(BytesSource source, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeRegions(Native, source.Buffers, source.Separator, parallel);
    }

    /// <summary>Encodes files described by <paramref name="source"/>.</summary>
    public uint[][] EncodeFiles(IFileSource source, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeFiles(Native, source.NativeHandle, parallel);
    }

    /// <summary>Encodes a single path (format defaults by extension).</summary>
    public uint[][] EncodeFiles(string path, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(path);
        var bufs = new[] { NativeBridge.Utf8Bytes(Path.GetFullPath(path)) };
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Encodes several paths.</summary>
    public uint[][] EncodeFiles(IEnumerable<string> paths, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(paths);
        var bufs = paths.Select(p => NativeBridge.Utf8Bytes(Path.GetFullPath(p))).ToArray();
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Encodes with padding/truncation into a dense matrix plus row lengths.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<string> texts, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(texts);
        return BackendOps.EncodeBatchPadded(Native, texts.Select(NativeBridge.Utf8Bytes).ToArray(), parallel, options);
    }

    /// <summary>Encodes byte documents with padding/truncation.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<byte[]> byteDocs, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatchPadded(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel, options);
    }

    /// <summary>Encodes in-memory buffers with padding/truncation (managed fallback: encode then pad).</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(BytesSource source, PadTruncateOptions options, bool? parallel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);
        var rows = EncodeBatch(source, parallel);
        return Tokenizer.PadRows(rows, options);
    }

    /// <summary>Decodes ids to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<uint> ids)
    {
        ThrowIfDisposed();
        return NativeBridge.Decode(Native, ids.ToArray());
    }

    /// <summary>Decodes ids (range-validated) to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<int> ids)
    {
        ThrowIfDisposed();
        var conv = new uint[ids.Length];
        for (var i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0) throw new OverflowException($"Negative token id at index {i}: {ids[i]}.");
            conv[i] = (uint)ids[i];
        }
        return NativeBridge.Decode(Native, conv);
    }

    /// <summary>Decodes ids (range-validated) to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<long> ids)
    {
        ThrowIfDisposed();
        var conv = new uint[ids.Length];
        for (var i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] > uint.MaxValue)
                throw new OverflowException($"Token id out of range at index {i}: {ids[i]}.");
            conv[i] = (uint)ids[i];
        }
        return NativeBridge.Decode(Native, conv);
    }

    /// <summary>Decodes ids to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<uint> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<int> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<long> ids) =>
        System.Text.Encoding.UTF8.GetString(Decode(ids));

    /// <summary>SentencePiece backend kind.</summary>
    public TokenizerKind Kind => TokenizerKind.SentencePiece;

    /// <summary>Number of vocabulary entries (including added tokens).</summary>
    public int VocabSize { get { ThrowIfDisposed(); return BackendOps.VocabSize(Native); } }
    /// <summary>Fresh id→bytes lookup.</summary>
    public FrozenDictionary<uint, byte[]> Vocab { get { ThrowIfDisposed(); return BackendOps.Vocab(Native); } }
    /// <summary>Vocabulary entries in ID order.</summary>
    public IReadOnlyList<(uint Id, byte[] Bytes)> VocabEntries { get { ThrowIfDisposed(); return BackendOps.VocabEntries(Native); } }
    /// <summary>Merges in merge-priority order.</summary>
    public IReadOnlyList<(byte[] Left, byte[] Right)> Merges { get { ThrowIfDisposed(); return BackendOps.Merges(Native); } }
    /// <summary>Single-doc encode-path cache entry count.</summary>
    public int CacheEntries { get { ThrowIfDisposed(); return BackendOps.CacheEntries(Native); } }

    /// <summary>Releases the native tokenizer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) Native.Dispose();
        GC.SuppressFinalize(this);
    }
}
