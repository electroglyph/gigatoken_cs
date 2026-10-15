using System.Collections.Frozen;
using System.Text;
using Gigatoken.Interop;

namespace Gigatoken;

/// <summary>Hugging Face Hub download / cache access (mirrors <c>hub_file</c> / <c>looks_like_repo_id</c> / <c>get_hf_token</c>).</summary>
public static class Hub
{
    /// <summary>Resolves (or downloads) a hub file and returns its local path.</summary>
    public static string GetFile(
        string repoId, string filename = "tokenizer.json",
        RepoType repoType = RepoType.Model, string revision = "main")
    {
        try
        {
            var bytes = NativeBridge.HubFile(
                NativeBridge.Utf8Bytes(repoId), NativeBridge.Utf8Bytes(filename),
                (int)repoType, NativeBridge.Utf8Bytes(revision));
            return Encoding.UTF8.GetString(bytes);
        }
        catch (FileNotFoundException ex)
        {
            throw new HubFileNotFoundException(ex.Message, ex.FileName, ex);
        }
    }

    /// <summary>Whether <paramref name="name"/> looks like a hub repo id.</summary>
    public static bool LooksLikeRepoId(string name) =>
        NativeBridge.LooksLikeRepoId(NativeBridge.Utf8Bytes(name));

    /// <summary>Configured HF token, or null.</summary>
    public static string? GetToken()
    {
        var bytes = NativeBridge.GetHfToken();
        return bytes is null ? null : Encoding.UTF8.GetString(bytes);
    }
}

/// <summary>Global pretoken-cache budget (mirrors <c>set_max_cache_bytes</c>). 512 MiB default.</summary>
public static class EncodeCache
{
    /// <summary>Default per-worker cache budget in bytes (512 MiB, mirrors upstream).</summary>
    public const long DefaultBytes = 512L << 20;

    /// <summary>Sets the per-worker cache budget in bytes (null = unbounded).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is negative.</exception>
    public static void SetMaxBytes(long? maxBytes)
    {
        if (maxBytes.HasValue && maxBytes.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(maxBytes));
        NativeBridge.SetMaxCacheBytes(maxBytes.HasValue ? (nuint)maxBytes.Value : null);
    }

    /// <summary>Current per-worker cache budget in bytes (null = unbounded).</summary>
    public static long? GetMaxBytes()
    {
        var v = NativeBridge.GetMaxCacheBytes();
        if (!v.HasValue) return null;
        // Saturate past long.MaxValue (unreachable budgets) instead of wrapping negative.
        return (ulong)v.Value > (ulong)long.MaxValue ? long.MaxValue : (long)v.Value;
    }
}

/// <summary>BPE training (mirrors <c>train_bpe</c>).</summary>
public static class BpeTrainer
{
    private static byte[][] CheckedSpecials(IEnumerable<string> specialTokens)
    {
        ArgumentNullException.ThrowIfNull(specialTokens);
        return specialTokens.Select(s => NativeBridge.Utf8Bytes(s ?? throw new ArgumentNullException(nameof(specialTokens)))).ToArray();
    }

    private static void CheckedVocabSize(int vocabSize)
    {
        if (vocabSize < 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), "vocabSize must be non-negative.");
    }

    /// <summary>Trains on an in-memory corpus blob.</summary>
    /// <exception cref="ArgumentNullException">Any argument is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="vocabSize"/> is negative.</exception>
    public static (System.Collections.Frozen.FrozenDictionary<uint, byte[]> Vocab, IReadOnlyList<(byte[] Left, byte[] Right)> Merges) Train(
        byte[] corpus, int vocabSize, IEnumerable<string> specialTokens,
        TieBreaking tieBreaking = TieBreaking.HuggingFace, byte[]? separator = null)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        CheckedVocabSize(vocabSize);
        var specials = CheckedSpecials(specialTokens);
        var (vocab, merges) = NativeBridge.TrainBpe(0, corpus, null, vocabSize, specials, (int)tieBreaking, separator);
        return (vocab.ToFrozenDictionary(), merges);
    }

    /// <summary>Trains on a corpus file.</summary>
    /// <exception cref="ArgumentNullException">Any argument is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="vocabSize"/> is negative.</exception>
    public static (System.Collections.Frozen.FrozenDictionary<uint, byte[]> Vocab, IReadOnlyList<(byte[] Left, byte[] Right)> Merges) Train(
        string path, int vocabSize, IEnumerable<string> specialTokens,
        TieBreaking tieBreaking = TieBreaking.HuggingFace, byte[]? separator = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        CheckedVocabSize(vocabSize);
        var specials = CheckedSpecials(specialTokens);
        var input = NativeBridge.Utf8Bytes(Path.GetFullPath(path));
        var (vocab, merges) = NativeBridge.TrainBpe(1, input, null, vocabSize, specials, (int)tieBreaking, separator);
        return (vocab.ToFrozenDictionary(), merges);
    }

    /// <summary>Trains on a file source.</summary>
    /// <exception cref="ArgumentNullException">Any argument is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="vocabSize"/> is negative.</exception>
    public static (System.Collections.Frozen.FrozenDictionary<uint, byte[]> Vocab, IReadOnlyList<(byte[] Left, byte[] Right)> Merges) Train(
        IFileSource source, int vocabSize, IEnumerable<string> specialTokens,
        TieBreaking tieBreaking = TieBreaking.HuggingFace, byte[]? separator = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        CheckedVocabSize(vocabSize);
        var specials = CheckedSpecials(specialTokens);
        var (vocab, merges) = NativeBridge.TrainBpe(2, null, source.NativeHandle, vocabSize, specials, (int)tieBreaking, separator);
        return (vocab.ToFrozenDictionary(), merges);
    }
}

/// <summary>GPT-2 pretokenizer iteration and counting (mirrors <c>pretokenizer</c> / <c>pretokenized_counts</c>).</summary>
public static class Pretokenizer
{
    /// <summary>Iterates GPT-2 pretokens over <paramref name="utf8"/>.</summary>
    public static PretokenizerIteration Pretokenize(ReadOnlySpan<byte> utf8) =>
        new(NativeBridge.PretokenIterNew(utf8.ToArray()));

    /// <summary>Counts pretokens (optionally split on <paramref name="separator"/>).</summary>
    public static IReadOnlyList<(byte[] Pretoken, long Count)> CountPretokens(ReadOnlySpan<byte> utf8, byte[]? separator = null) =>
        NativeBridge.PretokenizedCounts(utf8.ToArray(), separator);
}

/// <summary>Owns a native pretokenizer iterator. Materialize with <see cref="ToList"/> for repeated enumeration.</summary>
/// <remarks>
/// Single-pass like the upstream iterator: each piece is yielded once; a
/// second enumeration continues where the first left off. Single-thread
/// affinity (concurrent <c>next</c> on one iterator is a native data race).
/// </remarks>
public sealed class PretokenizerIteration : IDisposable, IEnumerable<ReadOnlyMemory<byte>>
{
    private PretokenIterHandle? _native;
    private int _disposed;

    internal PretokenizerIteration(PretokenIterHandle native) => _native = native;

    /// <summary>Whether the iterator is exhausted (a full pass completed).</summary>
    public bool IsExhausted { get; private set; }

    /// <summary>Returns all remaining pieces.</summary>
    public List<ReadOnlyMemory<byte>> ToList()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var list = new List<ReadOnlyMemory<byte>>();
        byte[]? next;
        while ((next = NativeBridge.PretokenIterNext(_native!)) is not null)
            list.Add(next);
        IsExhausted = true;
        return list;
    }

    /// <summary>Enumerates remaining pieces (single pass).</summary>
    public IEnumerator<ReadOnlyMemory<byte>> GetEnumerator()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        byte[]? next;
        while ((next = NativeBridge.PretokenIterNext(_native!)) is not null)
            yield return next;
        IsExhausted = true;
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Releases the native iterator.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) { _native?.Dispose(); _native = null; }
        GC.SuppressFinalize(this);
    }
}
