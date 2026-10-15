using Gigatoken.Interop;

namespace Gigatoken;

/// <summary>Aho-Corasick substring matcher (mirrors <c>_SubstringMatcher</c>). Used by <see cref="TiktokenCompat"/>.</summary>
/// <remarks>
/// Empty and duplicate patterns are accepted, matching upstream (Aho-Corasick
/// semantics: an empty pattern matches every text; duplicates yield one hit
/// index per entry). Null collections or null elements throw.
/// Single-thread affinity for concurrent <c>Present</c> on one instance is
/// safe (native scan holds no runtime lock); concurrent <c>Dispose</c> is
/// idempotent via interlocked guard.
/// </remarks>
public sealed class SubstringMatcher : IDisposable
{
    private MatcherHandle? _native;
    private int _disposed;

    /// <summary>Builds a matcher over UTF-8 patterns.</summary>
    /// <exception cref="ArgumentNullException">Any pattern or the collection is null.</exception>
    public SubstringMatcher(IEnumerable<string> patterns)
    {
        ArgumentNullException.ThrowIfNull(patterns);
        var list = patterns.ToList();
        for (var i = 0; i < list.Count; i++)
            ArgumentNullException.ThrowIfNull(list[i], nameof(patterns));
        var bufs = list.Select(NativeBridge.Utf8Bytes).ToArray();
        Count = bufs.Length;
        _native = NativeBridge.MatcherNew(bufs);
    }

    /// <summary>Number of patterns (mirrors upstream <c>__len__</c>).</summary>
    public int Count { get; }

    internal MatcherHandle Native => _native ?? throw new ObjectDisposedException(nameof(SubstringMatcher));

    /// <summary>Returns the sorted indices of patterns occurring in <paramref name="text"/>.</summary>
    public int[] Present(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return NativeBridge.MatcherPresent(Native, NativeBridge.Utf8Bytes(text));
    }

    /// <summary>Releases the native matcher.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _native?.Dispose();
            _native = null;
        }
        GC.SuppressFinalize(this);
    }
}
