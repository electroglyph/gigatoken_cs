namespace Gigatoken;

/// <summary>Padding / truncation options for <c>EncodeBatchPadded</c> (mirrors <c>PadTruncate</c>).</summary>
/// <param name="PadId">Token id used for padding.</param>
/// <param name="MaxLength">Optional maximum row length (content + prefix + suffix). Must be non-negative when set.</param>
/// <param name="PadToMaxLength">Pad every row to <paramref name="MaxLength"/> (requires it).</param>
/// <param name="Truncate">Truncate rows longer than <paramref name="MaxLength"/> (requires it).</param>
/// <param name="PadLeft">Pad on the left instead of the right.</param>
/// <param name="TruncateLeft">Truncate on the left instead of the right.</param>
/// <param name="Prefix">Token ids prepended to every row (snapshotted per call).</param>
/// <param name="Suffix">Token ids appended to every row (snapshotted per call).</param>
public readonly record struct PadTruncateOptions(
    uint PadId,
    int? MaxLength = null,
    bool PadToMaxLength = false,
    bool Truncate = false,
    bool PadLeft = false,
    bool TruncateLeft = false,
    uint[]? Prefix = null,
    uint[]? Suffix = null);

/// <summary>Per-call wrap options for batch encode (mirrors <c>_WrapTruncate</c>).</summary>
/// <remarks>
/// <c>Forbid</c> is borrower-owned: callers retain ownership of the matcher
/// they pass in (this type never disposes it). The <c>TiktokenCompat</c>
/// forbid cache owns its matchers and disposes them on eviction.
/// </remarks>
public sealed class WrapTruncateOptions : IDisposable
{
    /// <summary>Token ids prepended to every row.</summary>
    public uint[] Prefix { get; init; } = [];
    /// <summary>Token ids appended to every row.</summary>
    public uint[] Suffix { get; init; } = [];
    /// <summary>Optional per-row content cap. Must be non-negative when set.</summary>
    public int? MaxTokens { get; init; }
    /// <summary>Truncate content on the left.</summary>
    public bool TruncateLeft { get; init; }
    /// <summary>Optional forbidden-pattern matcher (borrowed, not owned).</summary>
    public SubstringMatcher? Forbid { get; init; }

    /// <summary>Builds the native options handle (owns it; dispose with the call).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><c>MaxTokens</c> is negative.</exception>
    internal Interop.WrapOptsHandle BuildNative()
    {
        if (MaxTokens.HasValue && MaxTokens.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxTokens), "MaxTokens must be non-negative.");
        // Snapshot arrays so caller mutation during the pinned native call cannot race tokenization.
        var prefix = Prefix is null ? [] : Prefix.ToArray();
        var suffix = Suffix is null ? [] : Suffix.ToArray();
        return Interop.NativeBridge.WrapOptsNew(prefix, suffix,
            MaxTokens.HasValue ? (nuint)MaxTokens.Value : null,
            TruncateLeft, Forbid?.Native);
    }

    /// <summary>Disposes managed state (native handles are per-call and disposed by callers; <c>Forbid</c> is borrowed and never disposed here).</summary>
    public void Dispose() { }
}

/// <summary>Global library options.</summary>
public static class GigatokenOptions
{
    // Stored as int for atomic cross-thread access (0 = null, 1 = false, 2 = true);
    // Nullable<bool> reads/writes would otherwise tear.
    private static int _defaultParallel;

    /// <summary>
    /// Default parallelism for encode calls. Null = automatic (parallel via the
    /// native worker pool). An explicit value wins as the default; an explicit
    /// per-call argument always wins over this.
    /// </summary>
    public static bool? DefaultParallel
    {
        get => Volatile.Read(ref _defaultParallel) switch { 1 => false, 2 => true, _ => null };
        set => Volatile.Write(ref _defaultParallel, value.HasValue ? (value.Value ? 2 : 1) : 0);
    }
}
