using System.Text;

namespace Gigatoken;

/// <summary>Tiktoken compatibility wrapper (mirrors <c>TiktokenCompat</c>).</summary>
/// <remarks>
/// The forbid-matcher cache never disposes entries on eviction: a matcher
/// borrowed by a concurrent encode may still be in use outside the cache lock,
/// so reclamation is left to SafeHandle finalization. Callers must not dispose
/// a matcher obtained from this cache. All cache access is lock-guarded for
/// pool-parallel encode.
/// </remarks>
public sealed class TiktokenCompat
{
    private readonly Tokenizer _tokenizer;
    private readonly Dictionary<string, uint> _specialTokens;
    private readonly HashSet<string> _specialsSet;
    private readonly HashSet<uint> _specialIds;
    private readonly Dictionary<string, WrapTruncateOptions> _matcherCache = new();
    private readonly object _cacheLock = new();
    private byte[][]? _byteValuesCache;
    private Dictionary<uint, byte[]>? _bytesByIdCache;
    private Dictionary<byte[], uint>? _idByBytesCache;

    private sealed class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public static readonly ByteArrayComparer Instance = new();
        public bool Equals(byte[]? x, byte[]? y) =>
            ReferenceEquals(x, y) || (x is not null && y is not null && x.AsSpan().SequenceEqual(y));
        public int GetHashCode(byte[] obj)
        {
            var h = new HashCode();
            foreach (var b in obj) h.Add(b);
            return h.ToHashCode();
        }
    }

    /// <summary>Builds the wrapper.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="tokenizer"/> is null.</exception>
    public TiktokenCompat(Tokenizer tokenizer, string name = "gigatoken")
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(name);
        _tokenizer = tokenizer;
        Name = name;
        _specialTokens = tokenizer.SpecialTokensMap();
        _specialsSet = new HashSet<string>(_specialTokens.Keys);
        _specialIds = new HashSet<uint>(_specialTokens.Values);
    }

    /// <summary>Encoding name.</summary>
    public string Name { get; }

    /// <summary>End-of-text token id.</summary>
    public uint EotToken =>
        _specialTokens.TryGetValue("<|endoftext|>", out var id) ? id
        : throw new KeyNotFoundException("No <|endoftext|> special token.");

    /// <summary>Vocabulary size (max id + 1).</summary>
    public int NVocab => _tokenizer.VocabSize;

    /// <summary>Maximum token value (vocab size - 1).</summary>
    public int MaxTokenValue => NVocab - 1;

    /// <summary>Special token strings.</summary>
    public ISet<string> SpecialTokensSet => new HashSet<string>(_specialsSet);

    /// <summary>Underlying tokenizer.</summary>
    public Tokenizer Tokenizer => _tokenizer;

    /// <summary>Whether <paramref name="id"/> is a special token.</summary>
    public bool IsSpecialToken(uint id) => _specialIds.Contains(id);

    /// <summary>Alias for <see cref="IsSpecialToken(uint)"/> (port.md §3 name).</summary>
    public bool IsSpecial(uint id) => IsSpecialToken(id);

    private static HashSet<string> ResolveAllowed(
        string allowedSpecial, ICollection<string>? allowedSet, HashSet<string> specialsSet)
    {
        ArgumentNullException.ThrowIfNull(allowedSpecial);
        // An explicit set always wins (mirrors the single upstream arg being a
        // collection); the string is only consulted when no set is given.
        if (allowedSet is not null) return [.. allowedSet];
        if (allowedSpecial == "all") return [.. specialsSet];
        if (allowedSpecial == "none") return [];
        return [allowedSpecial];
    }

    private static HashSet<string> ResolveDisallowed(
        string disallowedSpecial, ICollection<string>? disallowedSet,
        HashSet<string> specialsSet, HashSet<string> allowed)
    {
        ArgumentNullException.ThrowIfNull(disallowedSpecial);
        // An explicit set always wins; "all" no longer discards it.
        if (disallowedSet is not null) return [.. disallowedSet];
        if (disallowedSpecial == "all") return new HashSet<string>(specialsSet.Except(allowed));
        if (disallowedSpecial == "none") return [];
        return [disallowedSpecial];
    }

    private (WrapTruncateOptions? Wrap, string[] Patterns, HashSet<string> Disallowed) ForbidMatcher(
        string allowedSpecial, ICollection<string>? allowedSet,
        string disallowedSpecial, ICollection<string>? disallowedSet)
    {
        var allowed = ResolveAllowed(allowedSpecial, allowedSet, _specialsSet);
        var disallowed = ResolveDisallowed(disallowedSpecial, disallowedSet, _specialsSet, allowed);
        var scan = _specialsSet
            .Where(s => disallowed.Contains(s) || !allowed.Contains(s))
            .OrderBy(s => s, StringComparer.Ordinal).ToArray();
        if (scan.Length == 0) return (null, [], disallowed);
        // Length-prefixed framing: unambiguous unlike bare "\0" joins, so two
        // distinct scans (even with NUL-bearing specials) never share a key.
        var key = scan.Length + "\0" + string.Concat(scan.Select(s => s.Length + "\0" + s));
        lock (_cacheLock)
        {
            if (_matcherCache.Count > 32)
            {
                // Never dispose here: a matcher borrowed by a concurrent encode
                // may still be in use outside the lock. Evicted entries are left
                // for SafeHandle finalization once all borrowers drop them.
                _matcherCache.Clear();
            }
            if (!_matcherCache.TryGetValue(key, out var opts))
            {
                opts = new WrapTruncateOptions { Forbid = new SubstringMatcher(scan) };
                _matcherCache[key] = opts;
            }
            return (opts, scan, disallowed);
        }
    }

    private List<uint[]> EncodeList(
        IReadOnlyList<byte[]> texts,
        string allowedSpecial, ICollection<string>? allowedSet,
        string disallowedSpecial, ICollection<string>? disallowedSet,
        bool? parallel)
    {
        var (wrap, patterns, disallowed) = ForbidMatcher(allowedSpecial, allowedSet, disallowedSpecial, disallowedSet);
        List<uint[]> rows;
        try
        {
            rows = wrap is null
                ? _tokenizer.EncodeBatch(texts, parallel).Select(r => r).ToList()
                : _tokenizer.EncodeBatchWrapped(texts, wrap, parallel).Select(r => r).ToList();
        }
        catch (SpecialTokenFoundException ex)
        {
            // A stale matcher could report an index outside this scan; map it to
            // a GigatokenException instead of leaking IndexOutOfRange.
            if (ex.MatchedIndices.Any(i => i < 0 || i >= patterns.Length))
                throw new GigatokenException(
                    "Native forbid-matcher returned an out-of-range special index.");
            var present = new HashSet<string>(ex.MatchedIndices.Select(i => patterns[i]));
            var bad = present.Intersect(disallowed).OrderBy(s => s, StringComparer.Ordinal).ToArray();
            if (bad.Length > 0)
                throw new ArgumentException(
                    $"Encountered text corresponding to disallowed special token '{bad[0]}'. " +
                    $"If you want this text to be encoded as a special token, pass it to allowedSpecial. " +
                    $"If you want this text to be encoded as normal text, disable the check for this token.");
            var ordinary = present.OrderBy(s => s, StringComparer.Ordinal).ToArray();
            var first = ordinary.Length > 0 ? ordinary[0] : "?";
            throw new NotSupportedException(
                $"gigatoken always recognizes special tokens while encoding, so '{first}' cannot be " +
                "encoded as ordinary text; pass it in allowedSpecial (to encode it as a special token) " +
                "or leave it disallowed (to reject it, tiktoken's default).");
        }
        return rows;
    }

    private Dictionary<uint, byte[]> BytesById()
    {
        lock (_cacheLock)
        {
            if (_bytesByIdCache is not null) return _bytesByIdCache;
            var entries = _tokenizer.VocabEntries;
            var dict = new Dictionary<uint, byte[]>(entries.Count);
            foreach (var (id, b) in entries) dict[id] = b;
            _bytesByIdCache = dict;
            return dict;
        }
    }

    private Dictionary<byte[], uint> IdByBytes()
    {
        lock (_cacheLock)
        {
            if (_idByBytesCache is not null) return _idByBytesCache;
            var byId = BytesById();
            var dict = new Dictionary<byte[], uint>(byId.Count, ByteArrayComparer.Instance);
            foreach (var (id, b) in byId) dict[b] = id;
            _idByBytesCache = dict;
            return dict;
        }
    }

    /// <summary>Encodes text, allowing all/no specials per arguments.</summary>
    public uint[] Encode(
        string text, string allowedSpecial = "none", ICollection<string>? allowedSpecialSet = null,
        string disallowedSpecial = "all", ICollection<string>? disallowedSpecialSet = null,
        bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        var rows = EncodeList([Encoding.UTF8.GetBytes(text)],
            allowedSpecial, allowedSpecialSet, disallowedSpecial, disallowedSpecialSet, parallel);
        return rows[0];
    }

    /// <summary>Encodes text with no special-token handling.</summary>
    public uint[] EncodeOrdinary(string text, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeList([Encoding.UTF8.GetBytes(text)], "none", null, "none", null, parallel)[0];
    }

    /// <summary>Encodes a batch.</summary>
    public List<uint[]> EncodeBatch(
        IEnumerable<string> texts, int numThreads = 8,
        string allowedSpecial = "none", ICollection<string>? allowedSpecialSet = null,
        string disallowedSpecial = "all", ICollection<string>? disallowedSpecialSet = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts as IList<string> ?? texts.ToList();
        var docs = new byte[list.Count][];
        for (var i = 0; i < list.Count; i++)
            docs[i] = Encoding.UTF8.GetBytes(list[i] ?? throw new ArgumentNullException($"{nameof(texts)}[{i}]"));
        return EncodeList(docs,
            allowedSpecial, allowedSpecialSet, disallowedSpecial, disallowedSpecialSet,
            numThreads == 1 ? false : null);
    }

    /// <summary>Encodes a batch with no special-token handling.</summary>
    public List<uint[]> EncodeOrdinaryBatch(IEnumerable<string> texts, int numThreads = 8)
    {
        ArgumentNullException.ThrowIfNull(texts);
        return EncodeBatch(texts, numThreads, "none", null, "none", null);
    }

    /// <summary>Encodes a single token's bytes.</summary>
    public uint EncodeSingleToken(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeSingleToken(Encoding.UTF8.GetBytes(text));
    }

    /// <summary>Encodes a single token's bytes.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="bytes"/> is null.</exception>
    /// <exception cref="KeyNotFoundException">The bytes are not a single token.</exception>
    public uint EncodeSingleToken(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (IdByBytes().TryGetValue(bytes, out var id)) return id;
        throw new KeyNotFoundException("Bytes do not correspond to a single token.");
    }

    /// <summary>Always throws (mirrors upstream <c>NotImplementedError</c>).</summary>
    public uint[] EncodeWithUnstable(string text) =>
        throw new NotSupportedException("encode_with_unstable is not supported.");

    /// <summary>Decodes ids to a string.</summary>
    public string Decode(uint[] tokens, string errors = "replace")
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(errors);
        return DecodeWithErrors(DecodeBytes(tokens), errors);
    }

    private static string DecodeWithErrors(byte[] bytes, string errors) =>
        errors switch
        {
            "strict" => new UTF8Encoding(false, true).GetString(bytes),
            "replace" => Encoding.UTF8.GetString(bytes),
            "ignore" => Encoding.GetEncoding("utf-8", EncoderFallback.ExceptionFallback, new DecoderReplacementFallback(string.Empty)).GetString(bytes),
            _ => throw new ArgumentException($"Unknown errors strategy '{errors}'.", nameof(errors)),
        };

    /// <summary>Decodes ids to bytes.</summary>
    public byte[] DecodeBytes(uint[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        return _tokenizer.Decode(tokens);
    }

    /// <summary>Decodes a batch.</summary>
    public List<string> DecodeBatch(IEnumerable<uint[]> batch, string errors = "replace", int numThreads = 8)
    {
        ArgumentNullException.ThrowIfNull(batch);
        _ = numThreads;
        return batch.Select(t => Decode((t ?? throw new ArgumentNullException(nameof(batch))).ToArray(), errors)).ToList();
    }

    /// <summary>Decodes a batch to bytes.</summary>
    public List<byte[]> DecodeBytesBatch(IEnumerable<uint[]> batch)
    {
        ArgumentNullException.ThrowIfNull(batch);
        return batch.Select(t => DecodeBytes((t ?? throw new ArgumentNullException(nameof(batch))).ToArray())).ToList();
    }

    /// <summary>Decodes a single token to bytes (returns a copy; mutating it cannot poison the cache).</summary>
    public byte[] DecodeSingleTokenBytes(uint token) =>
        BytesById().TryGetValue(token, out var bytes) ? (byte[])bytes.Clone()
        : throw new KeyNotFoundException($"Unknown token id {token}.");

    /// <summary>Decodes tokens to per-token bytes (each array is a copy).</summary>
    public List<byte[]> DecodeTokensBytes(IEnumerable<uint> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        var byId = BytesById();
        return tokens.Select(t => byId.TryGetValue(t, out var b) ? (byte[])b.Clone()
            : throw new KeyNotFoundException($"Unknown token id {t}.")).ToList();
    }

    /// <summary>Always throws (mirrors upstream <c>NotImplementedError</c>).</summary>
    public (string Text, int[] Offsets) DecodeWithOffsets(uint[] tokens) =>
        throw new NotSupportedException("decode_with_offsets is not supported.");

    /// <summary>Mergeable token byte values (each array is a copy).</summary>
    public List<byte[]> TokenByteValues()
    {
        lock (_cacheLock)
        {
            if (_byteValuesCache is not null) return _byteValuesCache.Select(b => (byte[])b.Clone()).ToList();
            var byId = BytesById();
            var list = byId
                .Where(kv => !_specialIds.Contains(kv.Key))
                .OrderBy(kv => kv.Key).Select(kv => kv.Value).ToArray();
            _byteValuesCache = list;
            return list.Select(b => (byte[])b.Clone()).ToList();
        }
    }
}
