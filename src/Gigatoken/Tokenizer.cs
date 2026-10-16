using System.Collections.Frozen;
using System.Text;
using System.Text.Json;
using Gigatoken.Interop;

namespace Gigatoken;

/// <summary>
/// Unified tokenizer facade (mirrors <c>gigatoken.Tokenizer</c>).
/// Retains the source <c>tokenizer.json</c> bytes (like Python <c>_hf_json</c>);
/// <see cref="AsHf"/> throws <see cref="InvalidOperationException"/> when there is none.
/// </summary>
public sealed class Tokenizer : IDisposable
{
    private sealed class SharedBackend(TokenizerHandle handle)
    {
        internal TokenizerHandle Handle = handle;
        private int _refs = 1;
        internal void AddRef()
        {
            int observed;
            do
            {
                observed = Volatile.Read(ref _refs);
                if (observed == 0)
                    throw new ObjectDisposedException(nameof(Tokenizer));
            } while (Interlocked.CompareExchange(ref _refs, observed + 1, observed) != observed);
        }
        internal void Release()
        {
            if (Interlocked.Decrement(ref _refs) == 0) Handle.Dispose();
        }
    }

    private SharedBackend _backend;
    private readonly byte[]? _hfJson;
    private readonly Dictionary<string, uint>? _tiktokenSpecials;
    private readonly Dictionary<string, string> _namedSpecials = new();
    private readonly List<string> _additionalSpecials = new();
    private int _disposed;

    private Tokenizer(SharedBackend backend, byte[]? hfJson, Dictionary<string, uint>? tiktokenSpecials)
    {
        _backend = backend;
        _hfJson = hfJson;
        _tiktokenSpecials = tiktokenSpecials;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    private TokenizerHandle Native
    {
        get
        {
            ThrowIfDisposed();
            return _backend.Handle;
        }
    }

    /// <summary>Shallow copy sharing native worker state (like the Python copy).</summary>
    /// <exception cref="ObjectDisposedException">The source was disposed.</exception>
    public Tokenizer(Tokenizer other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Volatile.Read(ref other._disposed) != 0)
            throw new ObjectDisposedException(nameof(Tokenizer));
        other._backend.AddRef();
        _backend = other._backend;
        _hfJson = other._hfJson;
        _tiktokenSpecials = other._tiktokenSpecials;
        foreach (var kv in other._namedSpecials) _namedSpecials[kv.Key] = kv.Value;
        _additionalSpecials.AddRange(other._additionalSpecials);
    }

    /// <summary>Wraps a BPE backend (no HF metadata retained; ownership moves to this facade).</summary>
    public Tokenizer(BpeTokenizer backend)
        : this(new SharedBackend((backend ?? throw new ArgumentNullException(nameof(backend))).TakeNative()), null, null)
    {
    }

    /// <summary>Wraps a SentencePiece backend (no HF metadata retained; ownership moves to this facade).</summary>
    public Tokenizer(SentencePieceTokenizer backend)
        : this(new SharedBackend((backend ?? throw new ArgumentNullException(nameof(backend))).TakeNative()), null, null)
    {
    }

    /// <summary>Backend kind.</summary>
    public TokenizerKind Kind =>
        NativeMethods.gt_tokenizer_kind(Native) == 1 ? TokenizerKind.SentencePiece : TokenizerKind.Bpe;

    internal IReadOnlyDictionary<string, string> NamedSpecials => _namedSpecials;
    internal IReadOnlyList<string> AdditionalSpecials => _additionalSpecials;

    // -- Construction ------------------------------------------------------

    private static readonly IReadOnlyDictionary<string, (string Pretokenizer, IReadOnlyDictionary<string, uint> Specials)> Encodings =
        new Dictionary<string, (string, IReadOnlyDictionary<string, uint>)>(StringComparer.Ordinal)
        {
            ["r50k_base"] = ("gpt2", new Dictionary<string, uint> { ["<|endoftext|>"] = 50256 }),
            ["cl100k_base"] = ("gpt4", new Dictionary<string, uint>
            {
                ["<|endoftext|>"] = 100257, ["<|fim_prefix|>"] = 100258,
                ["<|fim_middle|>"] = 100259, ["<|fim_suffix|>"] = 100260,
                ["<|endofprompt|>"] = 100276,
            }),
            ["o200k_base"] = ("o200k", new Dictionary<string, uint>
            {
                ["<|endoftext|>"] = 199999, ["<|endofprompt|>"] = 200018,
            }),
        };

    private static readonly string[] NamedAttrs =
        ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"];

    private static void CaptureNamedSpecials(JsonElement root, Dictionary<string, string> named, List<string> additional)
    {
        foreach (var attr in NamedAttrs)
        {
            if (!root.TryGetProperty(attr, out var v)) continue;
            if (v.ValueKind == JsonValueKind.String)
            {
                var s = v.GetString();
                if (s is not null) named[attr] = s;
            }
            else if (v.ValueKind == JsonValueKind.Object &&
                v.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
            {
                var s = c.GetString();
                if (s is not null) named[attr] = s;
            }
        }
        if (root.TryGetProperty("additional_special_tokens", out var add) &&
            add.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in add.EnumerateArray())
            {
                if (t.ValueKind == JsonValueKind.String)
                {
                    var s = t.GetString();
                    if (s is not null) additional.Add(s);
                }
                else if (t.ValueKind == JsonValueKind.Object &&
                    t.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                {
                    var s = c.GetString();
                    if (s is not null) additional.Add(s);
                }
            }
        }
    }

    private static void CaptureNamedSpecialsFromFile(string? configPath, Dictionary<string, string> named, List<string> additional)
    {
        if (configPath is null) return;
        try
        {
            if (!File.Exists(configPath)) return;
            using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
            CaptureNamedSpecials(doc.RootElement, named, additional);
        }
        catch
        {
            // Best-effort only: named specials never fail a load.
        }
    }

    /// <summary>
    /// Loads from a hub repo id, local directory, <c>tokenizer.json</c> file, or
    /// <c>.tiktoken</c>/<c>.model</c> file. Applies Kimi config dispatch first.
    /// </summary>
    public static Tokenizer FromPretrained(string repoOrPathOrDir)
    {
        ArgumentNullException.ThrowIfNull(repoOrPathOrDir);
        // Config dispatch first (Kimi/Moonlight TikToken lines).
        var dispatched = TryLoadFromConfig(repoOrPathOrDir);
        if (dispatched is not null) return dispatched;

        // Local paths first: a local directory/file whose name also looks like
        // a hub repo id must resolve locally, never via the network.
        if (Directory.Exists(repoOrPathOrDir))
        {
            var json = Path.Combine(repoOrPathOrDir, "tokenizer.json");
            if (!File.Exists(json))
                throw new FileNotFoundException($"No tokenizer.json in directory: {repoOrPathOrDir}");
            var data = File.ReadAllBytes(json);
            var tok = new Tokenizer(new SharedBackend(NativeBridge.LoadHfJson(data)), data, null);
            CaptureNamedSpecialsFromFile(Path.Combine(repoOrPathOrDir, "tokenizer_config.json"), tok._namedSpecials, tok._additionalSpecials);
            return tok;
        }
        if (File.Exists(repoOrPathOrDir))
            return LoadFileRoute(repoOrPathOrDir);
        if (Hub.LooksLikeRepoId(repoOrPathOrDir))
        {
            var file = Hub.GetFile(repoOrPathOrDir);
            return LoadFileRoute(file);
        }
        throw new FileNotFoundException($"No such file, directory, or repo id: {repoOrPathOrDir}");
    }

    private static Tokenizer LoadFileRoute(string file)
    {
        if (file.EndsWith(".tiktoken", StringComparison.OrdinalIgnoreCase))
            return FromTiktoken(file);
        if (file.EndsWith(".model", StringComparison.OrdinalIgnoreCase))
            return FromSentencePiece(file);
        var data = File.ReadAllBytes(file);
        var tok = new Tokenizer(new SharedBackend(NativeBridge.LoadHfJson(data)), data, null);
        var dir = Path.GetDirectoryName(Path.GetFullPath(file));
        if (dir is not null)
            CaptureNamedSpecialsFromFile(Path.Combine(dir, "tokenizer_config.json"), tok._namedSpecials, tok._additionalSpecials);
        return tok;
    }

    /// <summary>Loads from <c>tokenizer.json</c> text (auto BPE vs SentencePiece via byte_fallback).</summary>
    public static Tokenizer FromJson(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var data = Encoding.UTF8.GetBytes(json);
        return new Tokenizer(new SharedBackend(NativeBridge.LoadHfJson(data)), data, null);
    }

    /// <summary>Loads from <c>tokenizer.json</c> UTF-8 bytes.</summary>
    public static Tokenizer FromJson(ReadOnlySpan<byte> utf8Json)
    {
        var data = utf8Json.ToArray();
        return new Tokenizer(new SharedBackend(NativeBridge.LoadHfJson(data)), data, null);
    }

    /// <summary>
    /// Loads a <c>.tiktoken</c> file. When <paramref name="pretokenizer"/> is null,
    /// the scheme is inferred from the file name (<c>r50k_base</c>/<c>cl100k_base</c>/<c>o200k_base</c>);
    /// unknown names throw (never guessed). Explicit values override filename-derived schemes.
    /// </summary>
    public static Tokenizer FromTiktoken(
        string path, string? pretokenizer = null, IReadOnlyDictionary<string, uint>? specialTokens = null)
    {
        string scheme;
        Dictionary<string, uint> specials;
        if (pretokenizer is not null)
        {
            scheme = pretokenizer;
            specials = specialTokens is null ? [] : new Dictionary<string, uint>(specialTokens);
        }
        else
        {
            var stem = Path.GetFileNameWithoutExtension(path);
            if (!Encodings.TryGetValue(stem, out var enc))
                throw new ArgumentException(
                    $"Cannot infer pretokenizer from file name '{stem}'; pass pretokenizer explicitly.", nameof(path));
            scheme = enc.Pretokenizer;
            specials = specialTokens is null
                ? new Dictionary<string, uint>(enc.Specials)
                : new Dictionary<string, uint>(specialTokens);
        }
        var keys = specials.Select(kv => NativeBridge.Utf8Bytes(kv.Key)).ToArray();
        var vals = specials.Values.ToArray();
        var handle = NativeBridge.BpeFromTiktoken(
            NativeBridge.Utf8Bytes(Path.GetFullPath(path)), NativeBridge.Utf8Bytes(scheme), keys, vals);
        return new Tokenizer(new SharedBackend(handle), null, specials);
    }

    /// <summary>Loads a SentencePiece <c>.model</c> file (ported converter, no protobuf dependency).</summary>
    public static Tokenizer FromSentencePiece(string path)
    {
        var json = NativeBridge.SentencePieceModelToJson(File.ReadAllBytes(path));
        var tok = new Tokenizer(new SharedBackend(NativeBridge.SentencePieceFromBytes(json)), json, null);
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (dir is not null)
            CaptureNamedSpecialsFromFile(Path.Combine(dir, "tokenizer_config.json"), tok._namedSpecials, tok._additionalSpecials);
        return tok;
    }

    /// <summary>Loads SentencePiece from raw <c>.model</c> bytes.</summary>
    public static Tokenizer FromSentencePiece(ReadOnlySpan<byte> modelBytes)
    {
        var json = NativeBridge.SentencePieceModelToJson(modelBytes.ToArray());
        return new Tokenizer(new SharedBackend(NativeBridge.SentencePieceFromBytes(json)), json, null);
    }

    // -- Kimi / config dispatch ---------------------------------------------

    private static Tokenizer? TryLoadFromConfig(string source)
    {
        string? configPath = null;
        string? dir = null;
        if (Directory.Exists(source))
        {
            configPath = Path.Combine(source, "tokenizer_config.json");
            dir = source;
        }
        else if (File.Exists(source))
        {
            dir = Path.GetDirectoryName(Path.GetFullPath(source));
            if (dir is not null) configPath = Path.Combine(dir, "tokenizer_config.json");
        }
        else if (Hub.LooksLikeRepoId(source))
        {
            try
            {
                var cfg = Hub.GetFile(source, "tokenizer_config.json");
                configPath = cfg;
                dir = Path.GetDirectoryName(cfg);
            }
            catch { return null; }
        }
        if (configPath is null || !File.Exists(configPath)) return null;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
            var root = doc.RootElement;
            var cls = root.TryGetProperty("tokenizer_class", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString() ?? "" : "";
            string autoMap = "";
            if (root.TryGetProperty("auto_map", out var am) && am.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in am.EnumerateObject())
                {
                    if (prop.Name.Contains("Tokenizer", StringComparison.OrdinalIgnoreCase) &&
                        prop.Value.ValueKind == JsonValueKind.String)
                        autoMap = prop.Value.GetString() ?? "";
                }
            }
            var isTikTokenLine = cls.Contains("TikToken", StringComparison.OrdinalIgnoreCase) ||
                autoMap.Contains("kimi", StringComparison.OrdinalIgnoreCase) ||
                autoMap.Contains("moonshot", StringComparison.OrdinalIgnoreCase);
            if (!isTikTokenLine) return null;
            // Resolve the vocab file (tiktoken.model next to the config).
            var vocabFile = dir is not null ? Path.Combine(dir, "tiktoken.model") : "tiktoken.model";
            if (!File.Exists(vocabFile)) return null;
            var specials = new Dictionary<string, uint>();
            if (root.TryGetProperty("added_tokens_decoder", out var atd) && atd.ValueKind == JsonValueKind.Object)
            {
                foreach (var entry in atd.EnumerateObject())
                {
                    if (entry.Value.ValueKind == JsonValueKind.Object &&
                        entry.Value.TryGetProperty("content", out var content) &&
                        content.ValueKind == JsonValueKind.String)
                        specials[content.GetString() ?? ""] = uint.TryParse(entry.Name, out var sid) ? sid : 0;
                }
            }
            var keys = specials.Select(kv => NativeBridge.Utf8Bytes(kv.Key)).ToArray();
            var vals = specials.Values.ToArray();
            var handle = NativeBridge.BpeFromTiktoken(
                NativeBridge.Utf8Bytes(vocabFile), NativeBridge.Utf8Bytes("kimi"), keys, vals);
            var tok = new Tokenizer(new SharedBackend(handle), null, specials);
            CaptureNamedSpecials(root, tok._namedSpecials, tok._additionalSpecials);
            return tok;
        }
        catch { return null; }
    }

    // -- Metadata ------------------------------------------------------------

    internal byte[] HfJsonBytes
    {
        get
        {
            ThrowIfDisposed();
            return _hfJson ?? throw new InvalidOperationException(
                "This tokenizer has no tokenizer.json config (.tiktoken / config-line / raw-backend source).");
        }
    }

    internal Dictionary<string, uint> SpecialTokensMap()
    {
        ThrowIfDisposed();
        if (_tiktokenSpecials is not null) return new Dictionary<string, uint>(_tiktokenSpecials);
        if (_hfJson is null) return new Dictionary<string, uint>();
        // No try/catch: _hfJson is bytes that already passed native LoadHfJson
        // validation, so System.Text.Json parsing and u32 reads cannot fail
        // (unparsable added_tokens ids fail load; verified by probe).
        using var doc = JsonDocument.Parse(_hfJson);
        var map = new Dictionary<string, uint>();
        if (doc.RootElement.TryGetProperty("added_tokens", out var added) &&
            added.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in added.EnumerateArray())
            {
                if (t.ValueKind != JsonValueKind.Object) continue;
                var special = t.TryGetProperty("special", out var s) && s.ValueKind == JsonValueKind.True;
                if (!special) continue;
                if (t.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String &&
                    t.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.Number)
                    map[c.GetString() ?? ""] = id.GetUInt32();
            }
        }
        return map;
    }

    // -- Encode / decode ------------------------------------------------------

    /// <summary>Encodes text.</summary>
    public uint[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return BackendOps.Encode(Native, NativeBridge.Utf8Bytes(text));
    }

    /// <summary>Encodes raw UTF-8 bytes.</summary>
    public uint[] Encode(ReadOnlySpan<byte> utf8) => BackendOps.Encode(Native, utf8.ToArray());

    /// <summary>
    /// Encodes without normalization (SentencePiece backends only; mirrors the
    /// unified <c>Tokenizer.encode_no_normalize</c> passthrough, which upstream
    /// exposes via <c>__getattr__</c>).
    /// </summary>
    /// <exception cref="InvalidOperationException">The backend is not SentencePiece.</exception>
    public uint[] EncodeNoNormalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Kind == TokenizerKind.SentencePiece
            ? NativeBridge.SpEncodeNoNormalize(Native, NativeBridge.Utf8Bytes(text))
            : throw new InvalidOperationException("EncodeNoNormalize is only supported by SentencePiece backends.");
    }

    /// <summary>Encodes a batch of strings.</summary>
    public uint[][] EncodeBatch(IEnumerable<string> texts, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts as IList<string> ?? texts.ToList();
        var docs = new byte[list.Count][];
        for (var i = 0; i < list.Count; i++)
            docs[i] = NativeBridge.Utf8Bytes(list[i] ?? throw new ArgumentNullException($"{nameof(texts)}[{i}]"));
        return BackendOps.EncodeBatch(Native, docs, parallel);
    }

    /// <summary>Encodes a batch of raw UTF-8 documents.</summary>
    public uint[][] EncodeBatch(IEnumerable<byte[]> byteDocs, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatch(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel);
    }

    /// <summary>Encodes in-memory buffers (Rust splits on the separator).</summary>
    public uint[][] EncodeBatch(BytesSource source, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeRegions(Native, source.Buffers, source.Separator, parallel);
    }

    internal uint[][] EncodeBatchWrapped(IEnumerable<byte[]> byteDocs, WrapTruncateOptions options, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(byteDocs);
        ArgumentNullException.ThrowIfNull(options);
        using var wrap = options.BuildNative();
        return BackendOps.EncodeBatch(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel, wrap);
    }

    /// <summary>Encodes with padding/truncation into a dense matrix plus row lengths.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<string> texts, PadTruncateOptions options, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts as IList<string> ?? texts.ToList();
        var docs = new byte[list.Count][];
        for (var i = 0; i < list.Count; i++)
            docs[i] = NativeBridge.Utf8Bytes(list[i] ?? throw new ArgumentNullException($"{nameof(texts)}[{i}]"));
        return BackendOps.EncodeBatchPadded(Native, docs, parallel, options);
    }

    /// <summary>Encodes byte documents with padding/truncation.</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(IEnumerable<byte[]> byteDocs, PadTruncateOptions options, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(byteDocs);
        return BackendOps.EncodeBatchPadded(Native, byteDocs.Select(b => (b ?? throw new ArgumentNullException(nameof(byteDocs))).ToArray()).ToArray(), parallel, options);
    }

    /// <summary>Encodes in-memory buffers with padding/truncation (buffers are first split on the separator).</summary>
    public (uint[,] Matrix, long[] Lengths) EncodeBatchPadded(BytesSource source, PadTruncateOptions options, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        var rows = EncodeBatch(source, parallel);
        return PadRows(rows, options);
    }

    internal static (uint[,] Matrix, long[] Lengths) PadRows(uint[][] rows, PadTruncateOptions o)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (o.MaxLength.HasValue && o.MaxLength.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(o), "MaxLength must be non-negative.");
        if (o.PadToMaxLength && !o.MaxLength.HasValue)
            throw new ArgumentException("PadToMaxLength requires MaxLength.", nameof(o));
        if (o.Truncate && !o.MaxLength.HasValue)
            throw new ArgumentException("Truncate requires MaxLength.", nameof(o));
        // Managed fallback for the BytesSource path (native padded path takes flat inputs).
        var prefix = o.Prefix is null ? [] : o.Prefix.ToArray();
        var suffix = o.Suffix is null ? [] : o.Suffix.ToArray();
        var outs = new List<uint[]>(rows.Length);
        foreach (var row in rows)
        {
            ArgumentNullException.ThrowIfNull(row);
            var content = row;
            if (o.MaxLength.HasValue)
            {
                var extra = prefix.Length + suffix.Length;
                if (o.MaxLength.Value < extra)
                    throw new ArgumentException($"max_length={o.MaxLength.Value} leaves no room for the {extra} special tokens added per sequence");
                // Native `pad_truncate_matrix` only caps content when
                // truncate=true (cap=usize::MAX otherwise); MaxLength alone
                // is inert. When extra>max we throw above instead of clamping
                // to 0 and overflowing the width.
                if (o.Truncate)
                {
                    var cap = o.MaxLength.Value - extra;
                    if (content.Length > cap)
                        content = cap == 0 ? [] : (o.TruncateLeft ? content[^cap..] : content[..cap]);
                }
            }
            outs.Add([.. prefix, .. content, .. suffix]);
        }
        var width = outs.Count == 0 ? 0 : outs.Max(r => r.Length);
        if (o is { PadToMaxLength: true, MaxLength: { } ml }) width = ml;
        var matrix = new uint[outs.Count, width];
        var lengths = new long[outs.Count];
        for (var i = 0; i < outs.Count; i++)
        {
            var r = outs[i];
            lengths[i] = r.Length;
            if (o.PadLeft)
            {
                var off = width - r.Length;
                for (var j = 0; j < off; j++) matrix[i, j] = o.PadId;
                for (var j = 0; j < r.Length; j++) matrix[i, off + j] = r[j];
            }
            else
            {
                for (var j = 0; j < r.Length; j++) matrix[i, j] = r[j];
                for (var j = r.Length; j < width; j++) matrix[i, j] = o.PadId;
            }
        }
        return (matrix, lengths);
    }

    /// <summary>Encodes files.</summary>
    public uint[][] EncodeFiles(IFileSource source, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return BackendOps.EncodeFiles(Native, source.NativeHandle, parallel);
    }

    /// <summary>Encodes a single path (format defaults by extension).</summary>
    public uint[][] EncodeFiles(string path, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var bufs = new[] { NativeBridge.Utf8Bytes(Path.GetFullPath(path)) };
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Encodes several paths.</summary>
    public uint[][] EncodeFiles(IEnumerable<string> paths, bool? parallel = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var bufs = paths.Select(p => NativeBridge.Utf8Bytes(Path.GetFullPath(p))).ToArray();
        using var src = NativeBridge.PathSource(bufs);
        return BackendOps.EncodeFiles(Native, src, parallel);
    }

    /// <summary>Decodes ids to raw bytes.</summary>
    public byte[] Decode(ReadOnlySpan<uint> ids) => NativeBridge.Decode(Native, ids.ToArray());

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
        Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<int> ids) =>
        Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Decodes ids (range-validated) to a UTF-8 string.</summary>
    public string DecodeToString(ReadOnlySpan<long> ids) =>
        Encoding.UTF8.GetString(Decode(ids));

    /// <summary>Number of vocabulary entries (including added tokens).</summary>
    public int VocabSize { get { ThrowIfDisposed(); return BackendOps.VocabSize(_backend.Handle); } }
    /// <summary>Fresh id→bytes lookup.</summary>
    public FrozenDictionary<uint, byte[]> Vocab { get { ThrowIfDisposed(); return BackendOps.Vocab(_backend.Handle); } }
    /// <summary>Vocabulary entries in ID order.</summary>
    public IReadOnlyList<(uint Id, byte[] Bytes)> VocabEntries { get { ThrowIfDisposed(); return BackendOps.VocabEntries(_backend.Handle); } }
    /// <summary>Merges in merge-priority order.</summary>
    public IReadOnlyList<(byte[] Left, byte[] Right)> Merges { get { ThrowIfDisposed(); return BackendOps.Merges(_backend.Handle); } }
    /// <summary>Single-doc encode-path cache entry count.</summary>
    public int CacheEntries { get { ThrowIfDisposed(); return BackendOps.CacheEntries(_backend.Handle); } }

    /// <summary>Hugging Face compatibility wrapper.</summary>
    public HfCompat AsHf() { ThrowIfDisposed(); return new(this); }

    /// <summary>Tiktoken compatibility wrapper.</summary>
    public TiktokenCompat AsTiktoken() { ThrowIfDisposed(); return new(this); }

    /// <summary>Releases the shared native backend.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) { _backend.Release(); }
        GC.SuppressFinalize(this);
    }
}
