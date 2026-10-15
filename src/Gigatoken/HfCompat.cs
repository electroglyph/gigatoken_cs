using System.Text;
using System.Text.Json;
using Gigatoken.Interop;

namespace Gigatoken;

/// <summary>Tensor output selection for <c>HfCompat.Call</c>.</summary>
public enum HfTensorKind
{
    /// <summary>Nested lists (<c>uint[][]</c>).</summary>
    None,
    /// <summary>Dense <c>uint[,]</c> ids + <c>long[,]</c> mask (as-is).</summary>
    NumpyUInt32,
    /// <summary>Bit-cast <c>int[,]</c> ids + <c>long[,]</c> mask (mirrors <c>matrix.view(int32)</c>).</summary>
    TorchInt32,
}

/// <summary>Batch encoding result (mirrors <c>BatchEncoding</c>).</summary>
public sealed class BatchEncoding
{
    /// <summary>Per-row token ids.</summary>
    public uint[][] InputIds { get; }
    /// <summary>Per-row attention mask (1 = content, 0 = pad), when requested.</summary>
    public long[][]? AttentionMask { get; }
    /// <summary>Dense tensor ids when <c>returnTensors</c> was specified.</summary>
    public Array? TensorIds { get; }
    /// <summary>Dense tensor mask when <c>returnTensors</c> was specified.</summary>
    public long[,]? TensorMask { get; }

    internal BatchEncoding(uint[][] inputIds, long[][]? attentionMask, Array? tensorIds = null, long[,]? tensorMask = null)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        TensorIds = tensorIds;
        TensorMask = tensorMask;
    }
}

/// <summary>Hugging Face <c>transformers</c> compatibility wrapper (mirrors <c>HFCompat</c>).</summary>
public sealed class HfCompat
{
    private readonly Tokenizer _tokenizer;
    private readonly Dictionary<string, uint> _modelVocab;
    private readonly List<(string Content, uint Id, bool Special)> _addedTokens;
    private readonly HashSet<uint> _specialIds;
    private readonly Dictionary<uint, string> _idToToken;
    private readonly Dictionary<string, uint> _tokenToId;
    private readonly bool _byteFallback;
    private readonly uint[] _prefixIds;
    private readonly uint[] _suffixIds;
    private readonly string? _postProcessorError;

    /// <summary>Input names reported to pipelines.</summary>
    public IReadOnlyList<string> ModelInputNames { get; } = ["input_ids", "attention_mask"];
    /// <summary>Always true (fast backend).</summary>
    public bool IsFast => true;
    /// <summary>Padding side.</summary>
    public string PaddingSide { get; set; } = "right";
    /// <summary>Truncation side.</summary>
    public string TruncationSide { get; set; } = "right";

    // Named specials (null unless the tokenizer was built from an HF object carrying them;
    // path/dir loads carry none, mirroring Python _named_specials).
    // Settable so callers can assign e.g. PadToken = EosToken like upstream.
    /// <summary>BOS token, if known.</summary>
    public string? BosToken { get; set; }
    /// <summary>EOS token, if known.</summary>
    public string? EosToken { get; set; }
    /// <summary>UNK token, if known.</summary>
    public string? UnkToken { get; set; }
    /// <summary>SEP token, if known.</summary>
    public string? SepToken { get; set; }
    /// <summary>PAD token, if known.</summary>
    public string? PadToken { get; set; }
    /// <summary>CLS token, if known.</summary>
    public string? ClsToken { get; set; }
    /// <summary>MASK token, if known.</summary>
    public string? MaskToken { get; set; }
    /// <summary>Additional special tokens, if known.</summary>
    public IReadOnlyList<string> AdditionalSpecialTokens { get; set; } = [];

    /// <summary>Builds the wrapper (throws <see cref="InvalidOperationException"/> without HF config).</summary>
    public HfCompat(Tokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        var json = tokenizer.HfJsonBytes; // throws InvalidOperationException when absent
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        _modelVocab = [];
        if (root.TryGetProperty("model", out var model) && model.ValueKind == JsonValueKind.Object)
        {
            if (model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Object)
            {
                foreach (var kv in vocab.EnumerateObject())
                {
                    if (kv.Value.ValueKind == JsonValueKind.Number)
                        _modelVocab[kv.Name] = kv.Value.GetUInt32();
                }
            }
            _byteFallback = model.TryGetProperty("byte_fallback", out var bf) &&
                bf.ValueKind == JsonValueKind.True;
        }

        _addedTokens = [];
        _specialIds = [];
        _idToToken = new Dictionary<uint, string>();
        _tokenToId = new Dictionary<string, uint>(StringComparer.Ordinal);
        // Mirror upstream _id_to_token: model vocab first, added tokens overwrite (last-wins).
        foreach (var kv in _modelVocab)
        {
            _idToToken[kv.Value] = kv.Key;
            _tokenToId[kv.Key] = kv.Value;
        }
        if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in added.EnumerateArray())
            {
                if (t.ValueKind != JsonValueKind.Object) continue;
                // Defense-in-depth: native load already rejects missing content/id, so
                // skip rather than mapping a missing id to 0 (which could collide).
                if (!t.TryGetProperty("content", out var c) || c.ValueKind != JsonValueKind.String)
                    continue;
                if (!t.TryGetProperty("id", out var i) || i.ValueKind != JsonValueKind.Number)
                    continue;
                var content = c.GetString() ?? "";
                uint id;
                try { id = i.GetUInt32(); }
                catch { continue; }
                var special = t.TryGetProperty("special", out var s) && s.ValueKind == JsonValueKind.True;
                _addedTokens.Add((content, id, special));
                _idToToken[id] = content;
                _tokenToId[content] = id;
                if (special)
                {
                    _specialIds.Add(id);
                }
            }
        }

        // Template specials from post_processor. Missing/null post_processor yields
        // ([],[]) upstream (U-HF:43-44); only encoding with add_special_tokens=True
        // needs the post-processor, so defer failures until then.
        try
        {
            (_prefixIds, _suffixIds) = TemplateSpecialIds(root);
        }
        catch (ArgumentException ex)
        {
            _prefixIds = [];
            _suffixIds = [];
            _postProcessorError = ex.Message;
        }

        // H5: populate named specials from the source Tokenizer (captured from
        // tokenizer_config.json / config). Upstream objects built from an HF
        // tokenizer carry these; path/dir loads carry none.
        var named = tokenizer.NamedSpecials;
        if (named.TryGetValue("bos_token", out var bos)) BosToken = bos;
        if (named.TryGetValue("eos_token", out var eos)) EosToken = eos;
        if (named.TryGetValue("unk_token", out var unk)) UnkToken = unk;
        if (named.TryGetValue("sep_token", out var sep)) SepToken = sep;
        if (named.TryGetValue("pad_token", out var pad)) PadToken = pad;
        if (named.TryGetValue("cls_token", out var cls)) ClsToken = cls;
        if (named.TryGetValue("mask_token", out var mask)) MaskToken = mask;
        if (tokenizer.AdditionalSpecials.Count > 0)
            AdditionalSpecialTokens = [.. tokenizer.AdditionalSpecials];
    }

    private (uint[] Prefix, uint[] Suffix) TemplateSpecialIds(JsonElement root)
    {
        // Upstream returns ([],[]) when there is no post_processor (U-HF:43-44).
        if (!root.TryGetProperty("post_processor", out var pp) || pp.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            return ([], []);
        return WalkProcessor(pp);
    }

    private (uint[] Prefix, uint[] Suffix) WalkProcessor(JsonElement node)
    {
        if (node.ValueKind != JsonValueKind.Object || !node.TryGetProperty("type", out var t))
            throw new ArgumentException("unsupported post_processor structure.");
        switch (t.GetString())
        {
            case "ByteLevel":
                return ([], []);
            case "Sequence":
            {
                var pre = new List<uint>();
                var suf = new List<uint>();
                if (node.TryGetProperty("processors", out var procs))
                {
                    foreach (var p in procs.EnumerateArray())
                    {
                        var (a, b) = WalkProcessor(p);
                        // Upstream accumulates prefix = pre + prefix (later processors go first, U-HF:54);
                        // suffix order matches (suffix = suffix + suf).
                        pre.InsertRange(0, a);
                        suf.AddRange(b);
                    }
                }
                return ([.. pre], [.. suf]);
            }
            case "TemplateProcessing":
            {
                // Upstream reads pp["special_tokens"][id]["ids"] (a list) from the
                // post-processor block itself (U-HF:58,65), never the model vocab,
                // so multi-id specials are representable and template names need
                // not be vocab entries.
                JsonElement specials = default;
                var hasSpecials = node.TryGetProperty("special_tokens", out specials) &&
                    specials.ValueKind == JsonValueKind.Object;
                // Upstream defaults a missing "single" to [] (pp.get("single", [])).
                if (!node.TryGetProperty("single", out var single) || single.ValueKind != JsonValueKind.Array)
                    return ([], []);
                var pre = new List<uint>();
                var suf = new List<uint>();
                var seenSequence = false;
                foreach (var item in single.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.Object) continue;
                    var isSpecial = item.TryGetProperty("SpecialToken", out var st);
                    var isSequence = item.TryGetProperty("Sequence", out _);
                    if (isSpecial)
                    {
                        var id = st.TryGetProperty("id", out var sid) && sid.ValueKind == JsonValueKind.String
                            ? sid.GetString() ?? "" : "";
                        if (!hasSpecials || !specials.TryGetProperty(id, out var entry) ||
                            entry.ValueKind != JsonValueKind.Object ||
                            !entry.TryGetProperty("ids", out var idsEl) || idsEl.ValueKind != JsonValueKind.Array)
                            throw new ArgumentException($"Template special token '{id}' not found in special_tokens.");
                        var resolved = new List<uint>();
                        foreach (var e in idsEl.EnumerateArray())
                        {
                            if (e.ValueKind == JsonValueKind.Number)
                            {
                                try { resolved.Add(e.GetUInt32()); }
                                catch { throw new ArgumentException($"Template special token '{id}' has an out-of-range id."); }
                            }
                            else if (e.ValueKind == JsonValueKind.String)
                            {
                                // Tolerate string ids (token text) by resolving through the vocab.
                                var tok = e.GetString() ?? "";
                                var tid = TokenToIdNoUnk(tok);
                                if (tid is null)
                                    throw new ArgumentException($"Template special token '{id}' id '{tok}' not in vocab.");
                                resolved.Add(tid.Value);
                            }
                            else
                            {
                                throw new ArgumentException($"Template special token '{id}' has an unsupported id kind.");
                            }
                        }
                        (seenSequence ? suf : pre).AddRange(resolved);
                    }
                    else if (isSequence)
                    {
                        seenSequence = true;
                    }
                    else
                    {
                        throw new ArgumentException($"unsupported TemplateProcessing item: {item}");
                    }
                }
                return ([.. pre], [.. suf]);
            }
            default:
                throw new ArgumentException($"unsupported post_processor type: '{t.GetString()}'.");
        }
    }

    private void CheckPostProcessor()
    {
        if (_postProcessorError is not null)
            throw new ArgumentException($"cannot add special tokens: {_postProcessorError}; pass add_special_tokens=False and add them yourself if needed");
    }

    private (uint[] Prefix, uint[] Suffix) Specials(bool addSpecialTokens)
    {
        if (!addSpecialTokens) return ([], []);
        CheckPostProcessor();
        return (_prefixIds, _suffixIds);
    }

    // -- Properties ----------------------------------------------------------

    /// <summary>Base vocabulary size (without added tokens).</summary>
    public int VocabSize => _modelVocab.Count;
    /// <summary>Total length (model union added).</summary>
    public int Count => GetVocab().Count;
    /// <summary>Backend tokenizer.</summary>
    public Tokenizer Tokenizer => _tokenizer;
    /// <summary>Full vocab token→id (model plus added).</summary>
    public IReadOnlyDictionary<string, uint> Vocab => GetVocab();
    /// <summary>Model vocab plus added tokens.</summary>
    public IReadOnlyDictionary<string, uint> GetVocab()
    {
        var d = new Dictionary<string, uint>(_modelVocab);
        foreach (var (content, id, _) in _addedTokens) d[content] = id;
        return d;
    }
    /// <summary>Added tokens only.</summary>
    public IReadOnlyDictionary<string, uint> GetAddedVocab()
    {
        var d = new Dictionary<string, uint>();
        foreach (var (content, id, _) in _addedTokens) d[content] = id;
        return d;
    }
    /// <summary>Added token content→id.</summary>
    public IReadOnlyDictionary<string, uint> AddedTokensEncoder => GetAddedVocab();
    /// <summary>Added token id→content.</summary>
    public IReadOnlyDictionary<uint, string> AddedTokensDecoder
    {
        get
        {
            var d = new Dictionary<uint, string>();
            foreach (var (content, id, _) in _addedTokens) d[id] = content;
            return d;
        }
    }
    /// <summary>Special token name→token (unset names omitted).</summary>
    public IReadOnlyDictionary<string, string?> SpecialsMap
    {
        get
        {
            var d = new Dictionary<string, string?>();
            if (BosToken is not null) d["bos_token"] = BosToken;
            if (EosToken is not null) d["eos_token"] = EosToken;
            if (UnkToken is not null) d["unk_token"] = UnkToken;
            if (SepToken is not null) d["sep_token"] = SepToken;
            if (PadToken is not null) d["pad_token"] = PadToken;
            if (ClsToken is not null) d["cls_token"] = ClsToken;
            if (MaskToken is not null) d["mask_token"] = MaskToken;
            return d;
        }
    }
    /// <summary>All special token strings (named + additional only, mirroring upstream).</summary>
    public IReadOnlyList<string> AllSpecialTokens
    {
        get
        {
            var seen = new LinkedHashSet();
            foreach (var t in new[] { BosToken, EosToken, UnkToken, SepToken, PadToken, ClsToken, MaskToken })
                if (t is not null) seen.Add(t);
            foreach (var t in AdditionalSpecialTokens) seen.Add(t);
            return seen.ToList();
        }
    }
    /// <summary>All special token ids (unresolvable entries preserved as null, mirroring upstream).</summary>
    public IReadOnlyList<uint?> AllSpecialIds =>
        AllSpecialTokens.Select(TokenToIdNoUnk).ToArray();

    private sealed class LinkedHashSet
    {
        private readonly HashSet<string> _set = new();
        private readonly List<string> _order = new();
        public void Add(string s) { if (_set.Add(s)) _order.Add(s); }
        public List<string> ToList() => [.. _order];
    }

    private uint? NamedId(string? token) =>
        token is null ? null : TokenToIdNoUnk(token);

    private uint? TokenToIdNoUnk(string token) =>
        _tokenToId.TryGetValue(token, out var id) ? id : null;

    /// <summary>BOS id, if known.</summary>
    public uint? BosTokenId => NamedId(BosToken);
    /// <summary>EOS id, if known.</summary>
    public uint? EosTokenId => NamedId(EosToken);
    /// <summary>UNK id, if known.</summary>
    public uint? UnkTokenId => NamedId(UnkToken);
    /// <summary>SEP id, if known.</summary>
    public uint? SepTokenId => NamedId(SepToken);
    /// <summary>PAD id, if known.</summary>
    public uint? PadTokenId => NamedId(PadToken);
    /// <summary>CLS id, if known.</summary>
    public uint? ClsTokenId => NamedId(ClsToken);
    /// <summary>MASK id, if known.</summary>
    public uint? MaskTokenId => NamedId(MaskToken);

    /// <summary>Number of special tokens added for single/pair mode.</summary>
    public int NumSpecialTokensToAdd(bool pair = false)
    {
        if (pair) throw new ArgumentException("gigatoken.HFCompat does not support sequence pairs");
        CheckPostProcessor();
        return _prefixIds.Length + _suffixIds.Length;
    }

    // -- Encoding --------------------------------------------------------------

    private static readonly HashSet<string> ValidPaddings =
        new(StringComparer.OrdinalIgnoreCase) { "false", "do_not_pad", "true", "longest", "max_length" };

    /// <summary>Tokenizes and encodes one text (mirrors <c>__call__</c>).</summary>
    public BatchEncoding Call(
        string text, bool addSpecialTokens = true, string padding = "do_not_pad",
        string? truncation = null, int? maxLength = null, string? returnTensors = null,
        bool? returnAttentionMask = null, string? textPair = null)
    {
        if (text is null) throw new ArgumentNullException(nameof(text), "text must be provided.");
        return Call([text], addSpecialTokens, padding, truncation, maxLength, returnTensors, returnAttentionMask, textPair);
    }

    /// <summary>Tokenizes and encodes one text with bool-or-string padding/truncation (mirrors upstream bool|string spellings).</summary>
    public BatchEncoding Call(
        string text, object padding,
        object? truncation = null, int? maxLength = null, string? returnTensors = null,
        bool? returnAttentionMask = null, string? textPair = null, bool addSpecialTokens = true) =>
        Call(text, addSpecialTokens, NormalizePadding(padding), NormalizeTruncation(truncation),
            maxLength, returnTensors, returnAttentionMask, textPair);

    /// <summary>Tokenizes and encodes a batch with bool-or-string padding/truncation (mirrors upstream bool|string spellings).</summary>
    public BatchEncoding Call(
        IEnumerable<string> texts, object padding,
        object? truncation = null, int? maxLength = null, string? returnTensors = null,
        bool? returnAttentionMask = null, string? textPair = null, bool addSpecialTokens = true) =>
        Call(texts, addSpecialTokens, NormalizePadding(padding), NormalizeTruncation(truncation),
            maxLength, returnTensors, returnAttentionMask, textPair);

    private static string NormalizePadding(object padding) => padding switch
    {
        bool b => b ? "longest" : "do_not_pad",
        string s => s,
        _ => throw new NotSupportedException($"unsupported padding strategy '{padding}'."),
    };

    private static string? NormalizeTruncation(object? truncation) => truncation switch
    {
        null => null,
        bool b => b ? "longest_first" : "do_not_truncate",
        string s => s,
        // Tolerate int spellings by mapping non-zero to enabled (defensive; upstream has no int truncation).
        int i => i != 0 ? "longest_first" : "do_not_truncate",
        _ => throw new NotSupportedException($"unsupported truncation strategy '{truncation}'."),
    };

    /// <summary>Tokenizes and encodes a batch (mirrors <c>__call__</c>).</summary>
    public BatchEncoding Call(
        IEnumerable<string> texts, bool addSpecialTokens = true, string padding = "do_not_pad",
        string? truncation = null, int? maxLength = null, string? returnTensors = null,
        bool? returnAttentionMask = null, string? textPair = null)
    {
        if (texts is null) throw new ArgumentNullException(nameof(texts), "text must be provided.");
        if (textPair is not null)
            throw new ArgumentException("gigatoken.HFCompat does not support sequence pairs");
        if (PaddingSide is not ("right" or "left"))
            throw new ArgumentException($"padding_side must be \"right\" or \"left\", got '{PaddingSide}'.");
        if (TruncationSide is not ("right" or "left"))
            throw new ArgumentException($"truncation_side must be \"right\" or \"left\", got '{TruncationSide}'.");
        if (!ValidPaddings.Contains(padding))
            throw new NotSupportedException($"unsupported padding strategy '{padding}'.");
        // Upstream reserves dedicated ValueError for pair-only truncation values.
        if (truncation is "only_first" or "only_second")
            throw new ArgumentException("gigatoken.HFCompat does not support sequence pairs");
        var trunc = (truncation ?? "do_not_truncate").ToLowerInvariant();
        if (trunc is not ("false" or "do_not_truncate" or "true" or "longest_first"))
            throw new NotSupportedException($"unsupported truncation strategy '{truncation}'.");
        var doTruncate = trunc is "true" or "longest_first";
        if (doTruncate && maxLength is null)
            throw new ArgumentException("truncation requires an explicit max_length; gigatoken.HFCompat has no model_max_length to fall back to");
        var padMaxLength = padding.Equals("max_length", StringComparison.OrdinalIgnoreCase);
        if (padMaxLength && maxLength is null)
            throw new ArgumentException("padding=\"max_length\" requires max_length");
        if (maxLength is not null && !doTruncate && !padMaxLength)
            throw new ArgumentException("max_length has no effect without truncation or padding=\"max_length\"");
        if (maxLength is not null && maxLength < 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength), "max_length must be non-negative.");
        HfTensorKind kind = returnTensors switch
        {
            null => HfTensorKind.None,
            "np" => HfTensorKind.NumpyUInt32,
            "pt" => HfTensorKind.TorchInt32,
            _ => throw new NotSupportedException($"gigatoken.HFCompat supports return_tensors=\"np\" and \"pt\" only, got '{returnTensors}'."),
        };

        var inputs = texts.ToArray();
        var (prefix, suffix) = Specials(addSpecialTokens);

        uint[][] rows;
        long[][]? mask = null;
        var withMask = returnAttentionMask ?? true;
        if (padMaxLength || padding.Equals("true", StringComparison.OrdinalIgnoreCase) ||
            padding.Equals("longest", StringComparison.OrdinalIgnoreCase))
        {
            if (PadTokenId is not { } padId)
                throw new ArgumentException("asking to pad but the tokenizer has no pad token; set pad_token (e.g. tok.pad_token = tok.eos_token) first");
            var opts = new PadTruncateOptions(
                PadId: padId, MaxLength: maxLength, PadToMaxLength: padMaxLength,
                Truncate: doTruncate, PadLeft: PaddingSide == "left",
                TruncateLeft: TruncationSide == "left",
                Prefix: prefix, Suffix: suffix);
            var (matrix, lengths) = _tokenizer.EncodeBatchPadded(inputs, opts);
            rows = new uint[matrix.GetLength(0)][];
            for (var r = 0; r < rows.Length; r++)
            {
                rows[r] = new uint[matrix.GetLength(1)];
                for (var c = 0; c < rows[r].Length; c++) rows[r][c] = matrix[r, c];
            }
            if (withMask)
            {
                var left = PaddingSide == "left";
                mask = new long[rows.Length][];
                for (var r = 0; r < rows.Length; r++)
                {
                    var width = rows[r].Length;
                    mask[r] = new long[width];
                    for (var c = 0; c < width; c++)
                        mask[r][c] = left ? (c >= width - lengths[r] ? 1 : 0) : (c < lengths[r] ? 1 : 0);
                }
            }
        }
        else
        {
            int? cap = null;
            if (doTruncate)
            {
                cap = maxLength!.Value - prefix.Length - suffix.Length;
                if (cap < 0)
                    throw new ArgumentException($"max_length={maxLength} leaves no room for the {prefix.Length + suffix.Length} special tokens added per sequence.");
            }
            var wrap = new WrapTruncateOptions
            {
                Prefix = prefix, Suffix = suffix,
                MaxTokens = cap, TruncateLeft = TruncationSide == "left",
            };
            rows = _tokenizer.EncodeBatchWrapped(
                inputs.Select(NativeBridge.Utf8Bytes), wrap);
            if (withMask)
                mask = rows.Select(r => Enumerable.Repeat(1L, r.Length).ToArray()).ToArray();
        }

        Array? tensorIds = null;
        long[,]? tensorMask = null;
        if (kind != HfTensorKind.None)
        {
            var width = rows.Length == 0 ? 0 : rows.Max(r => r.Length);
            if (rows.Any(r => r.Length != width))
                throw new ArgumentException("Unable to create tensor: the encoded sequences have different lengths; pass padding=True (or encode without return_tensors) instead");
            var uintMat = new uint[rows.Length, width];
            for (var r = 0; r < rows.Length; r++)
                for (var c = 0; c < width; c++) uintMat[r, c] = rows[r][c];
            tensorIds = kind == HfTensorKind.NumpyUInt32 ? uintMat : BitCastToInt(uintMat);
            // Mirror upstream with_mask gate (U-HF:359,374-378): no mask when
            // return_attention_mask=false. Otherwise reuse the attention mask so
            // padded left-pad placement matches.
            if (withMask)
            {
                tensorMask = new long[rows.Length, width];
                for (var r = 0; r < rows.Length; r++)
                    for (var c = 0; c < width; c++) tensorMask[r, c] = mask![r][c];
            }
        }
        return new BatchEncoding(rows, mask, tensorIds, tensorMask);
    }

    private static int[,] BitCastToInt(uint[,] m)
    {
        var (r, c) = (m.GetLength(0), m.GetLength(1));
        var o = new int[r, c];
        for (var i = 0; i < r; i++)
            for (var j = 0; j < c; j++) o[i, j] = unchecked((int)m[i, j]);
        return o;
    }

    /// <summary>Encodes one text to ids (mirrors upstream <c>encode</c>, which delegates to the batch call).</summary>
    public uint[] Encode(string text, bool addSpecialTokens = true) =>
        Call(text, addSpecialTokens, "do_not_pad", null, null, null, false).InputIds[0];

    /// <summary>Encodes one text to ids with padding/truncation options.</summary>
    public uint[] Encode(string text, string? textPair, bool addSpecialTokens = true,
        string padding = "do_not_pad", string? truncation = null, int? maxLength = null) =>
        Call(text, addSpecialTokens, padding, truncation, maxLength, null, false, textPair).InputIds[0];

    /// <summary>Tokenizes one text to tokens.</summary>
    public List<string?> Tokenize(string text, string? pair = null, bool addSpecialTokens = false)
    {
        if (pair is not null)
            throw new ArgumentException("gigatoken.HFCompat does not support sequence pairs");
        return ConvertIdsToTokens(Encode(text, addSpecialTokens));
    }

    /// <summary>Decodes a single id to text.</summary>
    public string Decode(uint tokenId, bool skipSpecialTokens = false) =>
        Decode([tokenId], skipSpecialTokens);

    /// <summary>Decodes a single id to text (range-validated).</summary>
    public string Decode(int tokenId, bool skipSpecialTokens = false)
    {
        if (tokenId < 0) throw new OverflowException($"Negative token id: {tokenId}.");
        return Decode([(uint)tokenId], skipSpecialTokens);
    }

    /// <summary>Decodes a single id to text (range-validated).</summary>
    public string Decode(long tokenId, bool skipSpecialTokens = false)
    {
        if (tokenId < 0 || tokenId > uint.MaxValue)
            throw new OverflowException($"Token id out of range: {tokenId}.");
        return Decode([(uint)tokenId], skipSpecialTokens);
    }

    /// <summary>Decodes ids to text.</summary>
    public string Decode(uint[] tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var ids = skipSpecialTokens ? tokenIds.Where(id => !_specialIds.Contains(id)).ToArray() : tokenIds;
        return Encoding.UTF8.GetString(_tokenizer.Decode(ids));
    }

    /// <summary>Decodes ids to text (range-validated).</summary>
    public string Decode(int[] tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var conv = new uint[tokenIds.Length];
        for (var i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] < 0) throw new OverflowException($"Negative token id at index {i}: {tokenIds[i]}.");
            conv[i] = (uint)tokenIds[i];
        }
        return Decode(conv, skipSpecialTokens);
    }

    /// <summary>Decodes ids to text (range-validated).</summary>
    public string Decode(long[] tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var conv = new uint[tokenIds.Length];
        for (var i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] < 0 || tokenIds[i] > uint.MaxValue)
                throw new OverflowException($"Token id out of range at index {i}: {tokenIds[i]}.");
            conv[i] = (uint)tokenIds[i];
        }
        return Decode(conv, skipSpecialTokens);
    }

    /// <summary>Decodes ids to text.</summary>
    public string Decode(IEnumerable<uint> tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        return Decode(tokenIds.ToArray(), skipSpecialTokens);
    }

    /// <summary>Decodes a nested batch to texts (mirrors upstream nested <c>decode</c> dispatch to batch).</summary>
    public List<string> Decode(IEnumerable<uint[]> batch, bool skipSpecialTokens = false) =>
        BatchDecode(batch, skipSpecialTokens);

    /// <summary>Decodes a batch to texts.</summary>
    public List<string> BatchDecode(IEnumerable<uint[]> batch, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(batch);
        return batch.Select(ids => Decode(ids.ToArray(), skipSpecialTokens)).ToList();
    }

    /// <summary>Decodes a batch to texts (range-validated).</summary>
    public List<string> BatchDecode(IEnumerable<int[]> batch, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(batch);
        return batch.Select(ids => Decode(ids.ToArray(), skipSpecialTokens)).ToList();
    }

    /// <summary>Decodes a batch to texts (range-validated).</summary>
    public List<string> BatchDecode(IEnumerable<long[]> batch, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(batch);
        return batch.Select(ids => Decode(ids.ToArray(), skipSpecialTokens)).ToList();
    }

    /// <summary>Converts a single id to its token (unknown → null).</summary>
    public string? ConvertIdsToTokens(uint id) =>
        _idToToken.TryGetValue(id, out var tok) ? tok : null;

    /// <summary>Converts a single id to its token (unknown → null; negative → null like upstream dict lookup).</summary>
    public string? ConvertIdsToTokens(int id, bool skipSpecialTokens = false)
    {
        if (id < 0) return null;
        var u = (uint)id;
        if (skipSpecialTokens && _specialIds.Contains(u)) return null;
        return ConvertIdsToTokens(u);
    }

    /// <summary>Converts a single id to its token (unknown → null; out-of-range → null like upstream dict lookup).</summary>
    public string? ConvertIdsToTokens(long id, bool skipSpecialTokens = false)
    {
        if (id < 0 || id > uint.MaxValue) return null;
        var u = (uint)id;
        if (skipSpecialTokens && _specialIds.Contains(u)) return null;
        return ConvertIdsToTokens(u);
    }

    /// <summary>Converts ids to tokens (added tokens first; unknown → null).</summary>
    public List<string?> ConvertIdsToTokens(IEnumerable<uint> ids, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(ids);
        if (skipSpecialTokens)
            ids = ids.Where(id => !_specialIds.Contains(id));
        return ids.Select(id => _idToToken.TryGetValue(id, out var tok) ? tok : null).ToList();
    }

    /// <summary>Converts ids to tokens (unknown → null).</summary>
    public List<string?> ConvertIdsToTokens(IEnumerable<int> ids, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return ids.Select(id => ConvertIdsToTokens(id, skipSpecialTokens)).ToList();
    }

    /// <summary>Converts ids to tokens (unknown → null).</summary>
    public List<string?> ConvertIdsToTokens(IEnumerable<long> ids, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return ids.Select(id => ConvertIdsToTokens(id, skipSpecialTokens)).ToList();
    }

    /// <summary>Converts a token to its id (added first, else unk fallback, else null).</summary>
    public uint? ConvertTokensToIds(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        var id = TokenToIdNoUnk(token);
        if (id is null && UnkToken is not null)
            id = TokenToIdNoUnk(UnkToken);
        return id;
    }

    /// <summary>Converts tokens to ids.</summary>
    public List<uint?> ConvertTokensToIds(IEnumerable<string> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        return tokens.Select(ConvertTokensToIds).ToList();
    }

    /// <summary>Detokenizes tokens to a string.</summary>
    public string ConvertTokensToString(IEnumerable<string> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        var list = tokens.ToList();
        if (_byteFallback)
        {
            var raw = new List<byte>();
            foreach (var token in list)
            {
                if (token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token.EndsWith(">", StringComparison.Ordinal))
                {
                    try
                    {
                        raw.Add(Convert.ToByte(token.Substring(3, 2), 16));
                    }
                    catch (Exception ex) when (ex is FormatException or OverflowException)
                    {
                        throw new ArgumentException($"Invalid byte-fallback token '{token}'.", ex);
                    }
                }
                else
                {
                    raw.AddRange(Encoding.UTF8.GetBytes(token.Replace("▁", " ")));
                }
            }
            var text = Encoding.UTF8.GetString(raw.ToArray());
            return text.StartsWith(' ') ? text[1..] : text;
        }
        var u2b = Gpt2UnicodeToByte();
        var bytes = new List<byte>(list.Count * 4);
        foreach (var tok in list)
            foreach (var ch in tok)
            {
                if (u2b.TryGetValue(ch, out var b)) bytes.Add(b);
                else bytes.AddRange(Encoding.UTF8.GetBytes([ch]));
            }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private static IReadOnlyDictionary<char, byte> Gpt2UnicodeToByte()
    {
        var bs = new List<int>();
        bs.AddRange(Enumerable.Range(33, 127 - 33));
        bs.AddRange(Enumerable.Range(161, 173 - 161));
        bs.AddRange(Enumerable.Range(174, 256 - 174));
        var cs = new List<int>(bs);
        var n = 0;
        for (var b = 0; b < 256; b++)
        {
            if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
        }
        var d = new Dictionary<char, byte>();
        for (var i = 0; i < bs.Count; i++) d[(char)cs[i]] = (byte)bs[i];
        return d;
    }
}
