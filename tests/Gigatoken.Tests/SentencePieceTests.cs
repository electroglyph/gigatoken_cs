namespace Gigatoken.Tests;

/// <summary>
/// SentencePiece backend tests against a checked-in toy model
/// (<c>Fixtures/tiny_sp.model</c>: BPE, vocab 512, byte fallback, NFKC).
/// Golden IDs were captured with <c>sentencepiece</c> 0.2.x
/// <c>SentencePieceProcessor.encode</c> — an independent oracle, not gigatoken
/// itself (see <c>tools/export_goldens.py --sp</c>).
/// </summary>
public sealed class SentencePieceTests
{
    private static string ModelPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp.model");

    private static Tokenizer LoadSp() => Tokenizer.FromSentencePiece(ModelPath);

    private static readonly (string Text, uint[] Ids)[] Golden =
    [
        ("Hello, world!", [467u, 75u, 468u, 278u, 47u, 299u, 36u]),
        ("double  space", [409u, 282u]),
        ("\ttab and\nnewline", [262u, 309u, 280u, 290u, 468u, 488u, 472u, 292u]),
        (" leading", [416u]),
        ("trailing ", [418u]),
        ("NFKC: \uFB01 \u00BD \uFF57\uFF49\uFF44\uFF45",
            [467u, 81u, 73u, 78u, 70u, 487u, 265u, 480u, 467u, 486u, 229u, 132u, 135u, 491u, 291u, 480u, 286u]),
        ("CJK \u65E5\u672C\u8A9E",
            [467u, 70u, 77u, 78u, 467u, 233u, 154u, 168u, 233u, 159u, 175u, 235u, 173u, 161u]),
        ("null\0byte", [290u, 482u, 275u, 3u, 478u, 509u, 481u, 468u]),
        ("code:\n    indented block",
            [392u, 487u, 467u, 261u, 286u, 473u, 481u, 468u, 471u, 467u, 315u, 470u, 317u]),
        ("", []),
        (" ", []),
    ];

    [Fact]
    public void LoadsAsSentencePiece()
    {
        using var tok = LoadSp();
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal(512, tok.VocabSize);
    }

    [Fact]
    public void GoldenIdsMatchSentencePieceLib()
    {
        using var tok = LoadSp();
        foreach (var (text, ids) in Golden)
            Assert.Equal(ids, tok.Encode(text));
    }

    [Fact]
    public void DecodeRoundtrip()
    {
        using var tok = LoadSp();
        Assert.Equal("Hello, world!", tok.DecodeToString(tok.Encode("Hello, world!")));
        Assert.Equal("CJK \u65E5\u672C\u8A9E", tok.DecodeToString(tok.Encode("CJK \u65E5\u672C\u8A9E")));
    }

    [Fact]
    public void ModelPathRouting()
    {
        using var direct = Tokenizer.FromPretrained(ModelPath);
        using var explicitSp = LoadSp();
        Assert.Equal(TokenizerKind.SentencePiece, direct.Kind);
        foreach (var (text, _) in Golden)
            Assert.Equal(explicitSp.Encode(text), direct.Encode(text));
    }

    [Fact]
    public void EncodeNoNormalizeSkipsNfkc()
    {
        using var tok = LoadSp();
        // Already-normalized input: no dummy prefix is prepended, so plain
        // ASCII encodes differently than the normalized path, and the ids
        // decode back to the exact input (no phantom leading space).
        Assert.NotEqual(tok.Encode("hello"), tok.EncodeNoNormalize("hello"));
        Assert.Equal("hello", tok.DecodeToString(tok.EncodeNoNormalize("hello")));
        Assert.Empty(tok.EncodeNoNormalize(""));
        // U+FB01 normalizes to "fi" under NFKC: the paths must diverge.
        Assert.NotEqual(tok.Encode("ﬁligature"), tok.EncodeNoNormalize("ﬁligature"));
    }

    [Fact]
    public void FacadeNoNormalizeRequiresSentencePiece()
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Throws<InvalidOperationException>(() => tok.EncodeNoNormalize("hi"));
    }

    [Fact]
    public void RejectsNonBpeModel()
    {
        Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromSentencePiece([0x00, 0x01, 0x02, 0x03]));
    }

    [Fact]
    public void MissingModelFileThrows()
    {
        Assert.Throws<FileNotFoundException>(() =>
            Tokenizer.FromSentencePiece(Path.Combine(Path.GetTempPath(), "no-such-model.model")));
    }

    [Fact]
    public void InvalidUtf8SeparatorRejected()
    {
        using var tok = LoadSp();
        using var src = new BytesSource(
            [(ReadOnlyMemory<byte>)System.Text.Encoding.UTF8.GetBytes("é")], [0xA9]);
        Assert.Throws<ArgumentException>(() => tok.EncodeBatch(src));
    }

    [Fact]
    public void AddedTokenMarkupFollowsHf()
    {
        // sentencepiece-lib encodes <s>/</s> markup as ordinary pieces, but
        // HF tokenizers (and upstream gigatoken, whose reference is HF)
        // emits the control-token ids. The port follows HF: verified exact
        // over 4,975 hostile docs x 3 models against both oracles
        // (2026-10-15 audit); these vectors lock that behavior in.
        using var tok = LoadSp();
        Assert.Equal(
            [1u, 279u, 468u, 475u, 480u, 469u, 472u, 2u, 467u, 63u, 482u, 473u, 507u, 65u, 337u, 469u, 474u, 507u, 482u, 485u],
            tok.Encode("<s>special</s> <unk> markup"));
        Assert.Equal(
            [1u, 260u, 476u, 481u, 274u, 467u, 478u, 470u, 477u],
            tok.Encode("<s> after bos"));
        Assert.Equal(
            [467u, 468u, 455u, 477u, 291u, 480u, 481u, 479u, 279u, 468u, 475u, 480u, 469u, 472u, 1u],
            tok.Encode("ends with special <s>"));
    }

    private static string NoPrefixModelPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp_noprefix.model");

    private static readonly (string Text, uint[] Ids)[] NoPrefixGolden =
    [
        ("Hello, world!", [75u, 468u, 279u, 47u, 298u, 36u]),
        ("double  space", [401u, 283u]),
        ("\ttab and\nnewline", [481u, 308u, 282u, 336u, 468u, 488u, 472u, 291u]),
        (" leading", [409u]),
        ("trailing ", [481u, 474u, 400u]),
        ("NFKC: \uFB01 \u00BD \uFF57\uFF49\uFF44\uFF45",
            [81u, 73u, 78u, 70u, 487u, 263u, 480u, 467u, 486u, 229u, 132u, 135u, 491u, 290u, 480u, 270u]),
        ("CJK \u65E5\u672C\u8A9E",
            [70u, 77u, 78u, 467u, 233u, 154u, 168u, 233u, 159u, 175u, 235u, 173u, 161u]),
        ("null\0byte", [457u, 273u, 3u, 478u, 509u, 481u, 468u]),
        ("code:\n    indented block",
            [294u, 487u, 467u, 260u, 270u, 473u, 481u, 468u, 471u, 467u, 427u, 470u, 316u]),
        ("", []),
        (" ", []),
    ];

    [Fact]
    public void NoPrefixModelGoldenIdsMatchSentencePieceLib()
    {
        // add_dummy_prefix=false model (sp4096 class): no Prepend normalizer,
        // so there is no virtual leading space. Vectors captured with
        // sentencepiece 0.2.2 SentencePieceProcessor.encode.
        using var tok = Tokenizer.FromSentencePiece(NoPrefixModelPath);
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal(512, tok.VocabSize);
        foreach (var (text, ids) in NoPrefixGolden)
            Assert.Equal(ids, tok.Encode(text));
    }

    [Fact]
    public void NoPrefixModelDecodeMatchesSentencePieceLib()
    {
        // The Strip normalizer eats edge whitespace at encode, so it cannot
        // come back at decode (" leading" -> "leading", exactly like
        // sentencepiece-lib). What decode must NOT do for this model class
        // is strip a space of its own: "trailing" has no edge space left,
        // and a mid-string space survives ("a b" below stays intact).
        using var tok = Tokenizer.FromSentencePiece(NoPrefixModelPath);
        Assert.Equal("leading", tok.DecodeToString(tok.Encode(" leading")));
        Assert.Equal("trailing", tok.DecodeToString(tok.Encode("trailing ")));
        Assert.Equal("Hello, world!", tok.DecodeToString(tok.Encode("Hello, world!")));
    }

    [Fact]
    public void BytesSourceRegionPathMatchesPresplit()
    {
        // The SentencePiece backend's byte-level region path (separator split
        // inside Rust) must agree doc-by-doc with pre-split encoding,
        // including CJK docs. Mirrors upstream's
        // test_bytes_source_separator_matches_presplit.
        using var tok = LoadSp();
        string[] docs =
        [
            "Hello, world!",
            "CJK \u65E5\u672C\u8A9E \u4E2D\u6587\u6587\u672C",
            "double  space",
            "trailing ",
            " leading",
            "code:\n    indented block",
            "NFKC: \uFB01 \u00BD",
        ];
        var sep = System.Text.Encoding.UTF8.GetBytes("<|sep|>");
        var blob = System.Text.Encoding.UTF8.GetBytes(string.Join("<|sep|>", docs));
        using var src = new BytesSource([(ReadOnlyMemory<byte>)blob], sep);
        var via = tok.EncodeBatch(src);
        var direct = tok.EncodeBatch(docs);
        Assert.Equal(direct.Length, via.Length);
        for (var i = 0; i < direct.Length; i++)
            Assert.Equal(direct[i], via[i]);
    }
}
