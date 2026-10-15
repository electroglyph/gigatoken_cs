namespace Gigatoken.Tests;

/// <summary>
/// Phase 7 load/validation guards: unsupported model families are refused by
/// name (upstream `parse_tokenizer_json` probe), unknown SP normalizers pass
/// through as load errors, and non-UTF8 SP separators are rejected.
/// </summary>
public sealed class GuardTests
{
    [Fact]
    public void WordPieceTyped_RefusedByName()
    {
        var ex = Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromJson("""{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"hello":1}}}"""));
        Assert.Contains("Unsupported model type \"WordPiece\"", ex.Message);
    }

    [Fact]
    public void WordPieceUntypedLegacy_RefusedByName()
    {
        // Pre-0.9 files omit model.type; max_input_chars_per_word marks WordPiece.
        var ex = Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromJson("""{"model":{"vocab":{"[UNK]":0},"max_input_chars_per_word":100}}"""));
        Assert.Contains("WordPiece (untyped legacy file)", ex.Message);
    }

    [Fact]
    public void Unigram_RefusedByName()
    {
        var ex = Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromJson("""{"model":{"type":"Unigram","vocab":{}}}"""));
        Assert.Contains("Unsupported model type \"Unigram\"", ex.Message);

        var legacy = Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromJson("""{"model":{"vocab":{},"unk_id":0}}"""));
        Assert.Contains("Unigram (untyped legacy file)", legacy.Message);
    }

    private const string TinySpJson = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""";

    [Fact]
    public void SentencePieceMinimal_LoadsAsSentencePiece()
    {
        using var tok = Tokenizer.FromJson(TinySpJson);
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal([3u], tok.Encode("Hi"));
    }

    [Fact]
    public void SentencePieceUnknownNormalizer_Passthrough()
    {
        const string json = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]},"normalizer":{"type":"Nfkc"}}""";
        var ex = Assert.Throws<TokenizerLoadException>(() => Tokenizer.FromJson(json));
        Assert.Contains("Unsupported normalizer type for SentencePiece tokenizers: Nfkc", ex.Message);
    }

    [Fact]
    public void SentencePieceInvalidUtf8Separator_Guarded()
    {
        using var tok = Tokenizer.FromJson(TinySpJson);
        using var src = new BytesSource([new byte[] { (byte)'H', (byte)'i' }], new byte[] { 0xA9 });
        var ex = Assert.Throws<ArgumentException>(() => tok.EncodeBatch(src));
        Assert.NotEmpty(ex.Message);
    }

    [Fact]
    public void SentencePieceValidSeparator_Works()
    {
        using var tok = Tokenizer.FromJson(TinySpJson);
        using var src = new BytesSource([new byte[] { (byte)'H', (byte)'i' }], "|");
        var rows = tok.EncodeBatch(src);
        Assert.Single(rows);
    }
}
