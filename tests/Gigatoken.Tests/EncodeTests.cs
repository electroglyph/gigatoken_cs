namespace Gigatoken.Tests;

public sealed class EncodeTests
{
    [Fact]
    public void GoldenVector()
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Equal([15496u, 11u, 995u, 0u], tok.Encode("Hello, world!"));
    }

    [Fact]
    public void Roundtrip()
    {
        using var tok = Fixtures.LoadGpt2();
        const string text = "Tokenize your text data at GB/s! 日本語 🎉";
        Assert.Equal(text, tok.DecodeToString(tok.Encode(text)));
    }

    [Fact]
    public void BatchMatchesSingle()
    {
        using var tok = Fixtures.LoadGpt2();
        string[] docs = ["Hello, world!", "", "aaaa", "The first document."];
        var batch = tok.EncodeBatch(docs);
        Assert.Equal(docs.Length, batch.Length);
        for (var i = 0; i < docs.Length; i++)
            Assert.Equal(tok.Encode(docs[i]), batch[i]);
        Assert.Empty(batch[1]);
    }

    [Fact]
    public void SerialMatchesParallel()
    {
        using var tok = Fixtures.LoadGpt2();
        string[] docs = ["Hello, world!", "aaaa bbbb cccc", "x"];
        var serial = tok.EncodeBatch(docs, parallel: false);
        var parallel = tok.EncodeBatch(docs, parallel: true);
        Assert.Equal(serial.Length, parallel.Length);
        for (var i = 0; i < serial.Length; i++)
            Assert.Equal(serial[i], parallel[i]);
    }

    [Fact]
    public void ByteDocsMatchStrings()
    {
        using var tok = Fixtures.LoadGpt2();
        var strings = tok.EncodeBatch(["Hello, world!"]);
        var bytes = tok.EncodeBatch([System.Text.Encoding.UTF8.GetBytes("Hello, world!")]);
        Assert.Equal(strings[0], bytes[0]);
    }

    [Fact]
    public void WrapPrefixSuffix()
    {
        using var tok = Fixtures.LoadGpt2();
        var opts = new WrapTruncateOptions { Prefix = [1u], Suffix = [2u] };
        var rows = tok.EncodeBatchWrapped([System.Text.Encoding.UTF8.GetBytes("Hi")], opts);
        Assert.Single(rows);
        Assert.Equal(1u, rows[0][0]);
        Assert.Equal(2u, rows[0][^1]);
        // max_tokens=0 keeps only prefix+suffix
        var opts0 = new WrapTruncateOptions { Prefix = [1u], Suffix = [2u], MaxTokens = 0 };
        var rows0 = tok.EncodeBatchWrapped([System.Text.Encoding.UTF8.GetBytes("Hi there")], opts0);
        Assert.Equal([1u, 2u], rows0[0]);
    }

    [Fact]
    public void PaddedMatrix()
    {
        using var tok = Fixtures.LoadGpt2();
        var opts = new PadTruncateOptions(PadId: 0, PadToMaxLength: false);
        var (matrix, lengths) = tok.EncodeBatchPadded(["Hi", "Hello, world!"], opts);
        Assert.Equal(2, matrix.GetLength(0));
        Assert.Equal(lengths.Max(), matrix.GetLength(1));
        Assert.Equal(tok.Encode("Hi").Length, lengths[0]);
    }

    [Fact]
    public void DecodeOverloadsAndRangeValidation()
    {
        using var tok = Fixtures.LoadGpt2();
        var ids = tok.Encode("Hello");
        Assert.Equal(tok.Decode(ids), tok.Decode(ids.Select(i => (int)i).ToArray()));
        Assert.Equal(tok.Decode(ids), tok.Decode(ids.Select(i => (long)i).ToArray()));
        Assert.Throws<OverflowException>(() => tok.Decode([-1]));
        Assert.Throws<OverflowException>(() => tok.Decode([-5L]));
    }

    [Fact]
    public void KindAndVocabSize()
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Equal(TokenizerKind.Bpe, tok.Kind);
        Assert.Equal(50257, tok.VocabSize);
        Assert.Equal(50000, tok.Merges.Count);
    }

    [Fact]
    public void CopySharesBackend()
    {
        using var tok = Fixtures.LoadGpt2();
        using var copy = new Tokenizer(tok);
        Assert.Equal(tok.Encode("Hello, world!"), copy.Encode("Hello, world!"));
    }

    [Fact]
    public void CacheEntriesNonNegative()
    {
        using var tok = Fixtures.LoadGpt2();
        _ = tok.Encode("Hello");
        Assert.True(tok.CacheEntries >= 0);
    }
}
