namespace Gigatoken.Tests;

/// <summary>BpeTokenizer backend parity with the Tokenizer facade (was untested).</summary>
public sealed class BackendParityTests
{
    private static void AssertPaddedEqual(
        (uint[,] Matrix, long[] Lengths) a, (uint[,] Matrix, long[] Lengths) b)
    {
        Assert.Equal(a.Lengths, b.Lengths);
        Assert.Equal(a.Matrix.GetLength(0), b.Matrix.GetLength(0));
        Assert.Equal(a.Matrix.GetLength(1), b.Matrix.GetLength(1));
        for (int i = 0; i < a.Matrix.GetLength(0); i++)
            for (int j = 0; j < a.Matrix.GetLength(1); j++)
                Assert.Equal(a.Matrix[i, j], b.Matrix[i, j]);
    }

    [Fact]
    public void PaddedMatchesFacade()
    {
        using var tok = Fixtures.LoadGpt2();
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var docs = new[] { "hello world", "hi" };
        var opts = new PadTruncateOptions(PadId: 0, MaxLength: 8, PadToMaxLength: true);
        AssertPaddedEqual(tok.EncodeBatchPadded(docs, opts), bpe.EncodeBatchPadded(docs, opts));
        var raw = docs.Select(System.Text.Encoding.UTF8.GetBytes).ToArray();
        AssertPaddedEqual(tok.EncodeBatchPadded(raw, opts), bpe.EncodeBatchPadded(raw, opts));
        using var src = new BytesSource(raw);
        AssertPaddedEqual(tok.EncodeBatchPadded(src, opts), bpe.EncodeBatchPadded(src, opts));
    }

    [Fact]
    public void EncodeFilesOverloads()
    {
        using var tok = Fixtures.LoadGpt2();
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var dir = Path.Combine(Path.GetTempPath(), $"bpe_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var a = Path.Combine(dir, "a.txt");
            var b = Path.Combine(dir, "b.txt");
            File.WriteAllText(a, "hello world");
            File.WriteAllText(b, "goodbye");
            Assert.Equal(tok.EncodeFiles(a), bpe.EncodeFiles(a));
            Assert.Equal(tok.EncodeFiles([a, b]), bpe.EncodeFiles([a, b]));
            using var src = new TextFileSource([a, b]);
            Assert.Equal(tok.EncodeFiles(src), bpe.EncodeFiles(src));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void DecodeOverloadsAndKind()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var ids = bpe.Encode("hello world");
        Assert.Equal(bpe.Decode([.. ids]), bpe.Decode(ids.Select(i => (int)i).ToArray()));
        Assert.Equal(bpe.Decode([.. ids]), bpe.Decode(ids.Select(i => (long)i).ToArray()));
        Assert.Equal(bpe.DecodeToString([.. ids]), bpe.DecodeToString(ids.Select(i => (int)i).ToArray()));
        Assert.Equal(bpe.DecodeToString([.. ids]), bpe.DecodeToString(ids.Select(i => (long)i).ToArray()));
        Assert.Throws<OverflowException>(() => bpe.Decode([-1]));
        Assert.Throws<OverflowException>(() => bpe.Decode([uint.MaxValue + 1L]));
        Assert.Equal(TokenizerKind.Bpe, bpe.Kind);
    }
}
