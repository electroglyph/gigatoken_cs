namespace Gigatoken.Tests;

public sealed class TrainTests
{
    private static byte[] Corpus() =>
        System.Text.Encoding.UTF8.GetBytes(
            string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 20)));

    [Fact]
    public void TinyVocabCounts()
    {
        var (vocab, merges) = BpeTrainer.Train(Corpus(), 300, []);
        Assert.Equal(300, vocab.Count);
        Assert.Equal(300 - 256, merges.Count);
        // Byte alphabet seeded first.
        for (var b = 0; b < 256; b++)
            Assert.Equal([(byte)b], vocab[(uint)b]);
    }

    [Theory]
    [InlineData(TieBreaking.HuggingFace)]
    [InlineData(TieBreaking.RawTokenIds)]
    [InlineData(TieBreaking.AssembledBytes)]
    public void AllTieBreakings(TieBreaking tie)
    {
        var (vocab, merges) = BpeTrainer.Train(Corpus(), 280, [], tie);
        Assert.Equal(280, vocab.Count);
        Assert.Equal(280 - 256, merges.Count);
    }

    [Fact]
    public void SpecialTokensReserved()
    {
        var (vocab, merges) = BpeTrainer.Train(Corpus(), 260, ["<|special|>"]);
        Assert.Equal(260, vocab.Count);
        Assert.Contains(vocab.Values, b => System.Text.Encoding.UTF8.GetString(b) == "<|special|>");
    }

    [Fact]
    public void MergePairsConcatenate()
    {
        var (vocab, merges) = BpeTrainer.Train(Corpus(), 300, []);
        var byBytes = vocab.ToDictionary(kv => Convert.ToHexString(kv.Value), kv => kv.Key);
        foreach (var (l, r) in merges)
        {
            var joined = l.Concat(r).ToArray();
            Assert.True(byBytes.ContainsKey(Convert.ToHexString(joined)),
                "merged token missing from vocab");
        }
    }

    [Fact]
    public void GzipCorpusTrains()
    {
        var path = Path.Combine(Path.GetTempPath(), $"corpus_{Guid.NewGuid():N}.txt.gz");
        try
        {
            using (var fs = File.Create(path))
            using (var gz = new System.IO.Compression.GZipStream(fs, System.IO.Compression.CompressionLevel.Fastest))
                gz.Write(Corpus());
            var (vocab, merges) = BpeTrainer.Train(path, 280, []);
            Assert.Equal(280, vocab.Count);
            Assert.Equal(280 - 256, merges.Count);
        }
        finally { File.Delete(path); }
    }
}
