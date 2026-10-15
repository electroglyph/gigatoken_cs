namespace Gigatoken.Tests;

/// <summary>TiktokenCompat surface coverage (was only a smoke test in MiscTests).</summary>
public sealed class TiktokenCompatTests
{
    private static (string Path, Dictionary<string, uint> Specials) TinyWithSpecial() =>
        (Fixtures.WriteTinyTiktokenWithSpecial(out var s), s);

    private static TiktokenCompat Load(out string path)
    {
        var (p, specials) = TinyWithSpecial();
        path = p;
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        return tok.AsTiktoken();
    }

    [Fact]
    public void RoundtripAndOrdinary()
    {
        var compat = Load(out var path);
        try
        {
            Assert.Equal([256u], compat.Encode("AB", allowedSpecial: "all"));
            Assert.Equal([256u], compat.EncodeOrdinary("AB"));
            Assert.Equal("AB", compat.Decode([256u]));
            Assert.Equal("AB"u8.ToArray(), compat.DecodeBytes([256u]));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void DisallowedSpecialThrowsArgumentException()
    {
        var compat = Load(out var path);
        try
        {
            // Default is disallowed="all": the EOT string must be rejected.
            Assert.Throws<ArgumentException>(() => compat.Encode("A<|endoftext|>B"));
            // Explicitly allowed: passes through as the special id.
            Assert.Contains(258u, compat.Encode("A<|endoftext|>B", allowedSpecial: "all"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BatchMatchesSingleAndThreadsAgree()
    {
        var compat = Load(out var path);
        try
        {
            var docs = new[] { "AB", "ABC", "hello" };
            var mt = compat.EncodeBatch(docs, numThreads: 1);
            var st = compat.EncodeBatch(docs, numThreads: 8);
            Assert.Equal(mt, st);
            Assert.Equal([256u], mt[0]);
            Assert.Equal(compat.EncodeOrdinaryBatch(docs), mt);
            var back = compat.DecodeBatch(mt);
            Assert.Equal(docs, back);
            Assert.Equal(docs.Select(d => System.Text.Encoding.UTF8.GetBytes(d)).ToList(),
                compat.DecodeBytesBatch(mt));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SingleTokenHelpers()
    {
        var compat = Load(out var path);
        try
        {
            Assert.Equal(256u, compat.EncodeSingleToken("AB"));
            Assert.Equal(256u, compat.EncodeSingleToken("AB"u8.ToArray()));
            Assert.Equal("AB"u8.ToArray(), compat.DecodeSingleTokenBytes(256u));
            Assert.Equal(["AB"u8.ToArray()], compat.DecodeTokensBytes([256u]));
            Assert.Throws<KeyNotFoundException>(() => compat.EncodeSingleToken("ABCDEF"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void MetadataAndSpecials()
    {
        var compat = Load(out var path);
        try
        {
            Assert.Equal(258u, compat.EotToken);
            Assert.Equal(259, compat.NVocab);
            Assert.Equal(258, compat.MaxTokenValue);
            Assert.Contains("<|endoftext|>", compat.SpecialTokensSet);
            Assert.True(compat.IsSpecialToken(258u));
            Assert.True(compat.IsSpecial(258u));
            Assert.False(compat.IsSpecialToken(0u));
            Assert.False(compat.IsSpecial(0u));
            Assert.NotEmpty(compat.TokenByteValues());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void UnstableAndOffsetsThrow()
    {
        var compat = Load(out var path);
        try
        {
            Assert.Throws<NotSupportedException>(() => compat.EncodeWithUnstable("AB"));
            Assert.Throws<NotSupportedException>(() => compat.DecodeWithOffsets([256u]));
        }
        finally { File.Delete(path); }
    }
}
