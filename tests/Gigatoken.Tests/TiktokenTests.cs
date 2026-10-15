namespace Gigatoken.Tests;

public sealed class TiktokenTests
{
    [Fact]
    public void TinyFileRoundtrip()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            using var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2");
            Assert.Equal(TokenizerKind.Bpe, tok.Kind);
            // Merged pairs encode as single ids (256 = AB, 257 = ABC).
            Assert.Equal([256u], tok.Encode("AB"));
            Assert.Equal([257u], tok.Encode("ABC"));
            Assert.Equal("AB", tok.DecodeToString([256]));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SpecialTokenAtomic()
    {
        var path = Fixtures.WriteTinyTiktokenWithSpecial(out var specials);
        try
        {
            using var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
            var ids = tok.Encode("A<|endoftext|>B");
            Assert.Contains(258u, ids);
            Assert.Equal("A<|endoftext|>B", tok.DecodeToString(ids));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void UnknownNameRequiresPretokenizer()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            Assert.Throws<ArgumentException>(() => Tokenizer.FromTiktoken(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ExplicitSchemeOverridesFilename()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            using var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2");
            Assert.Equal([256u], tok.Encode("AB"));
        }
        finally { File.Delete(path); }
    }
}
