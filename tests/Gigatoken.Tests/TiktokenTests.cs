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

    [Fact]
    public void NemotronRightAligned_GroupsDigitsFromRight()
    {
        // Upstream v0.11.0 (issue #60): "1234567" pretokens as 1|234|567.
        var path = WriteDigitTiktoken();
        try
        {
            using var aligned = Tokenizer.FromTiktoken(path, pretokenizer: "nemotron_right_aligned");
            Assert.Equal([49u, 257u, 259u], aligned.Encode("1234567"));
            using var plain = Tokenizer.FromTiktoken(path, pretokenizer: "nemotron");
            Assert.Equal([49u, 50u, 51u, 52u, 53u, 54u, 55u], plain.Encode("1234567"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void UnknownScheme_ErrorListsNemotronRightAligned()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                Tokenizer.FromTiktoken(path, pretokenizer: "nope"));
            Assert.Contains("nemotron_right_aligned", ex.Message);
        }
        finally { File.Delete(path); }
    }

    private static string WriteDigitTiktoken()
    {
        // .tiktoken ranks must form an exact-2-piece chain: every multi-byte
        // token splits into exactly two known pieces (else the loader panics).
        var lines = new List<string>(300);
        for (var b = 0; b < 256; b++)
            lines.Add($"{Convert.ToBase64String([(byte)b])} {b}");
        lines.Add($"{Convert.ToBase64String("23"u8.ToArray())} 256");
        lines.Add($"{Convert.ToBase64String("234"u8.ToArray())} 257");
        lines.Add($"{Convert.ToBase64String("56"u8.ToArray())} 258");
        lines.Add($"{Convert.ToBase64String("567"u8.ToArray())} 259");
        var path = Path.Combine(Path.GetTempPath(), $"digits_{Guid.NewGuid():N}.tiktoken");
        File.WriteAllLines(path, lines);
        return path;
    }
}
