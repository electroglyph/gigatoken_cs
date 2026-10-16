namespace Gigatoken.Tests;

public sealed class VocabMergesTests
{
    [Fact]
    public void VocabIdOrderAndBytes()
    {
        using var tok = Fixtures.LoadGpt2();
        var entries = tok.VocabEntries;
        Assert.Equal(50257, entries.Count);
        for (var i = 0; i < entries.Count; i++)
            Assert.Equal((uint)i, entries[i].Id);
        // GPT-2 ids start at byte 33 ('!'): id 0 is '!', id 32 is 'A' (byte 65).
        Assert.Equal("A", System.Text.Encoding.UTF8.GetString(tok.Vocab[32]));
        Assert.Equal("!", System.Text.Encoding.UTF8.GetString(tok.Vocab[0]));
        Assert.True(tok.Vocab.TryGetValue(50256, out var eot));
        Assert.Equal("<|endoftext|>", System.Text.Encoding.UTF8.GetString(eot));
    }

    [Fact]
    public void MergesNonEmpty()
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Equal(50000, tok.Merges.Count);
        foreach (var (l, r) in tok.Merges.Take(100))
        {
            Assert.NotEmpty(l);
            Assert.NotEmpty(r);
        }
    }
}

public sealed class HubTests
{
    [Theory]
    [InlineData("gpt2", true)]
    [InlineData("org/name", true)]
    [InlineData("Qwen/Qwen3.5-9B", true)]
    [InlineData("./local/path", false)]
    [InlineData("tokenizer.json", false)]
    [InlineData("/abs/path/model", false)]
    public void LooksLikeRepoIdMatrix(string name, bool expected)
    {
        Assert.Equal(expected, Hub.LooksLikeRepoId(name));
    }

    [Fact]
    public void GetTokenDoesNotThrow()
    {
        // Returns the configured token or null; must never throw.
        _ = Hub.GetToken();
    }

    // huggingface.co answers 404 (→ HubFileNotFoundException) or 401
    // (→ UnauthorizedAccessException) for nonexistent repos depending on
    // edge/auth behavior; either is the clean typed failure.
    private static void AssertHubMissing(Action action)
    {
        var ex = Record.Exception(action);
        Assert.True(ex is HubFileNotFoundException || ex is UnauthorizedAccessException,
            $"Expected a typed hub-missing error, got {ex?.GetType().FullName}: {ex?.Message}");
    }

    [Fact]
    public void MissingHubFileThrows()
    {
        AssertHubMissing(() =>
            Hub.GetFile("definitely-not-a-real-repo-xyz-123", "tokenizer.json"));
    }
}

public sealed class MiscTests
{
    [Fact]
    public void AsHfWithoutConfigThrows()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            using var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2");
            Assert.Throws<InvalidOperationException>(() => tok.AsHf());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void TiktokenCompatSmoke()
    {
        using var tok = Fixtures.LoadGpt2();
        var tt = tok.AsTiktoken();
        var ids = tt.Encode("Hello, world!");
        Assert.Equal(tok.Encode("Hello, world!"), ids);
        Assert.Equal("Hello, world!", tt.Decode(ids));
        Assert.Equal(tok.DecodeToString(tok.Encode("Hello")), tt.Decode(tt.Encode("Hello")));
    }

    [Fact]
    public void TiktokenCompatSpecialsGated()
    {
        using var tok = Fixtures.LoadGpt2();
        var tt = tok.AsTiktoken();
        // Default: <|endoftext|> is disallowed -> ArgumentException.
        Assert.Throws<ArgumentException>(() => tt.Encode("a<|endoftext|>b"));
        // Allowed explicitly -> encodes fine and contains the EOT id.
        var ids = tt.Encode("a<|endoftext|>b", allowedSpecial: "all");
        Assert.Contains(50256u, ids);
    }

    [Fact]
    public void SubstringMatcherPresent()
    {
        using var m = new SubstringMatcher(["<|endoftext|>", "end", "foo"]);
        Assert.Equal([0, 1], m.Present("x <|endoftext|> y"));
        Assert.Empty(m.Present("nothing here"));
    }
}
