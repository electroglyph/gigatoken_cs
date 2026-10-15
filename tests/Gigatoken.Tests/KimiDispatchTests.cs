namespace Gigatoken.Tests;

/// <summary>Moonshot/Kimi config dispatch (Tokenizer.TryLoadFromConfig).</summary>
public sealed class KimiDispatchTests
{
    private static string WriteModel(string dir)
    {
        // tiktoken.model uses the same line format as .tiktoken.
        var src = Fixtures.WriteTinyTiktoken();
        try
        {
            var dst = Path.Combine(dir, "tiktoken.model");
            File.Copy(src, dst);
            return dst;
        }
        finally { File.Delete(src); }
    }

    private static string MakeDir(string configJson)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"kimi_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        WriteModel(dir);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), configJson);
        return dir;
    }

    [Fact]
    public void TokenizerClassTikTokenDispatches()
    {
        var dir = MakeDir("""{"tokenizer_class": "TikTokenTokenizer"}""");
        try
        {
            using var tok = Tokenizer.FromPretrained(dir);
            Assert.Equal(TokenizerKind.Bpe, tok.Kind);
            Assert.Equal(258, tok.VocabSize);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void AutoMapMoonshotDispatches()
    {
        var dir = MakeDir("""{"auto_map": {"AutoTokenizer": "moonshot.kimi_tokenizer"}}""");
        try
        {
            using var tok = Tokenizer.FromPretrained(dir);
            Assert.Equal(TokenizerKind.Bpe, tok.Kind);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void NonTikTokenConfigFallsThrough()
    {
        var dir = MakeDir("""{"tokenizer_class": "BertTokenizer"}""");
        try
        {
            // No tokenizer.json present and no TikToken line: nothing to load.
            Assert.Throws<FileNotFoundException>(() => Tokenizer.FromPretrained(dir));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void MissingModelFileFallsThrough()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"kimi_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
            """{"tokenizer_class": "TikTokenTokenizer"}""");
        try
        {
            Assert.Throws<FileNotFoundException>(() => Tokenizer.FromPretrained(dir));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
