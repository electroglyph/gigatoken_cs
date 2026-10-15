namespace Gigatoken.Tests;

/// <summary>Shared fixture paths.</summary>
internal static class Fixtures
{
    internal static string Gpt2Json =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "gpt2_tokenizer.json");

    internal static Tokenizer LoadGpt2() =>
        Tokenizer.FromJson(File.ReadAllText(Gpt2Json));

    /// <summary>
    /// Writes a minimal valid .tiktoken file: 256 single-byte ranks, then
    /// exact-2-piece chain merges (AB=256, ABC=257), plus one special.
    /// </summary>
    internal static string WriteTinyTiktoken()
    {
        var lines = new List<string>(300);
        for (var b = 0; b < 256; b++)
            lines.Add($"{Convert.ToBase64String([(byte)b])} {b}");
        lines.Add($"{Convert.ToBase64String("AB"u8.ToArray())} 256");
        lines.Add($"{Convert.ToBase64String("ABC"u8.ToArray())} 257");
        var path = Path.Combine(Path.GetTempPath(), $"tiny_{Guid.NewGuid():N}.tiktoken");
        File.WriteAllLines(path, lines);
        return path;
    }

    internal static string WriteTinyTiktokenWithSpecial(out Dictionary<string, uint> specials)
    {
        var path = WriteTinyTiktoken();
        specials = new Dictionary<string, uint> { ["<|endoftext|>"] = 258 };
        return path;
    }
}
