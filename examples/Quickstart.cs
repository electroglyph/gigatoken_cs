// Quickstart: load a tokenizer and round-trip text.
// Runnable copy: executed by ExamplesTests.Quickstart.
namespace Gigatoken.Examples;

using Gigatoken;

/// <summary>Minimal load/encode/decode flow.</summary>
public static class Quickstart
{
    /// <summary>Encodes then decodes <c>"Hello, world!"</c>.</summary>
    /// <returns>The token ids and the decoded string.</returns>
    public static (uint[] Ids, string Text) Run(string tokenizerPath)
    {
        using var tok = Tokenizer.FromPretrained(tokenizerPath);
        var ids = tok.Encode("Hello, world!");
        return (ids, tok.DecodeToString(ids));
    }
}
