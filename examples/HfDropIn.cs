// HuggingFace drop-in: tokenizer.json through the HF-compatible wrapper.
// Runnable copy: executed by ExamplesTests.HfDropIn.
namespace Gigatoken.Examples;

using Gigatoken;

/// <summary>Batch encode plus id/token conversion through <see cref="HfCompat"/>.</summary>
public static class HfDropIn
{
    /// <summary>Encodes one text and converts the ids back to tokens.</summary>
    /// <returns>The encoded ids and their token strings.</returns>
    public static (uint[] Ids, List<string?> Tokens) Run(string tokenizerPath, string text)
    {
        using var tok = Tokenizer.FromPretrained(tokenizerPath);
        var hf = tok.AsHf();
        var batch = hf.Call([text]);
        var tokens = hf.ConvertIdsToTokens(batch.InputIds[0]);
        return (batch.InputIds[0], tokens);
    }
}
