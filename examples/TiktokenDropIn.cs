// Tiktoken drop-in: load a .tiktoken file and use the tiktoken-compatible API.
// Runnable copy: executed by ExamplesTests.TiktokenDropIn.
namespace Gigatoken.Examples;

using Gigatoken;

/// <summary>GPT-2-style encode/decode through <see cref="TiktokenCompat"/>.</summary>
public static class TiktokenDropIn
{
    /// <summary>Encodes <paramref name="text"/> and decodes it back.</summary>
    /// <returns>The token ids and the decoded string.</returns>
    public static (uint[] Ids, string Text) Run(string tiktokenPath, string pretokenizer, string text)
    {
        using var tok = Tokenizer.FromTiktoken(tiktokenPath, pretokenizer);
        var compat = tok.AsTiktoken();
        var ids = compat.EncodeOrdinary(text);
        return (ids, compat.Decode(ids));
    }
}
