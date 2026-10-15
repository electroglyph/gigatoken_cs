namespace Gigatoken.Tests;

/// <summary>
/// Checked-in golden vectors with an independent oracle: expected GPT-2 IDs
/// were produced by <c>tiktoken.get_encoding("gpt2").encode</c> (OpenAI
/// tiktoken, r50k_base — same merges as <c>gpt2_tokenizer.json</c>), not by
/// gigatoken itself (see <c>tools/export_goldens.py --gpt2</c>).
/// </summary>
public sealed class GoldenVectorTests
{
    public static IEnumerable<object[]> Gpt2Goldens() => new[]
    {
        new object[] { "Hello, world!", new uint[] { 15496u, 11u, 995u, 0u } },
        new object[] { "", Array.Empty<uint>() },
        new object[] { " ", new uint[] { 220u } },
        new object[] { "Hello", new uint[] { 15496u } },
        new object[] { "The quick brown fox jumps over the lazy dog.",
            new uint[] { 464u, 2068u, 7586u, 21831u, 18045u, 625u, 262u, 16931u, 3290u, 13u } },
        new object[] { "Unicode: caf\u00E9 \u65E5\u672C\u8A9E \U0001F389",
            new uint[] { 3118u, 291u, 1098u, 25u, 40304u, 10545u, 245u, 98u, 17312u, 105u, 45739u, 252u, 12520u, 236u, 231u } },
        new object[] { "'quotes' \"double\" `backticks`",
            new uint[] { 6u, 421u, 6421u, 6u, 366u, 23352u, 1u, 4600u, 1891u, 83u, 3378u, 63u } },
        new object[] { " Say 'hi' & <bye>\n",
            new uint[] { 13816u, 705u, 5303u, 6u, 1222u, 1279u, 16390u, 29u, 198u } },
    };

    [Theory]
    [MemberData(nameof(Gpt2Goldens))]
    public void EncodeMatchesTiktokenOracle(string text, uint[] expected)
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Equal(expected, tok.Encode(text));
    }

    [Theory]
    [MemberData(nameof(Gpt2Goldens))]
    public void DecodeRoundtrip(string text, uint[] expected)
    {
        using var tok = Fixtures.LoadGpt2();
        Assert.Equal(text, tok.DecodeToString(expected));
        Assert.Equal(expected, tok.Encode(tok.DecodeToString(expected)));
    }

    [Fact]
    public void BatchMatchesGoldens()
    {
        using var tok = Fixtures.LoadGpt2();
        var inputs = Gpt2Goldens().Select(g => (string)g[0]).ToArray();
        var rows = tok.EncodeBatch(inputs, parallel: false);
        var i = 0;
        foreach (var g in Gpt2Goldens())
            Assert.Equal((uint[])g[1], rows[i++]);
    }
}
