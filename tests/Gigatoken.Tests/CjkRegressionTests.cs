namespace Gigatoken.Tests;

/// <summary>
/// Phase 7 CJK-note regression tracking: locks exact GPT-2 (r50k) vectors for
/// CJK/emoji/NFC-divergent/RTL/edge docs through the full C# stack. Oracle:
/// tiktoken `r50k_base` (`disallowed_special=()`, no specials in docs).
/// Upstream note (README): CJK-heavy data tokenizes slower (cache-hostile);
/// behavior is locked here, performance is tracked in the Phase 7 sweep.
/// </summary>
public sealed class CjkRegressionTests
{
    private static readonly (string Doc, uint[] Ids)[] Cases =
    [
        ("日本語のテキスト 中文文本 한국어", [33768, 98, 17312, 105, 45739, 252, 5641, 24336, 25084, 43302, 220, 40792, 23877, 229, 23877, 229, 17312, 105, 220, 47991, 250, 166, 113, 255, 168, 244, 112]),
        ("北京北京北京北京", [44293, 245, 12859, 105, 44293, 245, 12859, 105, 44293, 245, 12859, 105, 44293, 245, 12859, 105]),
        ("ひらがな カタカナ 漢字", [2515, 110, 36853, 35585, 26945, 17433, 104, 23376, 21763, 26229, 10545, 120, 95, 27764, 245]),
        ("🚀 emoji test 🎉👍", [8582, 248, 222, 44805, 1332, 12520, 236, 231, 41840, 235]),
        ("e\u0301clair vs \u00E9clair (combining vs precomposed)", [68, 136, 223, 27659, 3691, 38251, 27659, 357, 24011, 3191, 3691, 662, 5589, 1335, 8]),
        ("مرحبا بالعالم עברית", [25405, 26897, 148, 255, 39848, 12919, 17550, 101, 23525, 44690, 23525, 25405, 14360, 95, 49603, 37778, 33951, 103]),
        ("Hello, world!", [15496, 11, 995, 0]),
        ("  leading and trailing  ", [220, 3756, 290, 25462, 220, 220]),
        ("ﬁ ligature ½ fraction ｗｉｄｅ", [171, 105, 223, 26106, 1300, 25208, 13390, 27332, 121, 245, 171, 121, 231, 171, 121, 226, 171, 121, 227]),
        ("Japón Japón", [41, 499, 18840, 449, 499, 18840]),
        ("mixed 日本語 and English 123 numbers", [76, 2966, 10545, 245, 98, 17312, 105, 45739, 252, 290, 3594, 17031, 3146]),
        ("zero\u200bwidth\u200bjoiner", [22570, 9525, 10394, 9525, 7639, 7274]),
        ("ﬁle ﬁnder", [171, 105, 223, 293, 27332, 105, 223, 681]),
        ("«guillemets» “quotes” ‘single’", [24328, 5162, 8270, 76, 1039, 17730, 564, 250, 421, 6421, 447, 251, 564, 246, 29762, 447, 247]),
        ("ő ű Hungarian", [129, 239, 25370, 109, 27304]),
        ("ﬁ\u0301", [171, 105, 223, 136, 223]),
    ];

    [Fact]
    public void CjkVectors_MatchOracle()
    {
        using var tok = Fixtures.LoadGpt2();
        foreach (var (doc, ids) in Cases)
            Assert.Equal(ids, tok.Encode(doc));
    }

    [Fact]
    public void CjkVectors_BatchMatchesSingle()
    {
        using var tok = Fixtures.LoadGpt2();
        var rows = tok.EncodeBatch(Cases.Select(c => c.Doc));
        Assert.Equal(Cases.Length, rows.Length);
        for (var i = 0; i < Cases.Length; i++)
            Assert.Equal(Cases[i].Ids, rows[i]);
    }

    [Fact]
    public void CjkVectors_DecodeRoundtrip()
    {
        using var tok = Fixtures.LoadGpt2();
        foreach (var (doc, ids) in Cases)
            Assert.Equal(doc, tok.DecodeToString(ids));
    }
}
