using System.Text;

namespace Gigatoken.Tests;

public sealed class PretokenizerTests
{
    [Fact]
    public void IteratePieces()
    {
        using var it = Pretokenizer.Pretokenize("Hello, world!"u8.ToArray());
        var pieces = it.ToList().Select(m => Encoding.UTF8.GetString(m.Span)).ToArray();
        Assert.Equal(["Hello", ",", " world", "!"], pieces);
    }

    [Fact]
    public void EnumerableSinglePass()
    {
        using var it = Pretokenizer.Pretokenize("ab cd"u8.ToArray());
        var joined = string.Concat(it.Select(m => Encoding.UTF8.GetString(m.Span)));
        Assert.Equal("ab cd", joined);
    }

    [Fact]
    public void Counts()
    {
        // R50k keeps the leading space on non-initial words: "Hello Hello world"
        // splits as ["Hello", " Hello", " world"].
        var counts = Pretokenizer.CountPretokens("Hello Hello world"u8.ToArray());
        var dict = counts.ToDictionary(c => Encoding.UTF8.GetString(c.Pretoken), c => c.Count);
        Assert.Equal(1L, dict["Hello"]);
        Assert.Equal(1L, dict[" Hello"]);
        Assert.Equal(1L, dict[" world"]);
    }
}
