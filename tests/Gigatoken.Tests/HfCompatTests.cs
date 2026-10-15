namespace Gigatoken.Tests;

public sealed class HfCompatTests
{
    private static HfCompat Load() => Fixtures.LoadGpt2().AsHf();

    [Fact]
    public void EncodeMatchesBackend()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Equal(tok.Encode("Hello, world!"), hf.Encode("Hello, world!"));
    }

    [Fact]
    public void ByteLevelHasNoTemplateSpecials()
    {
        var hf = Load();
        Assert.Equal(0, hf.NumSpecialTokensToAdd());
    }

    [Fact]
    public void CallBatchMaskByDefault()
    {
        var hf = Load();
        var enc = hf.Call(["Hello", "world"]);
        Assert.Equal(2, enc.InputIds.Length);
        Assert.NotNull(enc.AttentionMask); // upstream returns attention_mask unless return_attention_mask=false
        Assert.Equal([1L], enc.AttentionMask[0]);
        var noMask = hf.Call(["Hello", "world"], returnAttentionMask: false);
        Assert.Null(noMask.AttentionMask);
        Assert.Throws<ArgumentException>(() => hf.Call(["Hi", "Hello, world!"], padding: "longest"));
    }

    [Fact]
    public void PaddingRequiresPadToken()
    {
        var hf = Load();
        Assert.Throws<ArgumentException>(() => hf.Call(["Hi", "Hello"], padding: "longest"));
        hf.PadToken = "<|endoftext|>";
        var enc = hf.Call(["Hi", "Hello, world!"], padding: "longest");
        Assert.NotNull(enc.AttentionMask);
        Assert.Equal(enc.InputIds[0].Length, enc.InputIds[1].Length);
    }

    [Fact]
    public void TruncationNeedsMaxLength()
    {
        var hf = Load();
        Assert.Throws<ArgumentException>(() => hf.Call("Hello", truncation: "longest_first"));
        var enc = hf.Call("Hello, world!", truncation: "longest_first", maxLength: 2);
        Assert.Single(enc.InputIds);
        Assert.Equal(2, enc.InputIds[0].Length);
    }

    [Fact]
    public void Tensors()
    {
        var hf = Load();
        hf.PadToken = "<|endoftext|>";
        var np = hf.Call(["Hi", "Hello, world!"], padding: "longest", returnTensors: "np");
        Assert.IsType<uint[,]>(np.TensorIds);
        var pt = hf.Call(["Hi", "Hello, world!"], padding: "longest", returnTensors: "pt");
        Assert.IsType<int[,]>(pt.TensorIds);
        Assert.Throws<NotSupportedException>(() =>
            hf.Call("Hi", returnTensors: "tf"));
    }

    [Fact]
    public void DecodeAndSkipSpecials()
    {
        var hf = Load();
        var ids = hf.Encode("Hello");
        Assert.Equal("Hello", hf.Decode(ids));
        var withSpecial = hf.Encode("Hello").Concat([50256u]).ToArray();
        Assert.Equal("Hello", hf.Decode(withSpecial, skipSpecialTokens: true));
        Assert.Equal(["Hello"], hf.BatchDecode([ids]));
    }

    [Fact]
    public void ConvertRoundtrip()
    {
        var hf = Load();
        var tokens = hf.ConvertIdsToTokens([15496u]);
        Assert.Equal(["Hello"], tokens);
        Assert.Equal(15496u, hf.ConvertTokensToIds("Hello"));
        Assert.Null(hf.ConvertTokensToIds("<not-a-token-xyz>"));
        Assert.Equal("Hello world", hf.ConvertTokensToString(["Hello", "Ġworld"]));
    }

    [Fact]
    public void PaddingLongestWithTruncation()
    {
        var hf = Load();
        hf.PadToken = "<|endoftext|>";
        // padding=longest + truncation must truncate to max_length, then pad to the longest row.
        var enc = hf.Call(["Hi there friend", "Hi"], padding: "longest",
            truncation: "longest_first", maxLength: 3);
        Assert.Equal(2, enc.InputIds.Length);
        Assert.Equal(3, enc.InputIds[0].Length);
        Assert.Equal(3, enc.InputIds[1].Length);
        Assert.NotNull(enc.AttentionMask);
        // padding=max_length pads out to max_length.
        var exact = hf.Call(["Hi"], padding: "max_length", maxLength: 5);
        Assert.Equal(5, exact.InputIds[0].Length);
    }

    [Fact]
    public void EncodeSupportsPaddingOptions()
    {
        var hf = Load();
        hf.PadToken = "<|endoftext|>";
        var ids = hf.Encode("Hi", null, true, "max_length", null, 4);
        Assert.Equal(4, ids.Length);
        var trunc = hf.Encode("Hello, world!", null, true, "do_not_pad", "longest_first", 2);
        Assert.Equal(2, trunc.Length);
        Assert.Throws<ArgumentException>(() => hf.Encode("Hi", "pair"));
    }

    [Fact]
    public void VocabSurface()
    {
        var hf = Load();
        Assert.Equal(50257, hf.VocabSize);
        Assert.Equal(50257, hf.Count);
        Assert.True(hf.GetVocab().ContainsKey("<|endoftext|>"));
        Assert.True(hf.GetAddedVocab().ContainsKey("<|endoftext|>"));
        // Upstream all_special_tokens = named + additional only (added excluded).
        Assert.Empty(hf.AllSpecialTokens);
        Assert.Empty(hf.AllSpecialIds);
        hf.EosToken = "<|endoftext|>";
        Assert.Contains("<|endoftext|>", hf.AllSpecialTokens);
        Assert.Contains((uint?)50256u, hf.AllSpecialIds);
        Assert.True(hf.IsFast);
    }
}
