namespace Gigatoken.Tests;

public sealed class CoverageHfCompatTests
{
    private const string ByteLevelPp = "\"post_processor\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false}";

    private static string Gpt2Text() => File.ReadAllText(Fixtures.Gpt2Json);

    private static Tokenizer FromPp(string ppJson)
    {
        var raw = Gpt2Text().Replace(ByteLevelPp, "\"post_processor\":" + ppJson, StringComparison.Ordinal);
        return Tokenizer.FromJson(raw);
    }

    private static Tokenizer FromNoPp()
    {
        var raw = Gpt2Text().Replace("," + ByteLevelPp, string.Empty, StringComparison.Ordinal);
        return Tokenizer.FromJson(raw);
    }

    // -- BatchEncoding surface (line 28) + Vocab/Tokenizer (292,294) + Added vocab (310,316-318) --

    [Fact]
    public void TensorMaskSurface()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        hf.PadToken = "<|endoftext|>";
        var np = hf.Call(["Hi", "Hello, world!"], padding: "longest", returnTensors: "np");
        Assert.IsType<uint[,]>(np.TensorIds);
        Assert.NotNull(np.TensorMask);
        Assert.Equal(np.TensorIds.GetLength(0), np.TensorMask.GetLength(0));
        var pt = hf.Call(["Hi", "Hello, world!"], padding: "longest", returnTensors: "pt");
        Assert.IsType<int[,]>(pt.TensorIds);
        Assert.NotNull(pt.TensorMask);
        var noTensor = hf.Call(["Hi"], returnTensors: null);
        Assert.Null(noTensor.TensorIds);
        Assert.Null(noTensor.TensorMask);
        var noMask = hf.Call(["Hi", "Hello"], padding: "longest", returnAttentionMask: false, returnTensors: "np");
        Assert.Null(noMask.AttentionMask);
        Assert.Null(noMask.TensorMask);
        Assert.NotNull(noMask.TensorIds);
    }

    [Fact]
    public void VocabTokenizerAddedSurface()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.NotNull(hf.Tokenizer);
        Assert.Equal(50257, hf.VocabSize);
        Assert.Equal(hf.GetVocab().Count, hf.Count);
        Assert.Equal(hf.GetVocab().Count, hf.Vocab.Count);
        Assert.True(hf.Vocab.ContainsKey("Hello"));
        Assert.True(hf.AddedTokensEncoder.ContainsKey("<|endoftext|>"));
        Assert.Equal(50256u, hf.AddedTokensEncoder["<|endoftext|>"]);
        var dec = hf.AddedTokensDecoder;
        Assert.Equal("<|endoftext|>", dec[50256u]);
        Assert.Single(hf.GetAddedVocab());
    }

    // -- SpecialsMap + NamedIds + AllSpecials --

    [Fact]
    public void SpecialsMapEmptyWhenUnset()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Empty(hf.SpecialsMap);
        Assert.Null(hf.BosTokenId);
        Assert.Null(hf.EosTokenId);
        Assert.Null(hf.UnkTokenId);
        Assert.Null(hf.SepTokenId);
        Assert.Null(hf.PadTokenId);
        Assert.Null(hf.ClsTokenId);
        Assert.Null(hf.MaskTokenId);
    }

    [Fact]
    public void SpecialsMapPopulatedViaSetters()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        hf.BosToken = "<|endoftext|>";
        hf.EosToken = "<|endoftext|>";
        hf.UnkToken = "<|endoftext|>";
        hf.SepToken = "[SEP]";
        hf.PadToken = "<|endoftext|>";
        hf.ClsToken = "[CLS]";
        hf.MaskToken = "[MASK]";
        var map = hf.SpecialsMap;
        Assert.Equal(7, map.Count);
        Assert.Equal("<|endoftext|>", map["bos_token"]);
        Assert.Equal("<|endoftext|>", map["eos_token"]);
        Assert.Equal("<|endoftext|>", map["unk_token"]);
        Assert.Equal("[SEP]", map["sep_token"]);
        Assert.Equal("<|endoftext|>", map["pad_token"]);
        Assert.Equal("[CLS]", map["cls_token"]);
        Assert.Equal("[MASK]", map["mask_token"]);
        Assert.Equal(50256u, hf.BosTokenId);
        Assert.Equal(50256u, hf.EosTokenId);
        Assert.Equal(50256u, hf.UnkTokenId);
        Assert.Null(hf.SepTokenId);
        Assert.Equal(50256u, hf.PadTokenId);
        Assert.Null(hf.ClsTokenId);
        Assert.Null(hf.MaskTokenId);
        hf.AdditionalSpecialTokens = ["<extra1>", "<extra1>", "<extra2>"];
        Assert.Equal(["<|endoftext|>", "[SEP]", "[CLS]", "[MASK]", "<extra1>", "<extra2>"], hf.AllSpecialTokens);
        var ids = hf.AllSpecialIds;
        Assert.Equal(6, ids.Count);
        Assert.Equal(50256u, ids[0]);
        Assert.Null(ids[1]);
    }

    [Fact]
    public void AdditionalSpecialsViaPretrainedDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "hfcov_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"), Gpt2Text());
            File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
                """{"bos_token":"<|endoftext|>","eos_token":{"content":"<|endoftext|>"},"unk_token":"<unk-xyz>","sep_token":"[SEP]","pad_token":"<|endoftext|>","cls_token":"[CLS]","mask_token":"[MASK]","additional_special_tokens":["<extra1>","extra2",{"content":"<extra3>"}]}""");
            using var tok = Tokenizer.FromPretrained(dir);
            var hf = tok.AsHf();
            Assert.Equal("<|endoftext|>", hf.BosToken);
            Assert.Equal("<|endoftext|>", hf.EosToken);
            Assert.Equal("<unk-xyz>", hf.UnkToken);
            Assert.Equal("[SEP]", hf.SepToken);
            Assert.Equal("<|endoftext|>", hf.PadToken);
            Assert.Equal("[CLS]", hf.ClsToken);
            Assert.Equal("[MASK]", hf.MaskToken);
            Assert.Equal(["<extra1>", "extra2", "<extra3>"], hf.AdditionalSpecialTokens);
            Assert.Equal(7, hf.SpecialsMap.Count);
            Assert.Contains("<extra1>", hf.AllSpecialTokens);
            Assert.Equal(50256u, hf.BosTokenId);
            Assert.Null(hf.UnkTokenId);
            Assert.Null(hf.SepTokenId);
            Assert.Equal(50256u, hf.PadTokenId);
            Assert.Null(hf.ClsTokenId);
            Assert.Null(hf.MaskTokenId);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    // -- post_processor shapes --

    [Fact]
    public void NullAndMissingPostProcessorYieldNoSpecials()
    {
        using var t1 = FromPp("null");
        var h1 = t1.AsHf();
        Assert.Equal(0, h1.NumSpecialTokensToAdd());
        Assert.Single(h1.Call("hi", addSpecialTokens: true).InputIds);

        using var t2 = FromNoPp();
        var h2 = t2.AsHf();
        Assert.Equal(0, h2.NumSpecialTokensToAdd());
        Assert.Single(h2.Call("hi", addSpecialTokens: true).InputIds);
    }

    [Fact]
    public void SequenceWithProcessorsAccumulatesInUpstreamOrder()
    {
        using var tok = FromPp("""{"type":"Sequence","processors":[{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"a","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"b","type_id":0}}],"pair":[],"special_tokens":{"a":{"id":"a","ids":[10],"tokens":["x"]},"b":{"id":"b","ids":[11],"tokens":["y"]}}},{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"c","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"d","type_id":0}}],"pair":[],"special_tokens":{"c":{"id":"c","ids":[20],"tokens":["x"]},"d":{"id":"d","ids":[21],"tokens":["y"]}}},{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false}]}""");
        var hf = tok.AsHf();
        Assert.Equal(4, hf.NumSpecialTokensToAdd());
        var ids = hf.Call("hi", addSpecialTokens: true).InputIds[0];
        // Later processors prepend (U-HF:54): prefix [20,10]; suffix appends: [11,21].
        Assert.Equal([20u, 10u, 5303u, 11u, 21u], ids);
        var plain = hf.Call("hi", addSpecialTokens: false).InputIds[0];
        Assert.Equal([5303u], plain);
    }

    [Fact]
    public void SequenceWithoutProcessorsYieldsEmpty()
    {
        using var tok = FromPp("""{"type":"Sequence"}""");
        var hf = tok.AsHf();
        Assert.Equal(0, hf.NumSpecialTokensToAdd());
        Assert.Equal([5303u], hf.Call("hi", addSpecialTokens: true).InputIds[0]);
    }

    [Fact]
    public void TemplateMissingSingleYieldsEmpty()
    {
        using var t1 = FromPp("""{"type":"TemplateProcessing","pair":[],"special_tokens":{}}""");
        Assert.Equal(0, t1.AsHf().NumSpecialTokensToAdd());
        using var t2 = FromPp("""{"type":"TemplateProcessing","single":"oops","pair":[],"special_tokens":{}}""");
        Assert.Equal(0, t2.AsHf().NumSpecialTokensToAdd());
    }

    [Fact]
    public void TemplateNonObjectItemsSkipped()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[42,"s",{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}""");
        var hf = tok.AsHf();
        Assert.Equal(0, hf.NumSpecialTokensToAdd());
        Assert.Equal([5303u], hf.Call("hi", addSpecialTokens: true).InputIds[0]);
    }

    [Fact]
    public void TemplateValidPrefixSuffix()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"sep","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[101],"tokens":["[CLS]"]},"sep":{"id":"sep","ids":[102],"tokens":["[SEP]"]}}}""");
        var hf = tok.AsHf();
        Assert.Equal(2, hf.NumSpecialTokensToAdd());
        Assert.Equal([101u, 5303u, 102u], hf.Call("hi", addSpecialTokens: true).InputIds[0]);
    }

    [Fact]
    public void TemplateStringIdsResolveThroughVocab()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":["Hello"],"tokens":["Hello"]}}}""");
        var hf = tok.AsHf();
        Assert.Equal(1, hf.NumSpecialTokensToAdd());
        Assert.Equal([15496u, 5303u], hf.Call("hi", addSpecialTokens: true).InputIds[0]);
    }

    [Fact]
    public void TemplateMultiIdSuffix()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"a","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"b","type_id":0}},{"SpecialToken":{"id":"c","type_id":0}}],"pair":[],"special_tokens":{"a":{"id":"a","ids":[10],"tokens":["x"]},"b":{"id":"b","ids":[11,12],"tokens":["x","y"]},"c":{"id":"c","ids":[13],"tokens":["z"]}}}""");
        var hf = tok.AsHf();
        Assert.Equal(4, hf.NumSpecialTokensToAdd());
        Assert.Equal([10u, 5303u, 11u, 12u, 13u], hf.Call("hi", addSpecialTokens: true).InputIds[0]);
    }

    // -- deferred-error paths (CheckPostProcessor) --

    private static void AssertDeferred(string ppJson, string fragment)
    {
        using var tok = FromPp(ppJson);
        var hf = tok.AsHf();
        var ex1 = Assert.Throws<ArgumentException>(() => hf.NumSpecialTokensToAdd());
        Assert.Contains(fragment, ex1.Message);
        var ex2 = Assert.Throws<ArgumentException>(() => hf.Call("hi", addSpecialTokens: true));
        Assert.StartsWith("cannot add special tokens: ", ex2.Message);
        Assert.Contains(fragment, ex2.Message);
        // add_special_tokens=False bypasses the broken processor.
        Assert.Equal([5303u], hf.Call("hi", addSpecialTokens: false).InputIds[0]);
        Assert.Throws<ArgumentException>(() => hf.Encode("hi", addSpecialTokens: true));
    }

    [Fact]
    public void DeferredUnknownType()
    {
        AssertDeferred("""{"type":"Foo"}""", "unsupported post_processor type: 'Foo'.");
    }

    [Fact]
    public void DeferredBertProcessing()
    {
        AssertDeferred("""{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}""", "unsupported post_processor type: 'BertProcessing'.");
    }

    [Fact]
    public void DeferredMissingType()
    {
        AssertDeferred("""{"foo":1}""", "unsupported post_processor structure.");
    }

    [Fact]
    public void DeferredNestedSequenceError()
    {
        AssertDeferred("""{"type":"Sequence","processors":[{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}]}""", "Template special token 'cls' not found in special_tokens.");
    }

    [Fact]
    public void DeferredSpecialNotFound()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}""", "Template special token 'cls' not found in special_tokens.");
    }

    [Fact]
    public void DeferredSpecialTokensNotObject()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":[]}""", "Template special token 'cls' not found in special_tokens.");
    }

    [Fact]
    public void DeferredSpecialEntryMissingIds()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","tokens":["[CLS]"]}}}""", "Template special token 'cls' not found in special_tokens.");
    }

    [Fact]
    public void DeferredSpecialEmptyId()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}""", "Template special token '' not found in special_tokens.");
    }

    [Fact]
    public void DeferredSpecialNumericIdWithoutString()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":123,"type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}""", "Template special token '' not found in special_tokens.");
    }

    [Fact]
    public void DeferredOutOfRangeIds()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[9999999999],"tokens":["x"]}}}""", "has an out-of-range id.");
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[-1],"tokens":["x"]}}}""", "has an out-of-range id.");
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[1.5],"tokens":["x"]}}}""", "has an out-of-range id.");
    }

    [Fact]
    public void DeferredUnsupportedIdKind()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[true],"tokens":["x"]}}}""", "has an unsupported id kind.");
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[null],"tokens":["x"]}}}""", "has an unsupported id kind.");
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":[{}],"tokens":["x"]}}}""", "has an unsupported id kind.");
    }

    [Fact]
    public void DeferredStringIdNotInVocab()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"cls":{"id":"cls","ids":["<not-in-vocab-xyz>"],"tokens":["x"]}}}""", "not in vocab.");
    }

    [Fact]
    public void DeferredUnsupportedItem()
    {
        AssertDeferred("""{"type":"TemplateProcessing","single":[{"Foo":{}}],"pair":[],"special_tokens":{}}""", "unsupported TemplateProcessing item:");
    }

    // -- Call validation --

    [Fact]
    public void CallNullGuards()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<ArgumentNullException>(() => hf.Call((string)null!));
        Assert.Throws<ArgumentNullException>(() => hf.Call((IEnumerable<string>)null!));
    }

    [Fact]
    public void CallPairAndSideValidation()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<ArgumentException>(() => hf.Call("hi", textPair: "there"));
        Assert.Throws<ArgumentException>(() => hf.Call(["hi"], textPair: "there"));
        Assert.Throws<ArgumentException>(() => hf.Tokenize("hi", "there"));
        Assert.Equal(["Hello"], hf.Tokenize("Hello").Where(t => t is not null).Select(t => t!));
        hf.PaddingSide = "middle";
        Assert.Throws<ArgumentException>(() => hf.Call("hi"));
        hf.PaddingSide = "right";
        hf.TruncationSide = "middle";
        Assert.Throws<ArgumentException>(() => hf.Call("hi"));
        hf.TruncationSide = "right";
    }

    [Fact]
    public void CallPaddingTruncationStrategies()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", padding: "sometimes"));
        Assert.Throws<ArgumentException>(() => hf.Call("hi", truncation: "only_first"));
        Assert.Throws<ArgumentException>(() => hf.Call("hi", truncation: "only_second"));
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", truncation: "only_third"));
        Assert.Throws<ArgumentException>(() => hf.Call("hi", padding: "max_length"));
        Assert.Throws<ArgumentException>(() => hf.Call("hi", maxLength: 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => hf.Call("hi", truncation: "longest_first", maxLength: -1));
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", returnTensors: "tf"));
        // Valid spellings do not throw.
        hf.PadToken = "<|endoftext|>";
        var a = hf.Call("hi", padding: "max_length", maxLength: 4);
        Assert.Equal(4, a.InputIds[0].Length);
        var b = hf.Call("Hello, world!", truncation: "longest_first", maxLength: 2);
        Assert.Equal(2, b.InputIds[0].Length);
    }

    [Fact]
    public void CallObjectOverloadsNormalize()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        hf.PadToken = "<|endoftext|>";
        // bool padding spells longest / do_not_pad.
        var padded = hf.Call("hi", true, maxLength: null, returnTensors: null, returnAttentionMask: null, textPair: null, addSpecialTokens: false);
        Assert.Single(padded.InputIds);
        var unpadded = hf.Call("hi", false);
        Assert.Single(unpadded.InputIds);
        var batchPadded = hf.Call(["hi", "Hello, world!"], true);
        Assert.Equal(2, batchPadded.InputIds.Length);
        var batchUnpadded = hf.Call(["hi"], false);
        Assert.Single(batchUnpadded.InputIds);
        // string passthrough.
        Assert.Single(hf.Call("hi", "do_not_pad").InputIds);
        Assert.Single(hf.Call(["hi"], "do_not_pad").InputIds);
        // int truncation tolerance.
        Assert.Single(hf.Call("Hello, world!", "do_not_pad", 1, maxLength: 2).InputIds);
        Assert.Single(hf.Call("Hello, world!", "do_not_pad", 0).InputIds);
        Assert.Single(hf.Call(["Hello, world!"], "do_not_pad", 1, maxLength: 2).InputIds);
        // bool truncation.
        Assert.Single(hf.Call("Hello, world!", "do_not_pad", true, maxLength: 2).InputIds);
        Assert.Single(hf.Call("Hello, world!", "do_not_pad", false).InputIds);
        // invalid strategies throw NotSupportedException.
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", 123));
        Assert.Throws<NotSupportedException>(() => hf.Call(["hi"], 123));
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", "do_not_pad", 1.5));
        Assert.Throws<NotSupportedException>(() => hf.Call(["hi"], "do_not_pad", 1.5));
        Assert.Throws<NotSupportedException>(() => hf.Call("hi", "do_not_pad", truncation: "weird"));
    }

    [Fact]
    public void CallLeftPaddingAndTruncationSides()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        hf.PadToken = "<|endoftext|>";
        hf.PaddingSide = "left";
        var enc = hf.Call(["Hi", "Hello, world!"], padding: "longest");
        Assert.Equal(enc.InputIds[0].Length, enc.InputIds[1].Length);
        // Left pad puts pad id first on the short row.
        Assert.Equal(50256u, enc.InputIds[0][0]);
        Assert.Equal(0L, enc.AttentionMask![0][0]);
        hf.PaddingSide = "right";
        hf.TruncationSide = "left";
        var trunc = hf.Call("Hello, world!", truncation: "longest_first", maxLength: 2);
        Assert.Equal(2, trunc.InputIds[0].Length);
        hf.TruncationSide = "right";
        var trunc2 = hf.Call("Hello, world!", "do_not_pad", true, maxLength: 2);
        Assert.Equal(2, trunc2.InputIds[0].Length);
        var noTrunc = hf.Call("hi", "do_not_pad", false);
        Assert.Single(noTrunc.InputIds);
    }

    [Fact]
    public void CallMaxLengthLeavesNoRoomForSpecials()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"a","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"b","type_id":0}},{"SpecialToken":{"id":"c","type_id":0}}],"pair":[],"special_tokens":{"a":{"id":"a","ids":[10],"tokens":["x"]},"b":{"id":"b","ids":[11],"tokens":["x"]},"c":{"id":"c","ids":[12],"tokens":["x"]}}}""");
        var hf = tok.AsHf();
        Assert.Equal(3, hf.NumSpecialTokensToAdd());
        Assert.Throws<ArgumentException>(() => hf.Call("hi", truncation: "longest_first", maxLength: 2));
    }

    [Fact]
    public void CallTensorRequiresUniformLengths()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<ArgumentException>(() => hf.Call(["Hi", "Hello, world!"], returnTensors: "np"));
        Assert.Throws<ArgumentException>(() => hf.Call(["Hi", "Hello, world!"], returnTensors: "pt"));
    }

    [Fact]
    public void EncodePairThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<ArgumentException>(() => hf.Encode("hi", "pair"));
        Assert.Throws<ArgumentException>(() => hf.NumSpecialTokensToAdd(pair: true));
    }

    // -- Decode overloads --

    [Fact]
    public void DecodeSingles()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        var ids = hf.Encode("Hello");
        Assert.Equal("Hello", hf.Decode(ids[0]));
        Assert.Equal("Hello", hf.Decode((int)ids[0]));
        Assert.Equal("Hello", hf.Decode((long)ids[0]));
        Assert.Throws<OverflowException>(() => hf.Decode(-1));
        Assert.Throws<OverflowException>(() => hf.Decode(-1L));
        Assert.Throws<OverflowException>(() => hf.Decode((long)uint.MaxValue + 1));
        Assert.Throws<ArgumentNullException>(() => hf.Decode((uint[])null!));
        Assert.Throws<ArgumentNullException>(() => hf.Decode((int[])null!));
        Assert.Throws<ArgumentNullException>(() => hf.Decode((long[])null!));
        Assert.Throws<ArgumentNullException>(() => hf.Decode((IEnumerable<uint>)null!));
    }

    [Fact]
    public void DecodeArraysRangeValidated()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Equal("Hello", hf.Decode([15496]));
        Assert.Equal("Hello", hf.Decode([15496L]));
        Assert.Equal("Hello", hf.Decode((IEnumerable<uint>)[15496u]));
        Assert.Throws<OverflowException>(() => hf.Decode([-1]));
        Assert.Throws<OverflowException>(() => hf.Decode([15496, -2]));
        Assert.Throws<OverflowException>(() => hf.Decode([-1L]));
        Assert.Throws<OverflowException>(() => hf.Decode([(long)uint.MaxValue + 1]));
        Assert.Throws<OverflowException>(() => hf.Decode([15496L, -1L]));
    }

    [Fact]
    public void DecodeNestedAndBatch()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        var a = hf.Encode("Hello");
        var b = hf.Encode("Hi");
        Assert.Equal(["Hello", "Hi"], hf.Decode((IEnumerable<uint[]>)[a, b]));
        Assert.Equal(["Hello", "Hi"], hf.BatchDecode((IEnumerable<uint[]>)[a, b]));
        Assert.Equal(["Hello"], hf.BatchDecode((IEnumerable<int[]>)[[15496]]));
        Assert.Equal(["Hello"], hf.BatchDecode((IEnumerable<long[]>)[[15496L]]));
        Assert.Throws<ArgumentNullException>(() => hf.BatchDecode((IEnumerable<uint[]>)null!));
        Assert.Throws<ArgumentNullException>(() => hf.BatchDecode((IEnumerable<int[]>)null!));
        Assert.Throws<ArgumentNullException>(() => hf.BatchDecode((IEnumerable<long[]>)null!));
        Assert.Throws<OverflowException>(() => hf.BatchDecode((IEnumerable<int[]>)[[-1]]));
        Assert.Throws<OverflowException>(() => hf.BatchDecode((IEnumerable<long[]>)[[-1L]]));
    }

    // -- ConvertIdsToTokens --

    [Fact]
    public void ConvertIdsToTokensSingles()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Equal("Hello", hf.ConvertIdsToTokens(15496u));
        Assert.Null(hf.ConvertIdsToTokens(999999999u));
        Assert.Null(hf.ConvertIdsToTokens(-1));
        Assert.Null(hf.ConvertIdsToTokens(-1L));
        Assert.Null(hf.ConvertIdsToTokens((long)uint.MaxValue + 1));
        Assert.Equal("Hello", hf.ConvertIdsToTokens(15496, skipSpecialTokens: false));
        Assert.Equal("Hello", hf.ConvertIdsToTokens(15496L, skipSpecialTokens: false));
        // 50256 is a special id (added <|endoftext|>); skip filters it.
        Assert.Null(hf.ConvertIdsToTokens(50256, skipSpecialTokens: true));
        Assert.Equal("<|endoftext|>", hf.ConvertIdsToTokens(50256, skipSpecialTokens: false));
        Assert.Null(hf.ConvertIdsToTokens(50256L, skipSpecialTokens: true));
    }

    [Fact]
    public void ConvertIdsToTokensBatches()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Equal(["Hello", null], hf.ConvertIdsToTokens((IEnumerable<uint>)[15496u, 999999999u]));
        Assert.Equal(["Hello"], hf.ConvertIdsToTokens((IEnumerable<uint>)[15496u, 50256u], skipSpecialTokens: true));
        Assert.Equal(["Hello", null], hf.ConvertIdsToTokens((IEnumerable<int>)[15496, -1]));
        Assert.Equal(["Hello", null], hf.ConvertIdsToTokens((IEnumerable<long>)[15496L, -5L]));
        Assert.Throws<ArgumentNullException>(() => hf.ConvertIdsToTokens((IEnumerable<uint>)null!));
        Assert.Throws<ArgumentNullException>(() => hf.ConvertIdsToTokens((IEnumerable<int>)null!));
        Assert.Throws<ArgumentNullException>(() => hf.ConvertIdsToTokens((IEnumerable<long>)null!));
    }

    // -- ConvertTokensToIds --

    [Fact]
    public void ConvertTokensToIdsGuardsAndUnkFallback()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Throws<ArgumentNullException>(() => hf.ConvertTokensToIds((string)null!));
        Assert.Throws<ArgumentNullException>(() => hf.ConvertTokensToIds((IEnumerable<string>)null!));
        Assert.Equal(15496u, hf.ConvertTokensToIds("Hello"));
        Assert.Null(hf.ConvertTokensToIds("<not-a-token-xyz>"));
        hf.UnkToken = "<|endoftext|>";
        Assert.Equal(50256u, hf.ConvertTokensToIds("<not-a-token-xyz>"));
        hf.UnkToken = "<missing-unk-xyz>";
        Assert.Null(hf.ConvertTokensToIds("<not-a-token-xyz>"));
        Assert.Equal([15496u, null], hf.ConvertTokensToIds(["Hello", "<not-a-token-xyz>"]));
    }

    // -- ConvertTokensToString --

    [Fact]
    public void ConvertTokensToStringByteFallback()
    {
        using var tok = Tokenizer.FromJson("""{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""");
        var hf = tok.AsHf();
        Assert.Equal("A", hf.ConvertTokensToString(["<0x41>"]));
        Assert.Equal("Hello world", hf.ConvertTokensToString(["Hello", "▁world"]));
        Assert.Equal("world", hf.ConvertTokensToString(["▁world"]));
        Assert.Equal("<0x414>", hf.ConvertTokensToString(["<0x414>"]));
        var ex = Assert.Throws<ArgumentException>(() => hf.ConvertTokensToString(["<0xZZ>"]));
        Assert.Contains("Invalid byte-fallback token", ex.Message);
        Assert.Throws<ArgumentNullException>(() => hf.ConvertTokensToString(null!));
    }

    [Fact]
    public void ConvertTokensToStringGpt2ElseBranch()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        Assert.Equal("Hello world", hf.ConvertTokensToString(["Hello", "Ġworld"]));
        // Emoji chars miss the GPT-2 unicode→byte map, hitting the UTF-8 fallback.
        var s = hf.ConvertTokensToString(["😀"]);
        Assert.NotEmpty(s);
    }

    [Fact]
    public void TemplateWithoutSpecialTokensKeyYieldsDeferred()
    {
        using var tok = FromPp("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"cls","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[]}""");
        var hf = tok.AsHf();
        var ex = Assert.Throws<ArgumentException>(() => hf.NumSpecialTokensToAdd());
        Assert.Contains("not found in special_tokens", ex.Message);
    }

    [Fact]
    public void CallEmptyBatchCoversZeroWidth()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = tok.AsHf();
        var empty = hf.Call([], addSpecialTokens: false);
        Assert.Empty(empty.InputIds);
        var emptyT = hf.Call([], addSpecialTokens: false, returnTensors: "np");
        Assert.NotNull(emptyT.TensorIds);
        Assert.Equal(0, ((uint[,])emptyT.TensorIds!).GetLength(1));
    }

    [Fact]
    public void AddedTokensNonSpecialBranch()
    {
        var raw = Gpt2Text().Replace("\"special\":true", "\"special\":false", StringComparison.Ordinal);
        using var tok = Tokenizer.FromJson(raw);
        var hf = tok.AsHf();
        // special:false means 50256 is no longer skipped.
        Assert.Equal("<|endoftext|>", hf.ConvertIdsToTokens(50256, skipSpecialTokens: true));
        Assert.Equal(["<|endoftext|>"], hf.ConvertIdsToTokens((IEnumerable<uint>)[50256u], skipSpecialTokens: true));
        Assert.Equal("Hello<|endoftext|>", hf.Decode([15496u, 50256u], skipSpecialTokens: true));
    }
}
