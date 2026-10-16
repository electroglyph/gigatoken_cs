using System.Reflection;
using System.Text;
using Gigatoken.Interop;

namespace Gigatoken.Tests;

// Coverage gap-filler for Tokenizer.cs / HubCacheTrain.cs / TiktokenCompat.cs.
// Tests only: no source changes.
public sealed class CoverageCoreCopyDisposeTests
{
    private const string TinySpJson = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""";

    [Fact]
    public void CopyNullThrows()
    {
        Assert.Throws<ArgumentNullException>(() => new Tokenizer((Tokenizer)null!));
    }

    [Fact]
    public void CopyDisposedSourceThrows()
    {
        var tok = Fixtures.LoadGpt2();
        tok.Dispose();
        Assert.Throws<ObjectDisposedException>(() => new Tokenizer(tok));
    }

    [Fact]
    public void DisposeIsIdempotentAndUseAfterThrows()
    {
        var tok = Fixtures.LoadGpt2();
        tok.Dispose();
        tok.Dispose();
        Assert.Throws<ObjectDisposedException>(() => tok.Encode("hi"));
    }

    [Fact]
    public void AddRefAfterDisposeThrows()
    {
        // SharedBackend refs are 0 once the sole owner disposes; the copy
        // ctor's disposed-flag check is bypassed here via reflection so the
        // AddRef fail-fast (ObjectDisposedException) is exercised.
        var tok = Fixtures.LoadGpt2();
        tok.Dispose();
        var field = typeof(Tokenizer).GetField("_disposed", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(field);
        try
        {
            field.SetValue(tok, 0);
            Assert.Throws<ObjectDisposedException>(() => new Tokenizer(tok));
        }
        finally
        {
            field.SetValue(tok, 1);
        }
    }

    [Fact]
    public void CopyCopiesNamedAndAdditionalSpecials()
    {
        var dir = Directory.CreateTempSubdirectory("gt_cov_").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"), TinySpJson);
            File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), """
                {"bos_token": {"content": "<s>"}, "pad_token": "<pad>",
                 "additional_special_tokens": ["<extra1>", {"content": "<extra2>"}]}
                """);
            using var tok = Tokenizer.FromPretrained(dir);
            using var copy = new Tokenizer(tok);
            Assert.Equal("<s>", copy.NamedSpecials["bos_token"]);
            Assert.Equal("<pad>", copy.NamedSpecials["pad_token"]);
            Assert.Equal(["<extra1>", "<extra2>"], copy.AdditionalSpecials);
            Assert.Equal(tok.Encode("Hi"), copy.Encode("Hi"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void BackendWrapperNullGuards()
    {
        Assert.Throws<ArgumentNullException>(() => new Tokenizer((BpeTokenizer)null!));
        Assert.Throws<ArgumentNullException>(() => new Tokenizer((SentencePieceTokenizer)null!));
    }

    [Fact]
    public void WrapBpeBackend()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            var backend = BpeTokenizer.FromTiktoken(path, "gpt2");
            using var tok = new Tokenizer(backend);
            Assert.Equal([256u], tok.Encode("AB"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void WrapSentencePieceBackend()
    {
        var dir = Directory.CreateTempSubdirectory("gt_cov_").FullName;
        try
        {
            var jsonPath = Path.Combine(dir, "tokenizer.json");
            File.WriteAllText(jsonPath, TinySpJson);
            var backend = SentencePieceTokenizer.FromHf(jsonPath);
            using var tok = new Tokenizer(backend);
            Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
            Assert.Equal([3u], tok.Encode("Hi"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }
}

public sealed class CoverageCoreThrowIfDisposedTests
{
    [Fact]
    public void AllPublicMethodsThrowAfterDispose()
    {
        var tok = Fixtures.LoadGpt2();
        var existingFile = Path.GetTempFileName();
        var missingPath = Path.Combine(Path.GetTempPath(), $"gt_missing_{Guid.NewGuid():N}.txt");
        using var byteSrc = new BytesSource(new byte[][] { Encoding.UTF8.GetBytes("hi") });
        using var fileSrc = new TextFileSource([missingPath]);
        var padOpts = new PadTruncateOptions(PadId: 0);
        try
        {
            tok.Dispose();
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.Kind; });
            Assert.Throws<ObjectDisposedException>(() => tok.Encode("hi"));
            byte[] raw = [104, 105];
            Assert.Throws<ObjectDisposedException>(() => tok.Encode(raw.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeNoNormalize("hi"));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatch(["hi"]));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatch(new[] { Encoding.UTF8.GetBytes("hi") }));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatch(byteSrc));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatchPadded(["hi"], padOpts));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatchPadded(new[] { Encoding.UTF8.GetBytes("hi") }, padOpts));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeBatchPadded(byteSrc, padOpts));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles(fileSrc));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles(existingFile));
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles((IEnumerable<string>)new[] { existingFile }));
            uint[] u = [0]; int[] i = [0]; long[] l = [0];
            Assert.Throws<ObjectDisposedException>(() => tok.Decode(u.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.Decode(i.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.Decode(l.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.DecodeToString(u.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.DecodeToString(i.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => tok.DecodeToString(l.AsSpan()));
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.VocabSize; });
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.Vocab; });
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.VocabEntries; });
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.Merges; });
            Assert.Throws<ObjectDisposedException>(() => { _ = tok.CacheEntries; });
            Assert.Throws<ObjectDisposedException>(() => tok.AsHf());
            Assert.Throws<ObjectDisposedException>(() => tok.AsTiktoken());
        }
        finally { File.Delete(existingFile); }
    }
}

public sealed class CoverageCorePadRowsTests
{
    private static BytesSource OneBuffer(string text) =>
        new(new byte[][] { Encoding.UTF8.GetBytes(text) });

    [Fact]
    public void NegativeMaxLengthThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("hi");
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tok.EncodeBatchPadded(src, new PadTruncateOptions(PadId: 0, MaxLength: -1)));
    }

    [Fact]
    public void PadToMaxLengthRequiresMaxLength()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("hi");
        Assert.Throws<ArgumentException>(() =>
            tok.EncodeBatchPadded(src, new PadTruncateOptions(PadId: 0, PadToMaxLength: true)));
    }

    [Fact]
    public void TruncateRequiresMaxLength()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("hi");
        Assert.Throws<ArgumentException>(() =>
            tok.EncodeBatchPadded(src, new PadTruncateOptions(PadId: 0, Truncate: true)));
    }

    [Fact]
    public void MaxLengthAloneIsInert()
    {
        // MaxLength without Truncate/PadToMaxLength is inert (native
        // pad_truncate_matrix uses cap=MAX, width=longest). The managed
        // BytesSource fallback used to throw here; it must return the full row.
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("Hello, world!");
        var (matrix, lengths) = tok.EncodeBatchPadded(src, new PadTruncateOptions(PadId: 0, MaxLength: 1));
        Assert.True(lengths[0] > 1);
        Assert.Equal(lengths[0], matrix.GetLength(1));
    }

    [Fact]
    public void CapZeroKeepsOnlyAffixes()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("Hello, world!");
        var (matrix, lengths) = tok.EncodeBatchPadded(src, new PadTruncateOptions(
            PadId: 0, MaxLength: 2, Truncate: true, Prefix: [1u], Suffix: [2u]));
        Assert.Equal([2L], lengths);
        Assert.Equal(1u, matrix[0, 0]);
        Assert.Equal(2u, matrix[0, 1]);
    }

    [Fact]
    public void TruncateSides()
    {
        using var tok = Fixtures.LoadGpt2();
        var full = tok.Encode("Hello, world!");
        Assert.True(full.Length > 2);
        using var right = OneBuffer("Hello, world!");
        var (mRight, _) = tok.EncodeBatchPadded(right,
            new PadTruncateOptions(PadId: 0, MaxLength: 2, Truncate: true));
        Assert.Equal(full[0], mRight[0, 0]);
        Assert.Equal(full[1], mRight[0, 1]);
        using var left = OneBuffer("Hello, world!");
        var (mLeft, _) = tok.EncodeBatchPadded(left,
            new PadTruncateOptions(PadId: 0, MaxLength: 2, Truncate: true, TruncateLeft: true));
        Assert.Equal(full[^2], mLeft[0, 0]);
        Assert.Equal(full[^1], mLeft[0, 1]);
    }

    [Fact]
    public void PrefixSuffixAndPadLeft()
    {
        using var tok = Fixtures.LoadGpt2();
        var hiId = tok.Encode("Hi")[0];
        using var src = new BytesSource(new byte[][]
            { Encoding.UTF8.GetBytes("Hi"), Encoding.UTF8.GetBytes("Hello, world!") });
        var (matrix, lengths) = tok.EncodeBatchPadded(src, new PadTruncateOptions(
            PadId: 0, PadLeft: true, Prefix: [7u], Suffix: [9u]));
        Assert.Equal(2, matrix.GetLength(0));
        Assert.Equal(3L, lengths[0]);
        Assert.Equal(6L, lengths[1]);
        Assert.Equal(6, matrix.GetLength(1));
        Assert.Equal(0u, matrix[0, 0]);
        Assert.Equal(7u, matrix[0, 3]);
        Assert.Equal(hiId, matrix[0, 4]);
        Assert.Equal(9u, matrix[0, 5]);
    }
}

public sealed class CoverageCoreLoadRouteTests
{
    private const string TinySpJson = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""";

    private static string MakeTempDir() => Directory.CreateTempSubdirectory("gt_cov_").FullName;

    private static void DeleteDir(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch { }
    }

    [Fact]
    public void FromJsonSpanOverload()
    {
        using var tok = Tokenizer.FromJson(Encoding.UTF8.GetBytes(TinySpJson));
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal([3u], tok.Encode("Hi"));
    }

    [Fact]
    public void FromSentencePieceSpanOverload()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp.model");
        using var tok = Tokenizer.FromSentencePiece(File.ReadAllBytes(modelPath));
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal(512, tok.VocabSize);
    }

    [Fact]
    public void EncodeSpanAndDecodeToStringLongOverloads()
    {
        using var tok = Fixtures.LoadGpt2();
        var ids = tok.Encode("Hello, world!");
        Assert.Equal(ids, tok.Encode(Encoding.UTF8.GetBytes("Hello, world!").AsSpan()));
        Assert.Equal(tok.DecodeToString(ids),
            tok.DecodeToString(ids.Select(v => (long)v).ToArray().AsSpan()));
    }

    private static string WriteR50kFile(string dir)
    {
        var src = Fixtures.WriteTinyTiktoken();
        try
        {
            var dst = Path.Combine(dir, "r50k_base.tiktoken");
            File.Copy(src, dst);
            return dst;
        }
        finally { File.Delete(src); }
    }

    [Fact]
    public void FromTiktokenInferredSchemeAndDefaultSpecials()
    {
        var dir = MakeTempDir();
        try
        {
            var dst = WriteR50kFile(dir);
            using var tok = Tokenizer.FromTiktoken(dst);
            Assert.Equal([256u], tok.Encode("AB"));
            Assert.Equal(50256u, tok.AsTiktoken().EotToken);
        }
        finally { DeleteDir(dir); }
    }

    [Fact]
    public void FromTiktokenInferredSchemeExplicitSpecials()
    {
        var dir = MakeTempDir();
        try
        {
            var dst = WriteR50kFile(dir);
            using var tok = Tokenizer.FromTiktoken(dst,
                specialTokens: new Dictionary<string, uint> { ["<|eot|>"] = 50000 });
            Assert.Equal([256u], tok.Encode("AB"));
            Assert.True(tok.SpecialTokensMap().ContainsKey("<|eot|>"));
        }
        finally { DeleteDir(dir); }
    }

    [Fact]
    public void FromPretrainedTiktokenFileRoute()
    {
        var dir = MakeTempDir();
        try
        {
            var dst = WriteR50kFile(dir);
            using var tok = Tokenizer.FromPretrained(dst);
            Assert.Equal(TokenizerKind.Bpe, tok.Kind);
            Assert.Equal([256u], tok.Encode("AB"));
        }
        finally { DeleteDir(dir); }
    }

    [Fact]
    public void FromPretrainedDirRouteLoads()
    {
        var dir = MakeTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"), TinySpJson);
            using var tok = Tokenizer.FromPretrained(dir);
            Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
            Assert.Equal([3u], tok.Encode("Hi"));
        }
        finally { DeleteDir(dir); }
    }

    [Fact]
    public void FromPretrainedInvalidConfigFallsBack()
    {
        var dir = MakeTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"), TinySpJson);
            File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), "{ not valid json {{{");
            using var tok = Tokenizer.FromPretrained(dir);
            Assert.Equal([3u], tok.Encode("Hi"));
        }
        finally { DeleteDir(dir); }
    }

    [Fact]
    public void FromPretrainedMissingThrows()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"gt_nope_{Guid.NewGuid():N}", "tokenizer.json");
        var ex = Assert.Throws<FileNotFoundException>(() => Tokenizer.FromPretrained(missing));
        Assert.Equal($"No such file, directory, or repo id: {missing}", ex.Message);
    }

    // huggingface.co answers 404 (→ HubFileNotFoundException) or 401
    // (→ UnauthorizedAccessException) for nonexistent repos depending on
    // edge/auth behavior; either is the clean typed failure.
    private static void AssertHubMissing(Action action)
    {
        var ex = Record.Exception(action);
        Assert.True(ex is HubFileNotFoundException || ex is UnauthorizedAccessException,
            $"Expected a typed hub-missing error, got {ex?.GetType().FullName}: {ex?.Message}");
    }

    [Fact]
    public void FromPretrainedBogusRepoThrowsHubFileNotFound()
    {
        AssertHubMissing(() =>
            Tokenizer.FromPretrained("bogus-org-xyz/bogus-repo-xyz-123"));
    }

    [Fact]
    public void KimiDispatchReadsAddedTokensDecoder()
    {
        var dir = MakeTempDir();
        try
        {
            var src = Fixtures.WriteTinyTiktoken();
            try { File.Copy(src, Path.Combine(dir, "tiktoken.model")); }
            finally { File.Delete(src); }
            File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), """
                {"tokenizer_class": "TikTokenTokenizer",
                 "bos_token": "<|endoftext|>",
                 "additional_special_tokens": ["<|fim|>"],
                 "added_tokens_decoder": {
                   "258": {"content": "<|endoftext|>"},
                   "259": {"content": "<|fim|>"},
                   "oops": "not-an-object",
                   "261": {"content": 42}}}
                """);
            using var tok = Tokenizer.FromPretrained(dir);
            Assert.Equal(TokenizerKind.Bpe, tok.Kind);
            Assert.Equal(260, tok.VocabSize);
            Assert.Contains(258u, tok.Encode("A<|endoftext|>B"));
            Assert.Equal("<|endoftext|>", tok.NamedSpecials["bos_token"]);
            Assert.Equal(["<|fim|>"], tok.AdditionalSpecials);
        }
        finally { DeleteDir(dir); }
    }
}

public sealed class CoverageCoreHubCacheTests
{
    [Fact]
    public void HubGetFileErrorMapping()
    {
        var ex = Record.Exception(() =>
            Hub.GetFile("definitely-not-a-real-repo-xyz-123", "tokenizer.json"));
        if (ex is HubFileNotFoundException nf)
        {
            Assert.NotNull(nf.Message);
            Assert.NotNull(nf.InnerException);
            Assert.IsAssignableFrom<FileNotFoundException>(nf.InnerException);
        }
        else
        {
            // huggingface.co may answer 401 instead of 404 for nonexistent
            // repos; that path must also surface as a typed hub error.
            Assert.IsType<UnauthorizedAccessException>(ex);
        }
    }

    [Fact]
    public void SetMaxBytesNegativeGuard()
    {
        var before = EncodeCache.GetMaxBytes();
        Assert.Throws<ArgumentOutOfRangeException>(() => EncodeCache.SetMaxBytes(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EncodeCache.SetMaxBytes(long.MinValue));
        Assert.Equal(before, EncodeCache.GetMaxBytes());
    }

    [Fact]
    public void SaturatingGetMaxBytes()
    {
        var before = EncodeCache.GetMaxBytes();
        try
        {
            NativeBridge.SetMaxCacheBytes(unchecked((nuint)ulong.MaxValue));
            Assert.Equal((long?)long.MaxValue, EncodeCache.GetMaxBytes());
        }
        finally { EncodeCache.SetMaxBytes(before); }
    }
}

public sealed class CoverageCoreTrainTests
{
    private static byte[] TinyCorpus() =>
        Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("the cat sat on the mat. ", 10)));

    [Fact]
    public void TinyTrainByteBlob()
    {
        var (vocab, merges) = BpeTrainer.Train(TinyCorpus(), 258, []);
        Assert.Equal(258, vocab.Count);
        Assert.Equal(2, merges.Count);
    }

    [Fact]
    public void TinyTrainFileSource()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gt_corpus_{Guid.NewGuid():N}.txt");
        try
        {
            File.WriteAllBytes(path, TinyCorpus());
            using var src = new TextFileSource([path]);
            var (vocab, merges) = BpeTrainer.Train(src, 258, []);
            Assert.Equal(258, vocab.Count);
            Assert.Equal(2, merges.Count);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ByteBlobGuards()
    {
        byte[] corpus = [65, 66];
        Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train((byte[])null!, 258, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => BpeTrainer.Train(corpus, -1, []));
        Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(corpus, 258, (IEnumerable<string>)null!));
        Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(corpus, 258, new[] { (string)null! }));
    }

    [Fact]
    public void PathGuards()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gt_corpus_{Guid.NewGuid():N}.txt");
        try
        {
            File.WriteAllBytes(path, TinyCorpus());
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train((string)null!, 258, []));
            Assert.Throws<ArgumentOutOfRangeException>(() => BpeTrainer.Train(path, -1, []));
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(path, 258, (IEnumerable<string>)null!));
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(path, 258, new[] { (string)null! }));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void FileSourceGuards()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gt_corpus_{Guid.NewGuid():N}.txt");
        try
        {
            File.WriteAllBytes(path, TinyCorpus());
            using var src = new TextFileSource([path]);
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train((IFileSource)null!, 258, []));
            Assert.Throws<ArgumentOutOfRangeException>(() => BpeTrainer.Train(src, -1, []));
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(src, 258, (IEnumerable<string>)null!));
            Assert.Throws<ArgumentNullException>(() => BpeTrainer.Train(src, 258, new[] { (string)null! }));
        }
        finally { File.Delete(path); }
    }
}

public sealed class CoverageCorePretokenizerTests
{
    [Fact]
    public void ExhaustionAndSinglePass()
    {
        using var it = Pretokenizer.Pretokenize(Encoding.UTF8.GetBytes("Hello, world!"));
        Assert.False(it.IsExhausted);
        var pieces = it.ToList().Select(m => Encoding.UTF8.GetString(m.Span)).ToArray();
        Assert.Equal(["Hello", ",", " world", "!"], pieces);
        Assert.True(it.IsExhausted);
        Assert.Empty(it.ToList());

        using var it2 = Pretokenizer.Pretokenize(Encoding.UTF8.GetBytes("ab cd"));
        Assert.Equal("ab cd", string.Concat(it2.Select(m => Encoding.UTF8.GetString(m.Span))));
        Assert.True(it2.IsExhausted);
        Assert.Empty(it2);
    }

    [Fact]
    public void DisposedThrowsAndDisposeIsIdempotent()
    {
        var it = Pretokenizer.Pretokenize(Encoding.UTF8.GetBytes("hello"));
        it.Dispose();
        Assert.Throws<ObjectDisposedException>(() => it.ToList());
        Assert.Throws<ObjectDisposedException>(() => { foreach (var _ in it) { } });
        it.Dispose();
    }

    [Fact]
    public void NonGenericEnumerator()
    {
        using var it = Pretokenizer.Pretokenize(Encoding.UTF8.GetBytes("ab cd"));
        var enumerator = ((System.Collections.IEnumerable)it).GetEnumerator();
        var pieces = new List<string>();
        while (enumerator.MoveNext())
            pieces.Add(Encoding.UTF8.GetString(((ReadOnlyMemory<byte>)enumerator.Current).Span));
        Assert.Equal("ab cd", string.Concat(pieces));
        Assert.True(it.IsExhausted);
    }
}

public sealed class CoverageCoreTiktokenCompatTests
{
    private static (Tokenizer Tok, TiktokenCompat Compat, string Path) LoadTinySpecial()
    {
        var path = Fixtures.WriteTinyTiktokenWithSpecial(out var specials);
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        return (tok, tok.AsTiktoken(), path);
    }

    [Fact]
    public void NameAndTokenizerPropsAndCtorGuards()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            Assert.Equal("gigatoken", compat.Name);
            var custom = new TiktokenCompat(tok, "custom");
            Assert.Equal("custom", custom.Name);
            Assert.Same(tok, custom.Tokenizer);
            Assert.True(compat.IsSpecialToken(258u));
            Assert.True(compat.IsSpecial(258u));
            Assert.False(compat.IsSpecialToken(0u));
            Assert.Throws<ArgumentNullException>(() => new TiktokenCompat(null!));
            Assert.Throws<ArgumentNullException>(() => new TiktokenCompat(tok, null!));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void SingleStringAllowedAndDisallowed()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            Assert.Contains(258u, compat.Encode("A<|endoftext|>B", allowedSpecial: "<|endoftext|>"));
            Assert.Equal([256u], compat.Encode("AB", disallowedSpecial: "<|endoftext|>"));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void NeitherAllowedNorDisallowedIsNotSupported()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            var ex = Assert.Throws<NotSupportedException>(() =>
                compat.Encode("A<|endoftext|>B", "none", null, "none", null));
            Assert.Contains("allowedSpecial", ex.Message);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void DecodeErrorStrategies()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            Assert.Equal("A", compat.Decode([65u], "strict"));
            Assert.Equal("A", compat.Decode([65u], "replace"));
            Assert.Equal("A", compat.Decode([65u], "ignore"));
            Assert.Equal("\uFFFD", compat.Decode([255u], "replace"));
            Assert.Equal("", compat.Decode([255u], "ignore"));
            Assert.Throws<DecoderFallbackException>(() => compat.Decode([255u], "strict"));
            Assert.Throws<ArgumentException>(() => compat.Decode([65u], "bogus"));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void CachedSingleTokenAndByteValuePaths()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            Assert.Equal(compat.EncodeSingleToken("AB"), compat.EncodeSingleToken("AB"));
            Assert.Equal(compat.EncodeSingleToken("AB"u8.ToArray()), compat.EncodeSingleToken("AB"u8.ToArray()));
            Assert.Equal(compat.DecodeSingleTokenBytes(256u), compat.DecodeSingleTokenBytes(256u));
            var first = compat.DecodeTokensBytes([256u]);
            var second = compat.DecodeTokensBytes([256u]);
            Assert.Equal(first.Count, second.Count);
            Assert.Equal(first[0], second[0]);
            var tbv1 = compat.TokenByteValues();
            var tbv2 = compat.TokenByteValues();
            Assert.Equal(tbv1.Count, tbv2.Count);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void MatcherCacheEviction()
    {
        var specials = Enumerable.Range(0, 40).ToDictionary(i => $"<|s{i:D2}|>", i => (uint)(258 + i));
        var path = Fixtures.WriteTinyTiktoken();
        using var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        try
        {
            var compat = tok.AsTiktoken();
            var keys = specials.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
            for (var k = 0; k < 34; k++)
                Assert.Equal([256u], compat.Encode("AB",
                    allowedSpecialSet: keys.Take(k).ToArray(), disallowedSpecial: "none"));
        }
        finally { File.Delete(path); }
    }
}

public sealed class CoverageCoreNativeArmsTests
{
    private static byte[] TinyCorpus() =>
        System.Text.Encoding.UTF8.GetBytes("hello world hello there world hello\n");

    [Fact]
    public void InvalidTieBreaking_Throws()
    {
        var ex = Assert.ThrowsAny<Exception>(() =>
            BpeTrainer.Train(TinyCorpus(), 260, [], (TieBreaking)99));
        Assert.IsType<ArgumentException>(ex);
    }

    [Theory]
    [InlineData(TieBreaking.HuggingFace)]
    [InlineData(TieBreaking.RawTokenIds)]
    [InlineData(TieBreaking.AssembledBytes)]
    public void AllTieBreakings_Train(TieBreaking tie)
    {
        var (vocab, merges) = BpeTrainer.Train(TinyCorpus(), 260, [], tie);
        Assert.NotEmpty(vocab);
        Assert.NotNull(merges);
    }

    [Fact]
    public void InvalidRepoType_ThrowsWithoutNetwork()
    {
        // Native validates repo_type before any download: offline-safe.
        Assert.Throws<ArgumentException>(() =>
            Hub.GetFile("gpt2", "tokenizer.json", (RepoType)99));
    }

    [Fact]
    public void HfToken_MatchesAmbientSetup()
    {
        // In-process env writes do NOT propagate to native getenv in this
        // runtime (verified: managed sees the write, libc getenv does not),
        // so the token cannot be steered from the test. Assert the native
        // result matches the ambient setup instead: parent-process
        // HF_TOKEN/HUGGING_FACE_HUB_TOKEN, else ~/.cache/huggingface/token.
        // (Covers whichever GetHfToken arm the environment dictates.)
        string? expected = Environment.GetEnvironmentVariable("HF_TOKEN");
        expected = string.IsNullOrEmpty(expected)
            ? Environment.GetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN") : expected;
        if (string.IsNullOrEmpty(expected))
        {
            var tokenFile = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "huggingface", "token");
            if (File.Exists(tokenFile))
                expected = File.ReadAllText(tokenFile).Trim();
        }
        if (string.IsNullOrEmpty(expected))
            expected = null;
        Assert.Equal(expected, Hub.GetToken());
    }

    [Fact]
    public void InvalidSentencePiece_Throws()
    {
        Assert.Throws<TokenizerLoadException>(() =>
            Tokenizer.FromSentencePiece(new byte[] { 1, 2, 3, 4 }));
    }
}
