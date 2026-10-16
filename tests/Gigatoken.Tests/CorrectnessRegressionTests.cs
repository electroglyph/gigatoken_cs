using System.Text;
using Gigatoken;

namespace Gigatoken.Tests;

/// <summary>Regression tests for the 0.11.0 correctness fixes.</summary>
public sealed class CorrectnessRegressionTests
{
    private static BytesSource OneBuffer(string text) =>
        new(new byte[][] { Encoding.UTF8.GetBytes(text) });

    private static (Tokenizer Tok, TiktokenCompat Compat, string Path) LoadTinySpecial()
    {
        var path = Fixtures.WriteTinyTiktokenWithSpecial(out var specials);
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        return (tok, tok.AsTiktoken(), path);
    }

    private static (Tokenizer Tok, TiktokenCompat Compat, string Path) LoadTinyThreeSpecial()
    {
        var path = Fixtures.WriteTinyTiktoken();
        var specials = new Dictionary<string, uint>
        {
            ["<|a|>"] = 258, ["<|b|>"] = 259, ["<|c|>"] = 260,
        };
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        return (tok, tok.AsTiktoken(), path);
    }

    // -- PadRows throws when MaxLength leaves no room for affixes ---

    [Fact]
    public void PadRowsExtraExceedsMaxThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = OneBuffer("hi");
        var ex = Assert.Throws<ArgumentException>(() =>
            tok.EncodeBatchPadded(src, new PadTruncateOptions(
                PadId: 0, MaxLength: 1, Truncate: true, Prefix: [1u, 2u])));
        Assert.Contains("leaves no room", ex.Message);
    }

    // -- Tiktoken inference is case-sensitive ---

    [Fact]
    public void TiktokenInferenceIsCaseSensitive()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"case_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var src = Fixtures.WriteTinyTiktoken();
            // Wrong-case stem: must fail inference, never guess cl100k_base.
            var wrong = Path.Combine(dir, "CL100K_base.tiktoken");
            File.Copy(src, wrong);
            var ex = Assert.Throws<ArgumentException>(() => Tokenizer.FromTiktoken(wrong));
            Assert.Contains("Cannot infer", ex.Message);
            File.Delete(src);
            File.Delete(wrong);
        }
        finally { Directory.Delete(dir, true); }
    }

    // -- Local directory wins over hub ---

    [Fact]
    public void LocalDirectoryWinsOverHub()
    {
        var root = Path.Combine(Path.GetTempPath(), $"local_{Guid.NewGuid():N}");
        var repo = Path.Combine(root, "user", "repo");
        Directory.CreateDirectory(repo);
        File.Copy(Fixtures.Gpt2Json, Path.Combine(repo, "tokenizer.json"));
        var prev = Directory.GetCurrentDirectory();
        try
        {
            // "user/repo" looks like a hub repo id; the local directory must
            // resolve first (no network involved).
            Directory.SetCurrentDirectory(root);
            using var tok = Tokenizer.FromPretrained(Path.Combine("user", "repo"));
            Assert.NotEmpty(tok.Encode("hello"));
        }
        finally
        {
            Directory.SetCurrentDirectory(prev);
            Directory.Delete(root, true);
        }
    }

    // -- Null-argument ParamName precision ---

    [Fact]
    public void NullTextsReportsParamName()
    {
        using var tok = Fixtures.LoadGpt2();
        var ex = Assert.Throws<ArgumentNullException>(() =>
            tok.EncodeBatch((IEnumerable<string>)null!));
        Assert.Equal("texts", ex.ParamName);
    }

    [Fact]
    public void NullElementReportsIndexedParamName()
    {
        using var tok = Fixtures.LoadGpt2();
        var ex = Assert.Throws<ArgumentNullException>(() =>
            tok.EncodeBatch(new[] { "a", null!, "b" }));
        Assert.Equal("texts[1]", ex.ParamName);
    }

    [Fact]
    public void BackendFromTiktokenNullGuards()
    {
        var ex1 = Assert.Throws<ArgumentNullException>(() =>
            BpeTokenizer.FromTiktoken(null!, "gpt2"));
        Assert.Equal("path", ex1.ParamName);
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            var ex2 = Assert.Throws<ArgumentNullException>(() =>
                BpeTokenizer.FromTiktoken(path, null!));
            Assert.Equal("pretokenizer", ex2.ParamName);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void HubNullGuards()
    {
        var ex1 = Assert.Throws<ArgumentNullException>(() => Hub.GetFile(null!));
        Assert.Equal("repoId", ex1.ParamName);
        var ex2 = Assert.Throws<ArgumentNullException>(() => Hub.LooksLikeRepoId(null!));
        Assert.Equal("name", ex2.ParamName);
    }

    // -- Empty padded tensor keeps (0, max) ---

    [Fact]
    public void EmptyBatchKeepsPadWidth()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = new HfCompat(tok);
        hf.PadToken = "<|endoftext|>";
        var enc = hf.Call([], padding: "max_length", maxLength: 8, returnTensors: "np");
        Assert.NotNull(enc.TensorIds);
        Assert.Equal(0, enc.TensorIds!.GetLength(0));
        Assert.Equal(8, enc.TensorIds!.GetLength(1));
        var plain = hf.Call([], returnTensors: "np");
        Assert.NotNull(plain.TensorIds);
        Assert.Equal(0, plain.TensorIds!.GetLength(0));
        Assert.Equal(0, plain.TensorIds!.GetLength(1));
    }

    // -- Astral-plane round-trip ---

    [Fact]
    public void AstralPlaneRoundTrips()
    {
        using var tok = Fixtures.LoadGpt2();
        var hf = new HfCompat(tok);
        var ids = tok.Encode("😀");
        var pieces = hf.ConvertIdsToTokens(ids).Select(s => s!).ToList();
        Assert.Equal("😀", hf.ConvertTokensToString(pieces));
    }

    // -- Explicit disallowed set wins over "all" ---

    [Fact]
    public void ExplicitDisallowedSetWins()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            // Empty set with default "all": nothing is disallowed, but the
            // special still cannot encode as ordinary text -> NotSupported
            // (Python NotImplementedError), NOT ArgumentException.
            Assert.Throws<NotSupportedException>(() =>
                compat.Encode("A<|endoftext|>B", "none", null, "all", new HashSet<string>()));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void DisallowedSetNarrowsAll()
    {
        var (tok, compat, path) = LoadTinyThreeSpecial();
        try
        {
            // allowed={b}, disallowedSet={a}, text has <|c|>: c is merely
            // unallowed (NotSupported), not disallowed (ArgumentException).
            Assert.Throws<NotSupportedException>(() =>
                compat.Encode("X<|c|>Y", "none", new[] { "<|b|>" }, "all", new[] { "<|a|>" }));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Explicit allowed set wins over "all" ---

    [Fact]
    public void ExplicitAllowedSetWins()
    {
        var (tok, compat, path) = LoadTinyThreeSpecial();
        try
        {
            // allowed=all+{a}: b must be rejected, not encoded as a special.
            Assert.Throws<ArgumentException>(() =>
                compat.Encode("X<|b|>Y", "all", new[] { "<|a|>" }, "all", null));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Forbid-cache keys are collision-free ---

    private static (Tokenizer Tok, TiktokenCompat Compat, string Path) LoadNulSpecials()
    {
        var path = Fixtures.WriteTinyTiktoken();
        var specials = new Dictionary<string, uint>
        {
            ["X"] = 258, ["Y"] = 259, ["X\0Y"] = 260,
        };
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        return (tok, tok.AsTiktoken(), path);
    }

    [Fact]
    public void NulBearingScansDoNotCollide()
    {
        var (tok, compat, path) = LoadNulSpecials();
        try
        {
            // Prime scan {"X\0Y"}: "X" is allowed, so it encodes as the
            // special itself. Scan {X,Y} must use its own matcher and
            // reject "X" (stale-matcher bug returned [258] silently).
            Assert.Equal([258u], compat.Encode("X", "all", null, "none", new[] { "X\0Y" }));
            // Scan {X,Y} must use its own matcher and reject "X".
            Assert.Throws<ArgumentException>(() =>
                compat.Encode("X", "all", null, "none", new[] { "X", "Y" }));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void ReverseOrderScanUsesFreshMatcher()
    {
        var (tok, compat, path) = LoadNulSpecials();
        try
        {
            // Prime scan {X,Y} (rejects "Y"), then scan {"X\0Y"} must not see
            // a stale matcher: "Y" is allowed, so it encodes as special 259
            // (stale matcher threw here instead).
            Assert.Throws<ArgumentException>(() =>
                compat.Encode("Y", "all", null, "none", new[] { "X", "Y" }));
            Assert.Equal([259u], compat.Encode("Y", "all", null, "none", new[] { "X\0Y" }));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Returned byte arrays are copies ---

    [Fact]
    public void DecodeSingleTokenBytesReturnsCopy()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            var b1 = compat.DecodeSingleTokenBytes(65u);
            b1[0] ^= 0xFF;
            Assert.Equal((byte)'A', compat.DecodeSingleTokenBytes(65u)[0]);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    [Fact]
    public void TokenByteValuesReturnsCopies()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            var v1 = compat.TokenByteValues();
            var before = v1[65].ToArray();
            v1[65][0] ^= 0xFF;
            Assert.Equal(before, compat.TokenByteValues()[65]);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Cache churn never disables in-use matchers ---

    [Fact]
    public void MatcherCacheChurnStaysUsable()
    {
        var path = Fixtures.WriteTinyTiktoken();
        var specials = new Dictionary<string, uint>();
        for (var i = 0; i < 6; i++) specials[$"SPECIAL{i}"] = (uint)(258 + i);
        var tok = Tokenizer.FromTiktoken(path, pretokenizer: "gpt2", specialTokens: specials);
        var compat = tok.AsTiktoken();
        try
        {
            var names = specials.Keys.ToArray();
            // 40 distinct scans: forces repeated eviction (cap 32). Must not
            // throw ObjectDisposedException or return stale results.
            for (var mask = 1; mask <= 40; mask++)
            {
                var set = names.Where((_, i) => (mask & (1 << (i % 6))) != 0).ToArray();
                compat.Encode("plain text", "all", null, "none", set);
            }
            Assert.Throws<ArgumentException>(() =>
                compat.Encode("SPECIAL0!", "all", null, "none", new[] { "SPECIAL0" }));
            Assert.Contains(258u, compat.Encode(
                "SPECIAL0!", "all", new[] { "SPECIAL0" }, "none", null));
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Special-hit read path ---

    [Fact]
    public void SpecialHitReadPathSurfacesCleanError()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            // Forbid hit exercises ReadSpecialHits (rc-checked); the public
            // mapping stays ArgumentException, never IndexOutOfRange.
            var ex = Assert.Throws<ArgumentException>(() =>
                compat.Encode("A<|endoftext|>B"));
            Assert.Contains("<|endoftext|>", ex.Message);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- EncodeBatch null element ParamName ---

    [Fact]
    public void EncodeBatchNullElementParamName()
    {
        var (tok, compat, path) = LoadTinySpecial();
        try
        {
            var ex = Assert.Throws<ArgumentNullException>(() =>
                compat.EncodeBatch(new[] { "AB", null! }));
            Assert.Equal("texts[1]", ex.ParamName);
        }
        finally { tok.Dispose(); File.Delete(path); }
    }

    // -- Concurrent encoding smoke ---

    [Fact]
    public void ConcurrentEncodeAgrees()
    {
        using var tok = Fixtures.LoadGpt2();
        var expected = tok.Encode("Hello, world!");
        var results = new uint[16][];
        Parallel.For(0, 16, i => { results[i] = tok.Encode("Hello, world!"); });
        foreach (var r in results) Assert.Equal(expected, r);
    }

    // -- Cache budget validation ---

    [Fact]
    public void NegativeBudgetThrows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EncodeCache.SetMaxBytes(-1));
    }

    [Fact]
    public void NullAndValidBudgetAccepted()
    {
        try
        {
            EncodeCache.SetMaxBytes(1 << 20);
            EncodeCache.SetMaxBytes(null);
        }
        finally { EncodeCache.SetMaxBytes(null); }
    }
}
