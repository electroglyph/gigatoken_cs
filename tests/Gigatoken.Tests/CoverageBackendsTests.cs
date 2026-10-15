namespace Gigatoken.Tests;

using System.Text;

/// <summary>
/// Coverage-focused tests for Backends/Options/Sources/SubstringMatcher/Exceptions gaps.
/// </summary>
public sealed class CoverageBackendsTests
{
    private const string TinySpJson = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""";

    private static string WriteSpJson()
    {
        var path = Path.Combine(Path.GetTempPath(), $"covsp_{Guid.NewGuid():N}.json");
        File.WriteAllText(path, TinySpJson);
        return path;
    }

    private static string SpModelPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp.model");

    private static string SpNoPrefixPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp_noprefix.model");

    // -- BpeTokenizer.FromTiktoken -------------------------------------------------

    [Fact]
    public void CoverageBackends_BpeFromTiktoken_NullAndSpecials()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            using var bpeNull = BpeTokenizer.FromTiktoken(path, "gpt2", null);
            Assert.Equal([256u], bpeNull.Encode("AB"));
            Assert.Equal(TokenizerKind.Bpe, bpeNull.Kind);
        }
        finally { File.Delete(path); }

        var path2 = Fixtures.WriteTinyTiktokenWithSpecial(out var specials);
        try
        {
            using var bpeSpec = BpeTokenizer.FromTiktoken(path2, "gpt2", specials);
            var ids = bpeSpec.Encode("A<|endoftext|>B");
            Assert.Contains(258u, ids);
        }
        finally { File.Delete(path2); }
    }

    // -- Bpe encode overloads --------------------------------------------------------

    [Fact]
    public void CoverageBackends_BpeEncodeOverloads()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var viaString = bpe.Encode("Hello, world!");
        var viaSpan = bpe.Encode("Hello, world!"u8.ToArray().AsSpan());
        Assert.Equal(viaString, viaSpan);

        string[] texts = ["Hello, world!", "", "aaaa"];
        var batchStrings = bpe.EncodeBatch(texts);
        Assert.Equal(texts.Length, batchStrings.Length);
        Assert.Equal(viaString, batchStrings[0]);

        var byteDocs = texts.Select(Encoding.UTF8.GetBytes).ToArray();
        var batchBytes = bpe.EncodeBatch(byteDocs);
        Assert.Equal(batchStrings.Length, batchBytes.Length);
        for (var i = 0; i < batchStrings.Length; i++)
            Assert.Equal(batchStrings[i], batchBytes[i]);

        Assert.Throws<ArgumentNullException>(() => bpe.Encode((string)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeBatch((IEnumerable<string>)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeBatch((IEnumerable<byte[]>)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeBatch(new byte[][] { null! }));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeBatch((BytesSource)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeFiles((IFileSource)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeFiles((string)null!));
        Assert.Throws<ArgumentNullException>(() => bpe.EncodeFiles((IEnumerable<string>)null!));
    }

    [Fact]
    public void CoverageBackends_BpeVocabSurface()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        Assert.Equal(50257, bpe.VocabSize);
        var vocab = bpe.Vocab;
        Assert.Equal(50257, vocab.Count);
        Assert.True(vocab.ContainsKey(0u));
        var entries = bpe.VocabEntries;
        Assert.Equal(50257, entries.Count);
        Assert.Equal(50000, bpe.Merges.Count);
        _ = bpe.Encode("Hello");
        Assert.True(bpe.CacheEntries >= 0);
    }

    [Fact]
    public void CoverageBackends_BpeTakeNative_WrapAndDisposed()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        using var tok = new Tokenizer(bpe);
        Assert.Equal(TokenizerKind.Bpe, tok.Kind);
        Assert.Equal([15496u, 11u, 995u, 0u], tok.Encode("Hello, world!"));
        Assert.Throws<ObjectDisposedException>(() => bpe.Encode("hi"));
        Assert.Throws<ObjectDisposedException>(() => bpe.TakeNative());

        using var bpe2 = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var handle = bpe2.TakeNative();
        try
        {
            Assert.False(handle.IsInvalid);
        }
        finally
        {
            handle.Dispose();
        }
        Assert.Throws<ObjectDisposedException>(() => bpe2.Encode("hi"));

        using var bpe3 = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        bpe3.Dispose();
        bpe3.Dispose();
        Assert.Throws<ObjectDisposedException>(() => bpe3.TakeNative());
    }

    [Fact]
    public void CoverageBackends_BpeDisposedGuards()
    {
        var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        bpe.Dispose();
        Assert.Throws<ObjectDisposedException>(() => bpe.Encode("hi"));
        Assert.Throws<ObjectDisposedException>(() => bpe.Encode("hi"u8.ToArray().AsSpan()));
        Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatch(new[] { "hi" }));
        Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatch(new[] { "hi"u8.ToArray() }));
        using var src = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("hi")]);
        Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatch(src));
        var dir = Directory.CreateTempSubdirectory("covbpe_").FullName;
        try
        {
            var f = Path.Combine(dir, "a.txt");
            File.WriteAllText(f, "hi");
            using var fsrc = new TextFileSource([f]);
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeFiles(fsrc));
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeFiles(f));
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeFiles((IEnumerable<string>)[f]));
            var opts = new PadTruncateOptions(PadId: 0);
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatchPadded(new[] { "hi" }, opts));
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatchPadded(new[] { "hi"u8.ToArray() }, opts));
            Assert.Throws<ObjectDisposedException>(() => bpe.EncodeBatchPadded(src, opts));
            Assert.Throws<ObjectDisposedException>(() => bpe.Decode((ReadOnlySpan<uint>)[1u]));
            Assert.Throws<ObjectDisposedException>(() => bpe.Decode((ReadOnlySpan<int>)[1]));
            Assert.Throws<ObjectDisposedException>(() => bpe.Decode((ReadOnlySpan<long>)[1L]));
            Assert.Throws<ObjectDisposedException>(() => { _ = bpe.VocabSize; });
            Assert.Throws<ObjectDisposedException>(() => { _ = bpe.Vocab; });
            Assert.Throws<ObjectDisposedException>(() => { _ = bpe.VocabEntries; });
            Assert.Throws<ObjectDisposedException>(() => { _ = bpe.Merges; });
            Assert.Throws<ObjectDisposedException>(() => { _ = bpe.CacheEntries; });
        }
        finally { Directory.Delete(dir, recursive: true); }
        src.Dispose();
    }

    // -- EncodeBatchPadded guards + prefix/suffix ------------------------------------

    [Fact]
    public void CoverageBackends_PaddedGuards()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var neg = new PadTruncateOptions(PadId: 0, MaxLength: -1);
        Assert.Throws<ArgumentOutOfRangeException>(() => bpe.EncodeBatchPadded(["hi"], neg));
        Assert.Throws<ArgumentOutOfRangeException>(() => bpe.EncodeBatchPadded(["hi"u8.ToArray()], neg));
        var noMaxPad = new PadTruncateOptions(PadId: 0, PadToMaxLength: true);
        Assert.Throws<ArgumentException>(() => bpe.EncodeBatchPadded(["hi"], noMaxPad));
        var noMaxTrunc = new PadTruncateOptions(PadId: 0, Truncate: true);
        Assert.Throws<ArgumentException>(() => bpe.EncodeBatchPadded(["hi"], noMaxTrunc));
        Assert.Throws<ArgumentOutOfRangeException>(() => BackendOps.EncodeBatchPadded(bpe.Native, [Encoding.UTF8.GetBytes("hi")], null, neg));
        Assert.Throws<ArgumentException>(() => BackendOps.EncodeBatchPadded(bpe.Native, [Encoding.UTF8.GetBytes("hi")], null, noMaxPad));
        Assert.Throws<ArgumentException>(() => BackendOps.EncodeBatchPadded(bpe.Native, [Encoding.UTF8.GetBytes("hi")], null, noMaxTrunc));

        using var tok = Fixtures.LoadGpt2();
        Assert.Throws<ArgumentOutOfRangeException>(() => tok.EncodeBatchPadded(["hi"], neg));
    }

    [Fact]
    public void CoverageBackends_PaddedPrefixSuffix()
    {
        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var opts = new PadTruncateOptions(PadId: 0, MaxLength: 10, PadToMaxLength: true, Prefix: [1u], Suffix: [2u]);
        var (matrix, lengths) = bpe.EncodeBatchPadded(["Hi"], opts);
        Assert.Equal(10, matrix.GetLength(1));
        Assert.Equal(1u, matrix[0, 0]);
        var expectedLen = 1L + bpe.Encode("Hi").Length + 1L;
        Assert.Equal(expectedLen, lengths[0]);
        var contentLen = bpe.Encode("Hi").Length;
        Assert.Equal(2u, matrix[0, 1 + contentLen]);

        var optsLeft = new PadTruncateOptions(PadId: 7, MaxLength: 6, PadToMaxLength: true, PadLeft: true, Truncate: true, TruncateLeft: true, Prefix: [9u], Suffix: [8u]);
        var (m2, l2) = bpe.EncodeBatchPadded(["Hello, world! hello hello"], optsLeft);
        Assert.Equal(6, m2.GetLength(1));
        Assert.Equal(6L, l2[0]);

        using var src = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("Hi")]);
        var (m3, _) = bpe.EncodeBatchPadded(src, opts);
        Assert.Equal(1u, m3[0, 0]);
    }

    // -- SentencePiece via FromHf (tiny json) ----------------------------------------

    [Fact]
    public void CoverageBackends_SpFromHfTiny()
    {
        var path = WriteSpJson();
        try
        {
            using var sp = SentencePieceTokenizer.FromHf(path);
            Assert.Equal(TokenizerKind.SentencePiece, sp.Kind);
            Assert.Equal(5, sp.VocabSize);
            Assert.Equal([3u], sp.Encode("Hi"));
            Assert.Equal([3u], sp.Encode("Hi"u8.ToArray().AsSpan()));
            Assert.NotEmpty(sp.EncodeNoNormalize("Hi"));
            Assert.Throws<ArgumentNullException>(() => sp.Encode((string)null!));
            Assert.Throws<ArgumentNullException>(() => sp.EncodeNoNormalize(null!));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CoverageBackends_SpFacadeBothModels()
    {
        using var tok = Tokenizer.FromSentencePiece(SpModelPath);
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Equal(512, tok.VocabSize);
        Assert.Equal([467u, 75u, 468u, 278u, 47u, 299u, 36u], tok.Encode("Hello, world!"));
        Assert.Equal("Hello, world!", tok.DecodeToString(tok.Encode("Hello, world!")));

        using var tok2 = Tokenizer.FromSentencePiece(SpNoPrefixPath);
        Assert.Equal(TokenizerKind.SentencePiece, tok2.Kind);
        Assert.Equal(512, tok2.VocabSize);
        Assert.Equal([75u, 468u, 279u, 47u, 298u, 36u], tok2.Encode("Hello, world!"));

        var bytes = File.ReadAllBytes(SpModelPath);
        using var tok3 = Tokenizer.FromSentencePiece(bytes.AsSpan());
        Assert.Equal(tok.Encode("Hello, world!"), tok3.Encode("Hello, world!"));
    }

    private static SentencePieceTokenizer LoadSpBackend()
    {
        var path = WriteSpJson();
        try
        {
            var sp = SentencePieceTokenizer.FromHf(path);
            return sp;
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CoverageBackends_SpEncodeBatch()
    {
        using var sp = LoadSpBackend();
        var batch = sp.EncodeBatch(new[] { "Hi", "H" });
        Assert.Equal(2, batch.Length);
        Assert.Equal([3u], batch[0]);
        var batchBytes = sp.EncodeBatch(new[] { "Hi"u8.ToArray() });
        Assert.Equal(batch[0], batchBytes[0]);
        using var src = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("Hi")], "|");
        var via = sp.EncodeBatch(src);
        Assert.Single(via);
        Assert.Equal([3u], via[0]);
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatch((IEnumerable<string>)null!));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatch((IEnumerable<byte[]>)null!));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatch(new byte[][] { null! }));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatch((BytesSource)null!));
    }

    [Fact]
    public void CoverageBackends_SpEncodeFiles()
    {
        using var sp = LoadSpBackend();
        var dir = Directory.CreateTempSubdirectory("covsp_").FullName;
        try
        {
            var a = Path.Combine(dir, "a.txt");
            var b = Path.Combine(dir, "b.txt");
            File.WriteAllText(a, "Hi");
            File.WriteAllText(b, "H");
            using var src = new TextFileSource([a, b]);
            var viaSrc = sp.EncodeFiles(src);
            Assert.Equal(2, viaSrc.Length);
            var viaSingle = sp.EncodeFiles(a);
            Assert.Single(viaSingle);
            Assert.Equal(viaSrc[0], viaSingle[0]);
            var viaMulti = sp.EncodeFiles((IEnumerable<string>)[a, b]);
            Assert.Equal(2, viaMulti.Length);
            Assert.Equal(viaSrc[0], viaMulti[0]);
            Assert.Throws<ArgumentNullException>(() => sp.EncodeFiles((IFileSource)null!));
            Assert.Throws<ArgumentNullException>(() => sp.EncodeFiles((string)null!));
            Assert.Throws<ArgumentNullException>(() => sp.EncodeFiles((IEnumerable<string>)null!));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void CoverageBackends_SpPadded()
    {
        using var sp = LoadSpBackend();
        var opts = new PadTruncateOptions(PadId: 0, MaxLength: 5, PadToMaxLength: true, Prefix: [10u], Suffix: [20u]);
        var (m, l) = sp.EncodeBatchPadded(new[] { "Hi" }, opts);
        Assert.Equal(5, m.GetLength(1));
        Assert.Equal(10u, m[0, 0]);
        var (m2, _) = sp.EncodeBatchPadded(new[] { "Hi"u8.ToArray() }, opts);
        Assert.Equal(m[0, 0], m2[0, 0]);
        using var src = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("Hi")]);
        var (m3, l3) = sp.EncodeBatchPadded(src, opts);
        Assert.Equal(m.GetLength(1), m3.GetLength(1));
        Assert.Equal(l, l3);
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatchPadded((IEnumerable<string>)null!, opts));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatchPadded((IEnumerable<byte[]>)null!, opts));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatchPadded(new byte[][] { null! }, opts));
        Assert.Throws<ArgumentNullException>(() => sp.EncodeBatchPadded((BytesSource)null!, opts));
        var neg = new PadTruncateOptions(PadId: 0, MaxLength: -2);
        Assert.Throws<ArgumentOutOfRangeException>(() => sp.EncodeBatchPadded(["Hi"], neg));
        Assert.Throws<ArgumentException>(() => sp.EncodeBatchPadded(["Hi"], new PadTruncateOptions(PadId: 0, PadToMaxLength: true)));
        Assert.Throws<ArgumentException>(() => sp.EncodeBatchPadded(["Hi"], new PadTruncateOptions(PadId: 0, Truncate: true)));
    }

    [Fact]
    public void CoverageBackends_SpDecode()
    {
        using var sp = LoadSpBackend();
        var ids = sp.Encode("Hi");
        var bytes = sp.Decode((ReadOnlySpan<uint>)ids);
        Assert.Equal("Hi", Encoding.UTF8.GetString(bytes));
        var ints = ids.Select(i => (int)i).ToArray();
        Assert.Equal(bytes, sp.Decode((ReadOnlySpan<int>)ints));
        var longs = ids.Select(i => (long)i).ToArray();
        Assert.Equal(bytes, sp.Decode((ReadOnlySpan<long>)longs));
        Assert.Equal("Hi", sp.DecodeToString((ReadOnlySpan<uint>)ids));
        Assert.Equal("Hi", sp.DecodeToString((ReadOnlySpan<int>)ints));
        Assert.Equal("Hi", sp.DecodeToString((ReadOnlySpan<long>)longs));
        Assert.Throws<OverflowException>(() => sp.Decode((ReadOnlySpan<int>)new int[] { -1 }));
        Assert.Throws<OverflowException>(() => sp.Decode((ReadOnlySpan<long>)new long[] { -5L }));
        Assert.Throws<OverflowException>(() => sp.Decode((ReadOnlySpan<long>)new long[] { (long)uint.MaxValue + 1 }));
        Assert.Equal(Array.Empty<byte>(), sp.Decode(ReadOnlySpan<uint>.Empty));
    }

    [Fact]
    public void CoverageBackends_SpVocabSurface()
    {
        using var sp = LoadSpBackend();
        Assert.Equal(5, sp.VocabSize);
        Assert.Equal(5, sp.Vocab.Count);
        Assert.Equal(5, sp.VocabEntries.Count);
        Assert.Single(sp.Merges);
        _ = sp.Encode("Hi");
        Assert.True(sp.CacheEntries >= 0);
        Assert.Equal(TokenizerKind.SentencePiece, sp.Kind);
    }

    [Fact]
    public void CoverageBackends_SpTakeNativeAndDisposed()
    {
        using var sp = LoadSpBackend();
        using var tok = new Tokenizer(sp);
        Assert.Equal(TokenizerKind.SentencePiece, tok.Kind);
        Assert.Throws<ObjectDisposedException>(() => sp.Encode("Hi"));
        Assert.Throws<ObjectDisposedException>(() => sp.TakeNative());

        using var sp2 = LoadSpBackend();
        var h = sp2.TakeNative();
        try { Assert.False(h.IsInvalid); }
        finally { h.Dispose(); }

        var sp3 = LoadSpBackend();
        sp3.Dispose();
        sp3.Dispose();
        Assert.Throws<ObjectDisposedException>(() => sp3.Encode("Hi"));
        Assert.Throws<ObjectDisposedException>(() => sp3.Encode("Hi"u8.ToArray().AsSpan()));
        Assert.Throws<ObjectDisposedException>(() => sp3.EncodeNoNormalize("Hi"));
        Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatch(new[] { "Hi" }));
        Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatch(new[] { "Hi"u8.ToArray() }));
        using var bsrc = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("Hi")]);
        Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatch(bsrc));
        var dir = Directory.CreateTempSubdirectory("covspd_").FullName;
        try
        {
            var f = Path.Combine(dir, "a.txt");
            File.WriteAllText(f, "Hi");
            using var fsrc = new TextFileSource([f]);
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeFiles(fsrc));
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeFiles(f));
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeFiles((IEnumerable<string>)[f]));
            var opts = new PadTruncateOptions(PadId: 0);
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatchPadded(new[] { "Hi" }, opts));
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatchPadded(new[] { "Hi"u8.ToArray() }, opts));
            Assert.Throws<ObjectDisposedException>(() => sp3.EncodeBatchPadded(bsrc, opts));
            Assert.Throws<ObjectDisposedException>(() => sp3.Decode((ReadOnlySpan<uint>)[0u]));
            Assert.Throws<ObjectDisposedException>(() => sp3.Decode((ReadOnlySpan<int>)[0]));
            Assert.Throws<ObjectDisposedException>(() => sp3.Decode((ReadOnlySpan<long>)[0L]));
            Assert.Throws<ObjectDisposedException>(() => { _ = sp3.VocabSize; });
            Assert.Throws<ObjectDisposedException>(() => { _ = sp3.Vocab; });
            Assert.Throws<ObjectDisposedException>(() => { _ = sp3.VocabEntries; });
            Assert.Throws<ObjectDisposedException>(() => { _ = sp3.Merges; });
            Assert.Throws<ObjectDisposedException>(() => { _ = sp3.CacheEntries; });
            Assert.Throws<ObjectDisposedException>(() => sp3.TakeNative());
        }
        finally { Directory.Delete(dir, recursive: true); }
        bsrc.Dispose();
    }

    // -- WrapTruncateOptions ---------------------------------------------------------

    [Fact]
    public void CoverageBackends_WrapNegativeMaxTokens()
    {
        var opts = new WrapTruncateOptions { MaxTokens = -1 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.BuildNative().Dispose());
        using var tok = Fixtures.LoadGpt2();
        var bad = new WrapTruncateOptions { MaxTokens = -5 };
        Assert.Throws<ArgumentOutOfRangeException>(() => tok.EncodeBatchWrapped([Encoding.UTF8.GetBytes("hi")], bad));
    }

    [Fact]
    public void CoverageBackends_WrapDisposeAndSnapshot()
    {
        using var opts = new WrapTruncateOptions { Prefix = [1u], Suffix = [2u] };
        opts.Dispose();
        opts.Dispose();

        using var bpe = BpeTokenizer.FromHf(Fixtures.Gpt2Json);
        var prefix = new uint[] { 11u };
        var suffix = new uint[] { 22u };
        var w = new WrapTruncateOptions { Prefix = prefix, Suffix = suffix };
        using var handle = w.BuildNative();
        prefix[0] = 9999u;
        suffix[0] = 8888u;
        var rows = BackendOps.EncodeBatch(bpe.Native, [Encoding.UTF8.GetBytes("Hi")], false, handle);
        Assert.Single(rows);
        Assert.Equal(11u, rows[0][0]);
        Assert.Equal(22u, rows[0][^1]);

        using var tok = Fixtures.LoadGpt2();
        var p2 = new uint[] { 5u };
        var o2 = new WrapTruncateOptions { Prefix = p2, Suffix = [6u], MaxTokens = 0 };
        var r2 = tok.EncodeBatchWrapped([Encoding.UTF8.GetBytes("Hi there")], o2);
        Assert.Equal([5u, 6u], r2[0]);
        using var empty = new WrapTruncateOptions();
        using var h2 = empty.BuildNative();
        Assert.False(h2.IsInvalid);
    }

    // -- GigatokenOptions ------------------------------------------------------------

    [Fact]
    public async Task CoverageBackends_DefaultParallelGetSet()
    {
        var orig = GigatokenOptions.DefaultParallel;
        try
        {
            Assert.Equal(orig, GigatokenOptions.DefaultParallel);
            GigatokenOptions.DefaultParallel = true;
            Assert.Equal(true, GigatokenOptions.DefaultParallel);
            GigatokenOptions.DefaultParallel = false;
            Assert.Equal(false, GigatokenOptions.DefaultParallel);
            GigatokenOptions.DefaultParallel = null;
            Assert.Null(GigatokenOptions.DefaultParallel);

            GigatokenOptions.DefaultParallel = false;
            using var tok = Fixtures.LoadGpt2();
            var serial = tok.EncodeBatch(["a", "b"]);
            GigatokenOptions.DefaultParallel = true;
            var parallel = tok.EncodeBatch(["a", "b"]);
            Assert.Equal(serial[0], parallel[0]);

            var tasks = new System.Threading.Tasks.Task[8];
            for (var i = 0; i < tasks.Length; i++)
            {
                var v = i % 3 == 0 ? (bool?)null : (i % 2 == 0);
                tasks[i] = System.Threading.Tasks.Task.Run(() =>
                {
                    GigatokenOptions.DefaultParallel = v;
                    _ = GigatokenOptions.DefaultParallel;
                    using var t = Fixtures.LoadGpt2();
                    _ = t.Encode("hi");
                });
            }
            await System.Threading.Tasks.Task.WhenAll(tasks);
        }
        finally
        {
            GigatokenOptions.DefaultParallel = orig;
        }
        Assert.Equal(orig, GigatokenOptions.DefaultParallel);
    }

    // -- Sources ---------------------------------------------------------------------

    [Fact]
    public void CoverageBackends_BytesSourceSingleBuffer()
    {
        using var s1 = new BytesSource((ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("hello"), (byte[]?)null);
        using var tok = Fixtures.LoadGpt2();
        var rows = tok.EncodeBatch(s1);
        Assert.Single(rows);
        Assert.Equal(tok.Encode("hello"), rows[0]);

        using var s2 = new BytesSource((ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("a<|sep|>b"), Encoding.UTF8.GetBytes("<|sep|>"));
        var rows2 = tok.EncodeBatch(s2);
        Assert.Equal(2, rows2.Length);

        using var s3 = new BytesSource(new[] { (ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("hi") }, "|");
        Assert.Single(tok.EncodeBatch(s3));
        s1.Dispose();
        s2.Dispose();
        s3.Dispose();
    }

    [Fact]
    public void CoverageBackends_SourcesNullEmptyDisposeSnapshot()
    {
        Assert.Throws<ArgumentNullException>(() => new BytesSource((ReadOnlyMemory<byte>[])null!));
        Assert.Throws<ArgumentNullException>(() => new BytesSource((byte[][])null!));
        Assert.Throws<ArgumentNullException>(() => new BytesSource(new byte[][] { null! }));
        Assert.Throws<ArgumentNullException>(() => new BytesSource((ReadOnlyMemory<byte>[])null!, (string?)null));

        using var tok = Fixtures.LoadGpt2();
        using var emptyStr = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("ab")], "");
        Assert.NotNull(tok.EncodeBatch(emptyStr));
        using var emptyBytes = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("ab")], new byte[0]);
        Assert.NotNull(tok.EncodeBatch(emptyBytes));
        using var nullSep = new BytesSource([(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("ab")], (string?)null);
        Assert.Single(tok.EncodeBatch(nullSep));

        var sep = Encoding.UTF8.GetBytes("<|sep|>");
        var blob = Encoding.UTF8.GetBytes("a<|sep|>b<|sep|>");
        using var src = new BytesSource([(ReadOnlyMemory<byte>)blob], (byte[])sep.Clone());
        sep[0] = 0xFF;
        var rows = tok.EncodeBatch(src);
        Assert.Equal(2, rows.Length);
        Assert.Equal(tok.Encode("a"), rows[0]);

        var dir = Directory.CreateTempSubdirectory("covsrc_").FullName;
        try
        {
            var f = Path.Combine(dir, "a.txt");
            File.WriteAllText(f, "hello");
            var tfs = new TextFileSource([f]);
            tfs.Dispose();
            tfs.Dispose();
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles(tfs));
            using var tfs2 = new TextFileSource([f], "|");
            tfs2.Dispose();
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles(tfs2));
            using var js = new JsonlFileSource([f]);
            js.Dispose();
            Assert.Throws<ObjectDisposedException>(() => tok.EncodeFiles(js));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // -- SubstringMatcher --------------------------------------------------------------

    [Fact]
    public void CoverageBackends_SubstringMatcherGuards()
    {
        Assert.Throws<ArgumentNullException>(() => new SubstringMatcher(null!));
        Assert.Throws<ArgumentNullException>(() => new SubstringMatcher(new List<string> { "a", null! }));
        Assert.Throws<ArgumentNullException>(() => new SubstringMatcher(new string[] { null! }));

        using var empty = new SubstringMatcher([]);
        Assert.Equal(0, empty.Count);
        Assert.Empty(empty.Present("anything"));

        using var dup = new SubstringMatcher(["foo", "foo"]);
        Assert.Equal(2, dup.Count);
        Assert.Equal([0, 1], dup.Present("x foo y"));

        using var m = new SubstringMatcher(["<|endoftext|>", "end", "foo"]);
        Assert.Equal(3, m.Count);
        Assert.Equal([0, 1], m.Present("x <|endoftext|> y"));
        Assert.Throws<ArgumentNullException>(() => m.Present(null!));
        m.Dispose();
        m.Dispose();
        Assert.Throws<ObjectDisposedException>(() => m.Present("hi"));
    }

    // -- Exceptions --------------------------------------------------------------------

    [Fact]
    public void CoverageBackends_HubFileNotFoundCtors()
    {
        var e1 = new HubFileNotFoundException("missing file");
        Assert.Equal("missing file", e1.Message);
        Assert.IsAssignableFrom<FileNotFoundException>(e1);

        var inner = new InvalidOperationException("cause");
        var e2 = new HubFileNotFoundException("missing with cause", inner);
        Assert.Equal("missing with cause", e2.Message);
        Assert.Same(inner, e2.InnerException);

        var e3 = new HubFileNotFoundException("missing named", "tokenizer.json", inner);
        Assert.Equal("missing named", e3.Message);
        Assert.Equal("tokenizer.json", e3.FileName);
        Assert.Same(inner, e3.InnerException);

        var e4 = new HubFileNotFoundException("no file", null, null);
        Assert.Equal("no file", e4.Message);
        Assert.Null(e4.FileName);
    }
}
