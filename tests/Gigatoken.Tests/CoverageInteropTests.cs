using System.Diagnostics;
using System.Reflection;
using System.Text;
using Gigatoken.Interop;

namespace Gigatoken.Tests;

// Part A (in-process) + Part B (CLI subprocess) coverage tests for
// src/Gigatoken/Interop/{NativeBridge,NativeLib,SafeHandles}.cs and tools/Gigatoken.Cli/Program.cs.
// Tests-only file: no source file is modified.

public sealed class CoverageInteropSafeHandles
{
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void TokenizerHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { TokenizerHandle.FromRaw(new nint(raw)); });
        Assert.Contains("tokenizer handle", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void SourceHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { SourceHandle.FromRaw(new nint(raw)); });
        Assert.Contains("source handle", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void MatcherHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { MatcherHandle.FromRaw(new nint(raw)); });
        Assert.Contains("matcher handle", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void WrapOptsHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { WrapOptsHandle.FromRaw(new nint(raw)); });
        Assert.Contains("wrap-opts handle", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void PadOptsHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { PadOptsHandle.FromRaw(new nint(raw)); });
        Assert.Contains("pad-opts handle", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void PretokenIterHandle_FromRawInvalid_Throws(int raw)
    {
        var ex = Assert.Throws<GigatokenException>(() => { PretokenIterHandle.FromRaw(new nint(raw)); });
        Assert.Contains("pretoken-iterator handle", ex.Message);
    }

    [Fact]
    public void FreshHandles_AreInvalid()
    {
        using var t = new TokenizerHandle();
        using var s = new SourceHandle();
        using var m = new MatcherHandle();
        using var w = new WrapOptsHandle();
        using var p = new PadOptsHandle();
        using var i = new PretokenIterHandle();
        Assert.True(t.IsInvalid);
        Assert.True(s.IsInvalid);
        Assert.True(m.IsInvalid);
        Assert.True(w.IsInvalid);
        Assert.True(p.IsInvalid);
        Assert.True(i.IsInvalid);
    }
}

public sealed class CoverageInteropBridge
{
    private sealed class ThrowingBufferList : IReadOnlyList<byte[]>
    {
        public byte[] this[int index] => throw new InvalidOperationException("boom");
        public int Count => 1;
        public IEnumerator<byte[]> GetEnumerator() => throw new InvalidOperationException("boom");
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    [Fact]
    public void PinnedBuffers_CtorFailure_DisposesAndRethrows()
    {
        Assert.Throws<InvalidOperationException>(() => { new PinnedBuffers(new ThrowingBufferList()); });
    }

    [Fact]
    public void Check_InvalidArg_MapsToArgumentException()
    {
        Assert.Throws<ArgumentException>(() => { NativeBridge.Check(1); });
    }

    [Fact]
    public void Check_Unauthorized_MapsToUnauthorizedAccessException()
    {
        Assert.Throws<UnauthorizedAccessException>(() => { NativeBridge.Check(4); });
    }

    [Fact]
    public void Check_UnknownCode_MapsToGigatokenException()
    {
        var ex = Assert.Throws<GigatokenException>(() => { NativeBridge.Check(42); });
        Assert.Contains("native error 42", ex.Message);
    }

    [Fact]
    public void BpeFromTiktoken_SpecialLengthMismatch_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
        {
            NativeBridge.BpeFromTiktoken("p"u8.ToArray(), "gpt2"u8.ToArray(), ["k"u8.ToArray()], []);
        });
        Assert.Contains("length mismatch", ex.Message);
    }

    [Fact]
    public void SpFromHf_LoadsSentencePieceFile()
    {
        const string tinySpJson = """{"model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"H":1,"i":2,"Hi":3,"<0x20>":4},"merges":["H i"]}}""";
        var dir = Directory.CreateTempSubdirectory("gt_covsp_").FullName;
        try
        {
            var path = Path.Combine(dir, "tokenizer.json");
            File.WriteAllText(path, tinySpJson);
            using var sp = SentencePieceTokenizer.FromHf(path);
            Assert.Equal(TokenizerKind.SentencePiece, sp.Kind);
            Assert.Equal([3u], sp.Encode("Hi"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Unflatten_CountsExceedFlat_Throws()
    {
        var ex = Assert.Throws<GigatokenException>(() => { NativeBridge.Unflatten([1u, 2u, 3u], [2L, 2L]); });
        Assert.Contains("exceed", ex.Message);
    }

    [Fact]
    public void Unflatten_CountsMismatch_Throws()
    {
        var ex = Assert.Throws<GigatokenException>(() => { NativeBridge.Unflatten([1u, 2u], [1L]); });
        Assert.Contains("do not sum", ex.Message);
    }

    [Fact]
    public void Unflatten_HugeCount_ThrowsOverflow()
    {
        Assert.Throws<OverflowException>(() => { NativeBridge.Unflatten([1u], [(long)int.MaxValue + 1]); });
    }

    [Fact]
    public void Unflatten_Valid_SplitsRows()
    {
        var rows = NativeBridge.Unflatten([1u, 2u, 3u], [2L, 1L]);
        Assert.Equal(2, rows.Length);
        Assert.Equal([1u, 2u], rows[0]);
        Assert.Equal([3u], rows[1]);
    }

    [Fact]
    public void NativeBytesSource_NullAndNonNullSeparator()
    {
        using var noSep = NativeBridge.BytesSource(["hello "u8.ToArray(), "world"u8.ToArray()], null);
        Assert.False(noSep.IsInvalid);
        using var withSep = NativeBridge.BytesSource(["a|b|c"u8.ToArray()], "|"u8.ToArray());
        Assert.False(withSep.IsInvalid);
    }

    private static byte[] Corpus() =>
        Encoding.UTF8.GetBytes(
            string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 20)));

    [Fact]
    public void TrainBpe_WithSeparator_BytesCorpus()
    {
        var (vocab, merges) = BpeTrainer.Train(Corpus(), 270, [], separator: Encoding.UTF8.GetBytes(" "));
        Assert.Equal(270, vocab.Count);
        Assert.Equal(270 - 256, merges.Count);
    }

    [Fact]
    public void TrainBpe_WithSeparator_FileSource()
    {
        var path = Path.Combine(Path.GetTempPath(), $"covtrain_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 20)));
        try
        {
            using var src = new TextFileSource([path]);
            var (vocab, merges) = BpeTrainer.Train(src, 270, [], separator: Encoding.UTF8.GetBytes(" "));
            Assert.Equal(270, vocab.Count);
            Assert.Equal(270 - 256, merges.Count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void PretokenizedCounts_WithSeparator()
    {
        var text = "hello hello world foo"u8.ToArray();
        var counts = Pretokenizer.CountPretokens(text, Encoding.UTF8.GetBytes(" "));
        Assert.NotEmpty(counts);
        Assert.All(counts, kv => Assert.True(kv.Count > 0));
        var pieces = Pretokenizer.Pretokenize(text).ToList();
        Assert.Equal(pieces.Count, counts.Sum(kv => kv.Count));
    }
}

public sealed class CoverageInteropNativeLib
{
    private static MethodInfo ResolveMethod()
    {
        var mi = typeof(NativeLib).GetMethod("Resolve",
            BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(mi);
        return mi;
    }

    private static object? InvokeResolve(MethodInfo mi, string libraryName) =>
        mi.Invoke(null, new object?[]
        {
            libraryName, typeof(NativeLib).Assembly, null,
        });

    [Fact]
    public void Resolve_WrongLibraryName_ReturnsZero()
    {
        var result = (nint)InvokeResolve(ResolveMethod(), "definitely-not-the-library")!;
        Assert.Equal(nint.Zero, result);
    }

    // NOTE: NativeLib.Resolve's env-override (GIGATOKEN_NATIVE), bad-env, and
    // bundled-candidate arms are deliberately NOT tested in-process. The
    // DllImportResolver fires lazily per P/Invoke method, so mutating the
    // process env or dropping a lib copy next to the test dll lets concurrently
    // binding methods resolve to a *different* native copy with independent
    // thread-local error state (proven with /proc/self/maps: two lib objects
    // mapped at once). That split intermittently corrupts native error text
    // suite-wide (right codes, empty/stale messages). See final report.

    [Fact]
    public void Resolve_MapName_IsLinuxSo()
    {
        var mi = typeof(NativeLib).GetMethod("MapName",
            BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(mi);
        Assert.Equal("libgigatoken_cabi.so", (string)mi.Invoke(null, null)!);
    }

    [Fact]
    public void EnsureLoaded_DoubleRegister_ResetsFlagAndRethrows()
    {
        var field = typeof(NativeLib).GetField("_loaded",
            BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(field);
        var ensure = typeof(NativeLib).GetMethod("EnsureLoaded",
            BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
        Assert.NotNull(ensure);
        try
        {
            // Resolver is already registered (native calls work), so forcing a
            // re-register must throw and reset the loaded flag via the catch.
            field.SetValue(null, 0);
            var tie = Assert.Throws<TargetInvocationException>(() => ensure.Invoke(null, null));
            Assert.IsType<InvalidOperationException>(tie.InnerException);
            Assert.Equal(0, (int)field.GetValue(null)!);
        }
        finally
        {
            field.SetValue(null, 1);
        }
    }
}

public sealed class CoverageInteropCli : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("gt_covcli_").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private static string FindCliDll()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            foreach (var cfg in new[] { "Debug", "Release" })
            {
                var p = Path.Combine(dir.FullName, "tools", "Gigatoken.Cli",
                    "bin", cfg, "net10.0", "Gigatoken.Cli.dll");
                if (File.Exists(p)) return p;
            }
        }
        return "";
    }

    private static (int Exit, string Out, string Err) RunCli(string dll, params string[] args)
    {
        using var proc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "dotnet",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };
        proc.StartInfo.ArgumentList.Add(dll);
        foreach (var a in args) proc.StartInfo.ArgumentList.Add(a);
        proc.Start();
        var stdout = proc.StandardOutput.ReadToEnd();
        var stderr = proc.StandardError.ReadToEnd();
        if (!proc.WaitForExit(180_000))
        {
            try { proc.Kill(); } catch { }
            throw new TimeoutException("CLI subprocess timed out.");
        }
        return (proc.ExitCode, stdout, stderr);
    }

    private string WriteDoc(string name, string? content = null)
    {
        var p = Path.Combine(_dir, name);
        File.WriteAllText(p, content ??
            string.Concat(Enumerable.Repeat("Hello, world! This is a bench fixture line.\n", 40)));
        return p;
    }

    private string WriteTiktoken(string name, params (string Text, uint Rank)[] extras)
    {
        var lines = new List<string>(300);
        for (var b = 0; b < 256; b++)
            lines.Add($"{Convert.ToBase64String([(byte)b])} {b}");
        foreach (var (text, rank) in extras)
            lines.Add($"{Convert.ToBase64String(Encoding.UTF8.GetBytes(text))} {rank}");
        var p = Path.Combine(_dir, name);
        File.WriteAllLines(p, lines);
        return p;
    }

    private static string SpModelPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny_sp.model");

    [Fact]
    public void Help_TopLevel_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        foreach (var args in new string[][] { [], ["-h"], ["--help"] })
        {
            var (exit, stdout, _) = RunCli(dll, args);
            Assert.Equal(0, exit);
            Assert.Contains("Usage:", stdout);
        }
    }

    [Fact]
    public void Help_Bench_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var (exit, stdout, _) = RunCli(dll, "bench", "--help");
        Assert.Equal(0, exit);
        Assert.Contains("Usage:", stdout);
    }

    [Fact]
    public void UnknownCommand_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var (exit, _, stderr) = RunCli(dll, "frobnicate");
        Assert.Equal(2, exit);
        Assert.Contains("Unknown command", stderr);
    }

    [Fact]
    public void UnknownOption_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc, "--bogus-opt");
        Assert.Equal(2, exit);
        Assert.Contains("Unknown option", stderr);
    }

    [Fact]
    public void MissingTokenizerAndFiles_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var (exit1, _, stderr1) = RunCli(dll, "bench");
        Assert.Equal(2, exit1);
        Assert.Contains("requires <tokenizer>", stderr1);
        var (exit2, _, stderr2) = RunCli(dll, "bench", Fixtures.Gpt2Json);
        Assert.Equal(2, exit2);
        Assert.Contains("requires <tokenizer>", stderr2);
    }

    [Fact]
    public void NonexistentFile_Exit1()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var missing = Path.Combine(_dir, "does_not_exist.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, missing);
        Assert.Equal(1, exit);
        Assert.Contains("File not found", stderr);
    }

    [Fact]
    public void NonexistentTokenizer_Exit1()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var missing = Path.Combine(_dir, "no_such_tokenizer.json");
        var (exit, _, stderr) = RunCli(dll, "bench", missing, doc);
        Assert.Equal(1, exit);
        Assert.Contains("error:", stderr);
    }

    [Fact]
    public void ValidateWithoutCompare_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", "--validate", Fixtures.Gpt2Json, doc);
        Assert.Equal(2, exit);
        Assert.Contains("--validate requires --compare-to", stderr);
    }

    [Fact]
    public void InvalidComparisonLimit_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
            "--comparison-limit", "bogus!");
        Assert.Equal(2, exit);
        Assert.Contains("invalid --comparison-limit", stderr);
    }

    [Fact]
    public void MissingOptionValue_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, "--compare-to");
        Assert.Equal(2, exit);
        Assert.Contains("requires a value", stderr);
    }

    [Fact]
    public void EmptyDocSeparator_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
            "--doc-separator", "");
        Assert.Equal(2, exit);
        Assert.Contains("must not be empty", stderr);
    }

    [Fact]
    public void BenchSuccess_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void BenchStreamFromDisk_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", "--stream-from-disk", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void BenchStreamFromDiskWithSeparator_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt", "aaa|bbb|ccc|ddd\n");
        var (exit, stdout, _) = RunCli(dll, "bench", "--stream-from-disk",
            Fixtures.Gpt2Json, doc, "--doc-separator", "|");
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void BenchInMemoryWithSeparator_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt", "aaa|bbb|ccc|ddd\n");
        var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
            "--doc-separator", "|");
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void CompareToSelf_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("ref", stdout);
        Assert.Contains("than ref", stdout);
    }

    [Fact]
    public void ValidatePass_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            "--validate", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("validation OK", stdout);
    }

    [Fact]
    public void ValidateFail_Exit1()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var a = WriteTiktoken("a.tiktoken", ("AB", 256), ("ABC", 257));
        var b = WriteTiktoken("b.tiktoken", ("AB", 256), ("ABD", 257));
        var doc = WriteDoc("doc.txt", "ABC ABD AB ABC ABD\n");
        var (exit, _, stderr) = RunCli(dll, "bench", a, doc,
            "--compare-to", b, "--pretokenizer", "gpt2", "--validate");
        Assert.Equal(1, exit);
        Assert.Contains("validation FAILED", stderr);
    }

    [Fact]
    public void ComparisonLimitTooSmall_Exit1()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            "--comparison-limit", "0", Fixtures.Gpt2Json, doc);
        Assert.Equal(1, exit);
        Assert.Contains("nothing to compare", stderr);
    }

    [Fact]
    public void ComparisonLimitNone_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            "--comparison-limit", "none", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("than ref", stdout);
    }

    [Fact]
    public void ValidatePassTruncated_Exit0()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            "--validate", "--comparison-limit", "10", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("validation OK", stdout);
    }

    [Fact]
    public void CompareToUnloadable_Exit1()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var missing = Path.Combine(_dir, "no_such_ref.json");
        var (exit, _, stderr) = RunCli(dll, "bench", "--compare-to", missing,
            Fixtures.Gpt2Json, doc);
        Assert.Equal(1, exit);
        Assert.Contains("error:", stderr);
    }

    [Fact]
    public void TiktokenWithoutPretokenizer_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var tik = WriteTiktoken("tiny.tiktoken", ("AB", 256), ("ABC", 257));
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", tik, doc);
        Assert.Equal(2, exit);
        Assert.Contains(".tiktoken input requires --pretokenizer", stderr);
    }

    [Fact]
    public void TiktokenWithPretokenizer_Success()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var tik = WriteTiktoken("tiny.tiktoken", ("AB", 256), ("ABC", 257));
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", tik, doc, "--pretokenizer", "gpt2");
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void PretokenizerWithNonTiktoken_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
            "--pretokenizer", "gpt2");
        Assert.Equal(2, exit);
        Assert.Contains("--pretokenizer only applies to .tiktoken inputs", stderr);
    }

    [Fact]
    public void PretokenizerWithModel_Exit2()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, _, stderr) = RunCli(dll, "bench", SpModelPath, doc,
            "--pretokenizer", "gpt2");
        Assert.Equal(2, exit);
        Assert.Contains("--pretokenizer only applies to .tiktoken inputs", stderr);
    }

    [Fact]
    public void ModelInput_Success()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", SpModelPath, doc);
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void ComparisonLimitSuffixes_Parse()
    {
        // ParseSize k/g/t/unlimited/huge-clamp arms (Program.cs ParseSize).
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        foreach (var limit in new[] { "1k", "2g", "1t", "unlimited", "99999999999999999999t" })
        {
            var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
                "--compare-to", Fixtures.Gpt2Json, "--comparison-limit", limit);
            Assert.Equal(0, exit);
            Assert.Contains("gigatoken is", stdout);
        }
    }

    [Fact]
    public void TruncationGuard_ValidateTinyLimit()
    {
        // 1-byte subset: last-doc truncation guard + compareToLength<=0 skip
        // (Program.cs validate loop), guard-note suffix on success.
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = WriteDoc("doc.txt");
        var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc,
            "--compare-to", Fixtures.Gpt2Json, "--comparison-limit", "1", "--validate");
        Assert.Equal(0, exit);
        Assert.Contains("validation OK", stdout);
        Assert.Contains("ignored", stdout);
    }

    [Fact]
    public void LeadingInvalidUtf8_NothingToCompare()
    {
        // First doc starts with undecodable bytes: DecodeIgnoreEncode drops
        // them all -> empty subset -> exit 1 (SubsetDocs empty arm).
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var p = Path.Combine(_dir, "invalid.txt");
        var bytes = new byte[] { 0xFF, 0xFE, (byte)'H', (byte)'i' };
        File.WriteAllBytes(p, bytes);
        var (exit, _, stderr) = RunCli(dll, "bench", Fixtures.Gpt2Json, p,
            "--compare-to", Fixtures.Gpt2Json, "--comparison-limit", "2");
        Assert.Equal(1, exit);
        Assert.Contains("nothing to compare", stderr);
    }

    [Fact]
    public void SeparatorEdges_SplitOne()
    {
        // Docs starting/ending with the separator: SplitOne idx==0 skip and
        // trailing-empty arms (Program.cs SplitOne).
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var p = Path.Combine(_dir, "seps.txt");
        File.WriteAllText(p, "\nHello, world!\nSecond doc here.\n");
        var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, p,
            "--compare-to", Fixtures.Gpt2Json, "--doc-separator", "\n");
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }
}
