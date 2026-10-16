// Gigatoken.Benchmarks — BenchmarkDotNet harness (phase 6).
// Usage: dotnet run -c Release --project benches/Gigatoken.Benchmarks -- <tokenizer> [corpusFile]
//   <tokenizer>: HF repo-id, directory, tokenizer.json path, .tiktoken, or .model file.
//   Defaults to the test GPT-2 fixture when omitted.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Gigatoken;

var summary = BenchmarkSwitcher.FromAssembly(typeof(BenchConfig).Assembly).Run(args);
return 0;

public sealed class BenchConfig : ManualConfig
{
    public static string? TokenizerSpec { get; }
    public static string? CorpusFile { get; }

    static BenchConfig()
    {
        var args = Environment.GetCommandLineArgs();
        // dotnet run ... -- <tokenizer> [corpusFile]; BenchmarkDotNet consumes its own flags first.
        var rest = args.SkipWhile(a => a != "--").Skip(1).ToArray();
        TokenizerSpec = rest.Length > 0 ? rest[0] : null;
        CorpusFile = rest.Length > 1 ? rest[1] : null;
    }
}

[Config(typeof(BenchConfig))]
[MemoryDiagnoser]
public class EncodeBenchmarks
{
    private Tokenizer _tok = null!;
    private string[] _docs = null!;
    private BytesSource _bytes = null!;

    [GlobalSetup]
    public void Setup()
    {
        var spec = BenchConfig.TokenizerSpec ??
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                "tests", "Gigatoken.Tests", "Fixtures", "gpt2_tokenizer.json");
        _tok = Tokenizer.FromPretrained(spec);
        var corpus = BenchConfig.CorpusFile;
        string text = corpus is null
            ? string.Concat(System.Linq.Enumerable.Repeat(
                "Tokenize your text data at GB/s! Mixed scripts: 日本語テスト, Français, العربية. ", 200))
            : File.ReadAllText(corpus);
        _docs = Enumerable.Repeat(text, 16).ToArray();
        _bytes = new BytesSource(_docs.Select(System.Text.Encoding.UTF8.GetBytes).ToArray());
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _bytes.Dispose();
        _tok.Dispose();
    }

    [Benchmark] public uint[] Single() => _tok.Encode(_docs[0]);
    [Benchmark] public uint[][] BatchSerial() => _tok.EncodeBatch(_docs, parallel: false);
    [Benchmark] public uint[][] BatchParallel() => _tok.EncodeBatch(_docs, parallel: true);
    [Benchmark] public uint[][] BatchBytes() => _tok.EncodeBatch(_bytes, parallel: true);

    [Benchmark]
    public (uint[,] Matrix, long[] Lengths) BatchPadded() =>
        // Truncate must accompany PadToMaxLength: without it the native side
        // rejects any row longer than MaxLength, and the synthetic docs always
        // exceed 512 ids.
        _tok.EncodeBatchPadded(_docs, new PadTruncateOptions(PadId: 0, PadToMaxLength: true, MaxLength: 512, Truncate: true), parallel: true);
}
