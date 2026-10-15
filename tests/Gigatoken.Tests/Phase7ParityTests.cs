using System.Text.Json;
using Xunit.Abstractions;

namespace Gigatoken.Tests;

/// <summary>
/// Phase 7 full parity sweep: every doc of the OWT 100 MB subset and the DCLM
/// hostile sample must encode identically to the oracle goldens (tiktoken
/// r50k_base for GPT-2, HF `tokenizers` lib for Qwen2.5 and TinyLlama).
/// Data lives outside the repo (`GIGATOKEN_PHASE7_DIR`, default
/// `~/data/gigatoken_cs_phase7`, built by `tools/export_phase7_goldens.py`);
/// all tests skip when it is absent. Throughput is reported for CJK/perf
/// regression tracking (upstream: CJK-heavy data is cache-hostile and slower).
/// </summary>
public sealed class Phase7ParityTests
{
    private readonly ITestOutputHelper _out;

    public Phase7ParityTests(ITestOutputHelper output) => _out = output;

    private static string Phase7Dir =>
        Environment.GetEnvironmentVariable("GIGATOKEN_PHASE7_DIR")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "data", "gigatoken_cs_phase7");

    private static string[] ReadDocs(string path)
    {
        using var br = new BinaryReader(File.OpenRead(path));
        var n = (int)br.ReadUInt64();
        var docs = new string[n];
        for (var i = 0; i < n; i++)
        {
            var len = (int)br.ReadUInt64();
            docs[i] = System.Text.Encoding.UTF8.GetString(br.ReadBytes(len));
        }
        return docs;
    }

    private static uint[][] ReadIds(string path)
    {
        using var br = new BinaryReader(File.OpenRead(path));
        var n = (int)br.ReadUInt64();
        var rows = new uint[n][];
        for (var i = 0; i < n; i++)
        {
            var len = (int)br.ReadUInt64();
            rows[i] = new uint[len];
            for (var j = 0; j < len; j++)
                rows[i][j] = br.ReadUInt32();
        }
        return rows;
    }

    private void Sweep(string corpus, string backend)
    {
        var dir = Phase7Dir;
        var docsPath = Path.Combine(dir, $"{corpus}_docs.bin");
        var idsPath = Path.Combine(dir, $"ids.{corpus}.{backend}.bin");
        var manifestPath = Path.Combine(dir, "manifest.json");
        if (!(File.Exists(docsPath) && File.Exists(idsPath) && File.Exists(manifestPath)))
        {
            _out.WriteLine($"SKIP: phase7 data absent (run tools/export_phase7_goldens.py into {dir})");
            return;
        }

        using var manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var tokPaths = manifest.RootElement.GetProperty("tokenizers");
        using var tok = backend switch
        {
            "gpt2" => Fixtures.LoadGpt2(),
            "qwen2_5" => Tokenizer.FromPretrained(tokPaths.GetProperty("qwen2_5").GetProperty("path").GetString()!),
            "tinyllama" => Tokenizer.FromPretrained(tokPaths.GetProperty("tinyllama").GetProperty("path").GetString()!),
            _ => throw new InvalidOperationException(backend),
        };

        var docs = ReadDocs(docsPath);
        var want = ReadIds(idsPath);
        Assert.Equal(want.Length, docs.Length);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var got = tok.EncodeBatch(docs, parallel: true);
        sw.Stop();

        Assert.Equal(want.Length, got.Length);
        long tokens = 0;
        for (var i = 0; i < want.Length; i++)
            tokens += got[i].Length;
        // Doc-ID-level diffing (not a bare mismatch count): collect every
        // mismatching doc id with its first-diff token position and the
        // divergent ID windows, and fail with the first few plus the total.
        // Corpus metadata carries no known-mismatch allowance (upstream's
        // hostile suite likewise asserts exact parity), so expected is 0.
        var bad = new List<string>();
        for (var i = 0; i < want.Length; i++)
        {
            if (want[i].AsSpan().SequenceEqual(got[i]))
                continue;
            var w = want[i];
            var g = got[i];
            var common = Math.Min(w.Length, g.Length);
            var j = 0;
            while (j < common && w[j] == g[j]) j++;
            string W(int from, int n) => string.Join(",", w.Skip(from).Take(n));
            string G(int from, int n) => string.Join(",", g.Skip(from).Take(n));
            bad.Add($"doc {i}: len want={w.Length} got={g.Length}, " +
                (j < common
                    ? $"first diff at token {j} (want [{W(j, 5)}] vs got [{G(j, 5)}])"
                    : "common prefix identical, lengths differ"));
            if (bad.Count >= 5) break;
        }
        Assert.True(bad.Count == 0,
            $"{bad.Count}/{want.Length} {corpus}/{backend} documents differ (expected 0):\n" +
            string.Join("\n", bad));
        var mb = docs.Sum(d => System.Text.Encoding.UTF8.GetByteCount(d)) / 1e6;
        _out.WriteLine($"{corpus}/{backend}: {docs.Length} docs, {mb:F1} MB, " +
            $"{tokens / 1e6:F2}M tokens in {sw.Elapsed.TotalSeconds:F1}s " +
            $"({mb / sw.Elapsed.TotalSeconds:F1} MB/s)");
    }

    [Fact] public void OwtGpt2() => Sweep("owt", "gpt2");
    [Fact] public void OwtQwen() => Sweep("owt", "qwen2_5");
    [Fact] public void OwtTinyllama() => Sweep("owt", "tinyllama");
    [Fact] public void DclmGpt2() => Sweep("dclm", "gpt2");
    [Fact] public void DclmQwen() => Sweep("dclm", "qwen2_5");
    [Fact] public void DclmTinyllama() => Sweep("dclm", "tinyllama");
}
