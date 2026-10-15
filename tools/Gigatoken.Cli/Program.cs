// Gigatoken.Cli — `gigatoken bench` port (mirrors gigatoken/_cli.py).
// Hand-rolled argument parsing (no external CLI package).

using System.Buffers;
using System.Diagnostics;
using System.Text;
using Gigatoken;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    PrintUsage();
    return 0;
}
if (args[0] != "bench")
{
    Console.Error.WriteLine($"Unknown command '{args[0]}'. Only 'bench' is supported.");
    PrintUsage();
    return 2;
}
return Bench(args[1..]);

static void PrintUsage()
{
    Console.WriteLine("""
        Usage: gigatoken bench <tokenizer> <files...> [options]

          <tokenizer>            HF repo-id, directory, tokenizer.json path, .tiktoken file, or .model file.
          <files...>             Input text files to encode.

        Options:
          --stream-from-disk     Encode via file sources in Rust (no in-memory copy).
          --compare-to <tok>     Second tokenizer spec for parity + speedup comparison
                                 (port adaptation: upstream only supports 'hf'; compares
                                 two Gigatoken tokenizers instead — no live HF path).
          --comparison-limit <n> Max bytes compared (default 100MB; decimal SI suffixes
                                 k/m/g/t with optional 'i'/'b', e.g. 100MB; 'none' = all).
          --validate             Fail if any compared doc diverges (requires --compare-to).
          --doc-separator <sep>  Document separator for in-memory/whole-file mode.
          --pretokenizer <name>  Pretokenizer scheme for .tiktoken inputs (gpt2, gpt4, ...).
        """);
}

static int Bench(string[] args)
{
    try
    {
        return BenchInner(args);
    }
    catch (CliUsageException ex)
    {
        Console.Error.WriteLine(ex.Message);
        return 2;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"error: {ex.Message}");
        return 1;
    }
}

static int BenchInner(string[] args)
{
    string? tokenizerSpec = null;
    var files = new List<string>();
    bool streamFromDisk = false;
    string? compareTo = null;
    string comparisonLimitRaw = "100MB";
    bool validate = false;
    string? docSeparator = null;
    string? pretokenizer = null;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "-h":
            case "--help": PrintUsage(); return 0;
            case "--stream-from-disk": streamFromDisk = true; break;
            case "--compare-to": compareTo = NeedValue(args, ref i, "--compare-to"); break;
            case "--comparison-limit": comparisonLimitRaw = NeedValue(args, ref i, "--comparison-limit"); break;
            case "--validate": validate = true; break;
            case "--doc-separator": docSeparator = NeedValue(args, ref i, "--doc-separator"); break;
            case "--pretokenizer": pretokenizer = NeedValue(args, ref i, "--pretokenizer"); break;
            default:
                if (args[i].StartsWith("--")) { Console.Error.WriteLine($"Unknown option '{args[i]}'."); return 2; }
                if (tokenizerSpec is null) tokenizerSpec = args[i];
                else files.Add(args[i]);
                break;
        }
    }
    if (tokenizerSpec is null || files.Count == 0)
    {
        Console.Error.WriteLine("bench requires <tokenizer> and at least one file.");
        PrintUsage();
        return 2;
    }
    foreach (var f in files)
    {
        if (!File.Exists(f)) { Console.Error.WriteLine($"File not found: {f}"); return 1; }
    }
    if (validate && compareTo is null)
    {
        Console.Error.WriteLine("error: --validate requires --compare-to.");
        return 2;
    }
    long comparisonLimit;
    try
    {
        comparisonLimit = ParseSize(comparisonLimitRaw);
    }
    catch (FormatException)
    {
        Console.Error.WriteLine($"error: invalid --comparison-limit '{comparisonLimitRaw}'.");
        return 2;
    }

    Console.WriteLine($"{"cpu",9}: {CpuInfo()}");

    using var tokenizer = LoadTokenizer(tokenizerSpec, pretokenizer);
    byte[]? sepBytes = docSeparator is null ? null : Encoding.UTF8.GetBytes(docSeparator);
    if (sepBytes is not null && sepBytes.Length == 0)
        throw new CliUsageException("error: --doc-separator must not be empty.");

    uint[][] rows;
    long totalBytes;
    Stopwatch sw;
    if (streamFromDisk)
    {
        // Bare paths keep their extension-based format detection (.jsonl);
        // with a separator, hand whole files + separator to Rust (never pre-split).
        totalBytes = files.Sum(f => new FileInfo(f).Length);
        sw = Stopwatch.StartNew();
        if (sepBytes is null)
        {
            rows = tokenizer.EncodeFiles(files);
        }
        else
        {
            using var src = new TextFileSource([.. files], sepBytes);
            rows = tokenizer.EncodeFiles(src);
        }
        sw.Stop();
    }
    else
    {
        var raws = files.Select(File.ReadAllBytes).ToArray();
        totalBytes = raws.Sum(b => (long)b.Length);
        sw = Stopwatch.StartNew();
        using var src = new BytesSource(raws, sepBytes);
        rows = tokenizer.EncodeBatch(src);
        sw.Stop();
    }

    long totalTokens = rows.Sum(r => (long)r.Length);
    Report("gigatoken", sw.Elapsed, totalBytes, totalTokens);

    if (compareTo is not null)
    {
        // Port adaptation: upstream compares against the `tokenizers` library
        // ("hf"); no .NET equivalent exists, so the comparison runs a second
        // Gigatoken tokenizer on the same size-capped subset instead.
        using var other = LoadTokenizer(compareTo, pretokenizer);
        byte[][] subset;
        bool lastTruncated;
        if (streamFromDisk)
        {
            var split = SplitDocs(files.Select(File.ReadAllBytes), sepBytes);
            (subset, lastTruncated) = SubsetDocs(split, comparisonLimit);
        }
        else
        {
            var raws = files.Select(File.ReadAllBytes).ToArray();
            (subset, lastTruncated) = SubsetDocs(SplitDocs(raws, sepBytes), comparisonLimit);
        }
        if (subset.Length == 0)
        {
            Console.Error.WriteLine("error: --comparison-limit is smaller than the first document; nothing to compare");
            return 1;
        }
        var sw2 = Stopwatch.StartNew();
        uint[][] refRows = other.EncodeBatch(new BytesSource(subset, sepBytes));
        sw2.Stop();
        uint[][] ourRows = tokenizer.EncodeBatch(new BytesSource(subset, sepBytes));

        long subsetBytes = subset.Sum(b => (long)b.Length);
        Report("ref", sw2.Elapsed, subsetBytes, refRows.Sum(r => (long)r.Length));
        double speedup = (totalBytes / Math.Max(sw.Elapsed.TotalSeconds, 1e-9))
            / (subsetBytes / Math.Max(sw2.Elapsed.TotalSeconds, 1e-9));
        Console.WriteLine($"gigatoken is {speedup:F2}x {(speedup >= 1 ? "faster" : "slower")} than ref");

        if (!validate)
            return 0;
        const int truncationGuardTokens = 100; // mirrors upstream _TRUNCATION_GUARD_TOKENS
        for (int i = 0; i < subset.Length; i++)
        {
            var gtDoc = ourRows[i];
            var refDoc = refRows[i];
            if (lastTruncated && i == subset.Length - 1)
            {
                // The final document was cut at a byte limit, which can change
                // its last merges (or split a UTF-8 character), so only the
                // common prefix minus a guard is expected to match.
                int compareToLength = Math.Min(gtDoc.Length, refDoc.Length) - truncationGuardTokens;
                if (compareToLength <= 0)
                    continue;
                gtDoc = gtDoc[..compareToLength];
                refDoc = refDoc[..compareToLength];
            }
            if (!gtDoc.AsSpan().SequenceEqual(refDoc))
            {
                int mismatch = 0;
                while (mismatch < Math.Min(gtDoc.Length, refDoc.Length) && gtDoc[mismatch] == refDoc[mismatch])
                    mismatch++;
                Console.Error.WriteLine(
                    $"validation FAILED: document {i}: first mismatch at token {mismatch} " +
                    $"(gigatoken [{string.Join(", ", gtDoc.Skip(mismatch).Take(5))}]... vs " +
                    $"ref [{string.Join(", ", refDoc.Skip(mismatch).Take(5))}]..., " +
                    $"lengths {gtDoc.Length} vs {refDoc.Length})");
                return 1;
            }
        }
        string guardNote = lastTruncated ? $" (last {truncationGuardTokens} tokens of the truncated final document ignored)" : "";
        Console.WriteLine($"validation OK: {subset.Length} documents match{guardNote}");
    }
    return 0;
}

static Tokenizer LoadTokenizer(string spec, string? pretokenizer)
{
    if (spec.EndsWith(".tiktoken", StringComparison.Ordinal))
    {
        if (pretokenizer is null) throw new CliUsageException(".tiktoken input requires --pretokenizer.");
        return Tokenizer.FromTiktoken(spec, pretokenizer);
    }
    if (spec.EndsWith(".model", StringComparison.Ordinal))
    {
        if (pretokenizer is not null) throw new CliUsageException("--pretokenizer only applies to .tiktoken inputs.");
        return Tokenizer.FromSentencePiece(spec);
    }
    if (pretokenizer is not null) throw new CliUsageException("--pretokenizer only applies to .tiktoken inputs.");
    return Tokenizer.FromPretrained(spec);
}

static string NeedValue(string[] args, ref int i, string name)
{
    if (++i >= args.Length) throw new CliUsageException($"Option {name} requires a value.");
    return args[i];
}

static long ParseSize(string s)
{
    // Mirrors upstream _parse_size: decimal (SI) multipliers only — even
    // "MiB" means 1e6. Throws FormatException on bad input (upstream: BadParameter).
    s = s.Trim();
    if (s is "none" or "unlimited") return long.MaxValue;
    var m = System.Text.RegularExpressions.Regex.Match(s, @"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
    if (!m.Success)
        throw new FormatException($"'{s}' is not a valid size.");
    double value = double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
    double mult = m.Groups[2].Value.ToLowerInvariant() switch
    {
        "k" => 1e3,
        "m" => 1e6,
        "g" => 1e9,
        "t" => 1e12,
        _ => 1,
    };
    double product = value * mult;
    // Unchecked float->int overflow is platform-undefined (historically
    // long.MinValue on x64 cvttsd2si); clamp explicitly so huge limits
    // behave as unlimited instead of taking the misleading empty-subset path.
    if (double.IsNaN(product) || double.IsInfinity(product) || product >= (double)long.MaxValue)
        return long.MaxValue;
    if (product <= (double)long.MinValue)
        return long.MinValue;
    return (long)product;
}

static void Report(string name, TimeSpan seconds, long nBytes, long nTokens)
{
    double secs = Math.Max(seconds.TotalSeconds, 1e-9);
    Console.WriteLine($"{name,9}: {secs,8:F3} s | {nBytes / 1e6,10:F2} MB at {nBytes / 1e6 / secs,8:F2} MB/s | {nTokens / 1e6,8:F2} Mtok at {nTokens / 1e6 / secs,7:F2} Mtok/s");
}

static string CpuInfo() =>
    $"{System.Runtime.InteropServices.RuntimeInformation.OSArchitecture} x{Environment.ProcessorCount} ({System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription})";

/// <summary>Cap split documents at a byte limit; the final document may be
/// cut mid-stream, so it is truncated back to a UTF-8 character boundary
/// (mirrors upstream _subset_docs). Returns the subset and whether the last
/// document was truncated.</summary>
static (byte[][] Subset, bool LastTruncated) SubsetDocs(IEnumerable<byte[]> docs, long limitBytes)
{
    var subset = new List<byte[]>();
    long total = 0;
    bool lastTruncated = false;
    foreach (var doc in docs)
    {
        if (total + doc.Length > limitBytes)
        {
            int remaining = (int)Math.Max(limitBytes - total, 0);
            // Mirrors upstream doc[:room].decode("utf-8", errors="ignore")
            // .encode("utf-8"): drops ALL invalid bytes in the prefix, not
            // just a trailing partial sequence, so both sides see identical bytes.
            var truncated = DecodeIgnoreEncode(doc, remaining);
            if (truncated.Length == 0)
                return (subset.ToArray(), false);
            subset.Add(truncated);
            lastTruncated = true;
            break;
        }
        subset.Add(doc);
        total += doc.Length;
    }
    return (subset.ToArray(), lastTruncated);
}

static byte[] DecodeIgnoreEncode(byte[] doc, int count)
{
    // Port of Python errors="ignore": skip undecodable bytes, re-encode the
    // valid runes. Skipping one byte per failure matches CPython's per-byte
    // ignore for overlong/surrogate/lone-continuation/truncated-tail cases.
    var sb = new StringBuilder(count);
    int i = 0;
    while (i < count)
    {
        if (Rune.DecodeFromUtf8(new ReadOnlySpan<byte>(doc, i, count - i), out var rune, out int consumed) == OperationStatus.Done)
        {
            sb.Append(rune.ToString());
            i += consumed;
        }
        else
        {
            i += 1;
        }
    }
    return Encoding.UTF8.GetBytes(sb.ToString());
}

static IEnumerable<byte[]> SplitDocs(IEnumerable<byte[]> raws, byte[]? sep)
{
    foreach (var raw in raws)
    {
        if (sep is null) { yield return raw; continue; }
        foreach (var part in SplitOne(raw, sep))
            yield return part;
    }
}

static byte[][] SplitOne(byte[] raw, byte[] sep)
{
    // Upstream raw.split(b"") raises ValueError (empty separator); reject
    // here so the compare path cannot spin forever on IndexOf(empty) == 0.
    if (sep.Length == 0)
        throw new CliUsageException("error: --doc-separator must not be empty.");
    var parts = new List<byte[]>();
    int start = 0;
    while (true)
    {
        int idx = raw.AsSpan(start).IndexOf(sep);
        if (idx < 0) break;
        if (idx > 0) parts.Add(raw[start..(start + idx)]);
        start += idx + sep.Length;
    }
    if (start < raw.Length) parts.Add(raw[start..]);
    return parts.ToArray();
}

sealed class CliUsageException(string message) : Exception(message);
