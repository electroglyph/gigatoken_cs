using System.Diagnostics;

namespace Gigatoken.Tests;

/// <summary>
/// Process-isolation tests: the CLI runs as a real subprocess (the closest
/// .NET analogue of upstream <c>test_multiprocessing.py</c> — .NET has no
/// <c>fork</c>, and in-process parallelism is covered by
/// <c>EncodeTests.SerialMatchesParallel</c>). Requires the CLI to be built
/// (<c>dotnet build tools/Gigatoken.Cli</c>, also done by CI before test).
/// </summary>
public sealed class ProcessIsolationTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("gt_proc_").FullName;

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
            StartInfo = new ProcessStartInfo("dotnet", $"\"{dll}\" {string.Join(' ', args)}")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };
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

    [Fact]
    public void BenchInSubprocess()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(doc, "Hello, world! This is a subprocess test.\n");
        var (exit, stdout, _) = RunCli(dll, "bench", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void StreamFromDiskInSubprocess()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(doc, "Hello, world!\n");
        var (exit, stdout, _) = RunCli(dll, "bench", "--stream-from-disk", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("gigatoken:", stdout);
    }

    [Fact]
    public void CompareToSelfValidatesOkInSubprocess()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(doc, "Hello, world!\n");
        var (exit, stdout, _) = RunCli(dll, "bench", "--compare-to", Fixtures.Gpt2Json,
            "--validate", Fixtures.Gpt2Json, doc);
        Assert.Equal(0, exit);
        Assert.Contains("validation OK", stdout);
    }

    [Fact]
    public void ValidateWithoutCompareFailsInSubprocess()
    {
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var doc = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(doc, "Hello, world!\n");
        var (exit, _, _) = RunCli(dll, "bench", "--validate", Fixtures.Gpt2Json, doc);
        Assert.Equal(2, exit);
    }
}
