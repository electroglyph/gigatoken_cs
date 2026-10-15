using System.Diagnostics;
using System.Runtime.InteropServices;
using Xunit;

namespace Gigatoken.Tests;

// NativeLib.LoadLibrary resolution paths (each runs in a hermetic subprocess:
// the loader pins one library per process, so env/filesystem control must
// happen before first load) plus live-hub success paths (skipped offline).
// Covers src/Gigatoken/Interop/NativeLib.cs LoadLibrary arms,
// HubCacheTrain.cs Hub.GetFile success, Tokenizer.cs hub/config routes,
// NativeBridge.cs HubFile success.
public sealed class CoverageNativeLibTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("covnativelib_").FullName;

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

    private static (int Exit, string Out, string Err) RunDotnet(
        string dll, string[] args, string? workDir, Dictionary<string, string?> env)
    {
        using var proc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "dotnet",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                WorkingDirectory = workDir ?? "",
            },
        };
        proc.StartInfo.ArgumentList.Add(dll);
        foreach (var a in args) proc.StartInfo.ArgumentList.Add(a);
        foreach (var (k, v) in env)
        {
            if (v is null) proc.StartInfo.Environment.Remove(k);
            else proc.StartInfo.Environment[k] = v;
        }
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

    private string StagedCliDir()
    {
        // Copy the CLI output next to no cargo workspace (/tmp tree) so the
        // cargo-target walk and bundled candidates both miss: forces the
        // system-probing and DllNotFoundException arms.
        var dll = FindCliDll();
        Assert.True(dll != "", "CLI dll not built (dotnet build tools/Gigatoken.Cli).");
        var dest = Path.Combine(_dir, "cli");
        Directory.CreateDirectory(dest);
        foreach (var f in Directory.GetFiles(Path.GetDirectoryName(dll)!))
            File.Copy(f, Path.Combine(dest, Path.GetFileName(f)), overwrite: true);
        return Path.Combine(dest, "Gigatoken.Cli.dll");
    }

    private string WriteDoc()
    {
        var p = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(p, string.Concat(Enumerable.Repeat("Hello, world! Bench probe.\n", 20)));
        return p;
    }

    [Fact]
    public void EnvOverrideValidLoads()
    {
        // GIGATOKEN_NATIVE pointing at a good library: LoadLibrary env arm.
        // cwd under /tmp also defeats the cargo walk, proving env wins first.
        var staged = StagedCliDir();
        var (exit, _, err) = RunDotnet(staged,
            ["bench", Fixtures.Gpt2Json, WriteDoc()], _dir,
            new() { ["GIGATOKEN_NATIVE"] = "/home/anon/gigatoken_cs/target/release/libgigatoken_cabi.so" });
        Assert.Equal(0, exit);
        Assert.DoesNotContain("Unable to load native library", err);
    }

    [Fact]
    public void EnvOverrideBadFailsWithEcho()
    {
        // GIGATOKEN_NATIVE pointing at garbage: badEnv echo in the throw arm.
        var staged = StagedCliDir();
        var bad = Path.Combine(_dir, "nope.so");
        var (exit, _, err) = RunDotnet(staged,
            ["bench", Fixtures.Gpt2Json, WriteDoc()], _dir,
            new() { ["GIGATOKEN_NATIVE"] = bad });
        Assert.Equal(1, exit);
        Assert.Contains($"GIGATOKEN_NATIVE='{bad}' failed to load", err);
    }

    [Fact]
    public void NoLibraryAnywhereThrowsUnset()
    {
        // No env, no workspace, no system lib: system-probing + unset throw arm.
        var staged = StagedCliDir();
        var (exit, _, err) = RunDotnet(staged,
            ["bench", Fixtures.Gpt2Json, WriteDoc()], _dir,
            new() { ["GIGATOKEN_NATIVE"] = null });
        Assert.Equal(1, exit);
        Assert.Contains("Unable to load native library", err);
        Assert.Contains("GIGATOKEN_NATIVE is unset", err);
    }

    [Fact]
    public void HubDownloadSuccess()
    {
        // Live hub success: Hub.GetFile return + NativeBridge.HubFile success.
        string path;
        // Skipped where there is no network (same policy as the upstream sweeps).
        try { path = Hub.GetFile("gpt2", "tokenizer.json"); }
        catch { return; }
        Assert.True(File.Exists(path));
        Assert.True(new FileInfo(path).Length > 100_000);
    }

    [Fact]
    public void FromPretrainedHubRepoLoads()
    {
        // Hub + config routes: Tokenizer.cs hub/config arms + LoadFileRoute.
        Tokenizer tok;
        try { tok = Tokenizer.FromPretrained("gpt2"); }
        catch { return; }
        using (tok)
        {
            Assert.Equal([15496u, 11u, 995u, 0u], tok.Encode("Hello, world!"));
        }
    }
}

// In-process LoadLibrary arms (no global loader state touched: the overload
// takes baseDir/env as parameters and returned handles are freed).
public sealed class CoverageLoadLibraryTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("covloadlib_").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private static string? FindNativeLib()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
            foreach (var profile in new[] { "debug", "release" })
            {
                var p = Path.Combine(dir.FullName, "target", profile, "libgigatoken_cabi.so");
                if (File.Exists(p)) return p;
            }
        return null;
    }

    private static string EmptyDir(string name)
    {
        var p = Path.Combine(Path.GetTempPath(), name + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(p);
        return p;
    }

    [Fact]
    public void EnvHitReturnsHandle()
    {
        var lib = FindNativeLib();
        if (lib is null) return;
        var h = Gigatoken.Interop.NativeLib.LoadLibrary(_dir, lib);
        Assert.NotEqual(nint.Zero, h);
        NativeLibrary.Free(h);
    }

    [Fact]
    public void EnvMissFallsThroughToWorkspace()
    {
        // Bad env path does not abort: the cargo walk still finds the lib.
        var h = Gigatoken.Interop.NativeLib.LoadLibrary(AppContext.BaseDirectory, "/nonexistent/x.so");
        Assert.NotEqual(nint.Zero, h);
        NativeLibrary.Free(h);
    }

    [Fact]
    public void BundledCandidatesHit()
    {
        var lib = FindNativeLib();
        if (lib is null) return;
        var rid = System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier;
        var ridDir = Path.Combine(_dir, "runtimes", rid, "native");
        Directory.CreateDirectory(ridDir);
        File.Copy(lib, Path.Combine(ridDir, "libgigatoken_cabi.so"));
        var h1 = Gigatoken.Interop.NativeLib.LoadLibrary(_dir, null);
        Assert.NotEqual(nint.Zero, h1);
        NativeLibrary.Free(h1);

        var flat = Path.Combine(_dir, "flat");
        Directory.CreateDirectory(flat);
        File.Copy(lib, Path.Combine(flat, "libgigatoken_cabi.so"));
        var h2 = Gigatoken.Interop.NativeLib.LoadLibrary(flat, null);
        Assert.NotEqual(nint.Zero, h2);
        NativeLibrary.Free(h2);
    }

    [Fact]
    public void CargoWalkHit()
    {
        var h = Gigatoken.Interop.NativeLib.LoadLibrary(AppContext.BaseDirectory, null);
        Assert.NotEqual(nint.Zero, h);
        NativeLibrary.Free(h);
    }

    [Fact]
    public void SystemProbeHitWhenVisible()
    {
        // glibc reads LD_LIBRARY_PATH once at process startup, so in-process
        // manipulation cannot enable this arm. It runs when the parent
        // environment exposes the lib (verified manually with
        // LD_LIBRARY_PATH preset: passes); otherwise there is nothing
        // deterministic to do — the fallthrough throw is covered below.
        if (!NativeLibrary.TryLoad("libgigatoken_cabi.so", typeof(object).Assembly, null, out var pre))
            return;
        NativeLibrary.Free(pre);
        var h = Gigatoken.Interop.NativeLib.LoadLibrary(EmptyDir("covbase_"), null);
        Assert.NotEqual(nint.Zero, h);
        NativeLibrary.Free(h);
    }

    [Fact]
    public void ThrowMentionsUnset()
    {
        var ex = Assert.Throws<DllNotFoundException>(() =>
            Gigatoken.Interop.NativeLib.LoadLibrary(EmptyDir("covmiss_"), null));
        Assert.Contains("GIGATOKEN_NATIVE is unset", ex.Message);
        Assert.Contains("runtimes/", ex.Message);
    }

    [Fact]
    public void ThrowEchoesBadEnv()
    {
        const string bad = "/nonexistent/y.so";
        var ex = Assert.Throws<DllNotFoundException>(() =>
            Gigatoken.Interop.NativeLib.LoadLibrary(EmptyDir("covmiss_"), bad));
        Assert.Contains($"GIGATOKEN_NATIVE='{bad}' failed to load", ex.Message);
    }
}
