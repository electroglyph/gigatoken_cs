using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
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

    private static string FindCargoLib()
    {
        // Same layout NativeLib falls back to: walk up to the cargo
        // workspace and take target/release (CI gate) or target/debug.
        var name = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "gigatoken_cabi.dll"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libgigatoken_cabi.dylib"
            : "libgigatoken_cabi.so";
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            foreach (var profile in new[] { "release", "debug" })
            {
                var p = Path.Combine(dir.FullName, "target", profile, name);
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
        var lib = FindCargoLib();
        Assert.True(lib != "", "cargo cdylib not built (run ./build.sh).");
        var (exit, _, err) = RunDotnet(staged,
            ["bench", Fixtures.Gpt2Json, WriteDoc()], _dir,
            new() { ["GIGATOKEN_NATIVE"] = lib });
        Assert.Equal(0, exit);
        Assert.DoesNotContain("Unable to load native library", err);
    }

    // Minimal localhost hub stub: answers every request with one fixed
    // status so the HTTP-status → typed-exception mapping is tested
    // hermetically (real huggingface.co answers 401 or 404 for missing
    // repos depending on edge behavior, which CI cannot rely on).
    private sealed class HubStub : IDisposable
    {
        private readonly TcpListener _listener;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _serve;
        public int Port { get; }

        public HubStub(int status, string reason)
        {
            _listener = new TcpListener(IPAddress.Loopback, 0);
            _listener.Start();
            Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
            var response = Encoding.ASCII.GetBytes(
                $"HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            _serve = Task.Run(async () =>
            {
                while (!_cts.Token.IsCancellationRequested)
                {
                    TcpClient client;
                    try { client = await _listener.AcceptTcpClientAsync(_cts.Token); }
                    catch { break; }
                    _ = Task.Run(async () =>
                    {
                        using (client)
                        {
                            var stream = client.GetStream();
                            var seen = new StringBuilder();
                            var buf = new byte[4096];
                            while (seen.Length < 65536)
                            {
                                int n;
                                try { n = await stream.ReadAsync(buf, _cts.Token); }
                                catch { return; }
                                if (n == 0) break;
                                seen.Append(Encoding.ASCII.GetString(buf, 0, n));
                                if (seen.ToString().Contains("\r\n\r\n")) break;
                            }
                            try { await stream.WriteAsync(response, _cts.Token); }
                            catch { }
                        }
                    });
                }
            });
        }

        public void Dispose()
        {
            try { _cts.Cancel(); } catch { }
            try { _listener.Stop(); } catch { }
            try { _serve.Wait(5000); } catch { }
        }
    }

    private (int Exit, string Err) BenchRepoAgainstStub(
        string repo, HubStub stub, string lib, string dir, string doc)
    {
        var staged = StagedCliDir();
        var (exit, _, err) = RunDotnet(staged,
            ["bench", repo, doc], dir,
            new()
            {
                ["GIGATOKEN_NATIVE"] = lib,
                ["HF_ENDPOINT"] = $"http://127.0.0.1:{stub.Port}",
                ["HTTP_PROXY"] = null,
                ["HTTPS_PROXY"] = null,
                ["http_proxy"] = null,
                ["https_proxy"] = null,
            });
        return (exit, err);
    }

    [Fact]
    public void StubHub401SurfacesUnauthorized()
    {
        using var stub = new HubStub(401, "Unauthorized");
        var lib = FindCargoLib();
        Assert.True(lib != "", "cargo cdylib not built (run ./build.sh).");
        var (exit, err) = BenchRepoAgainstStub("stuborg401/stub-repo-401", stub, lib, _dir, WriteDoc());
        Assert.NotEqual(0, exit);
        Assert.Contains("HTTP 401", err);
    }

    [Fact]
    public void StubHub404SurfacesHubFileNotFound()
    {
        using var stub = new HubStub(404, "Not Found");
        var lib = FindCargoLib();
        Assert.True(lib != "", "cargo cdylib not built (run ./build.sh).");
        var (exit, err) = BenchRepoAgainstStub("stuborg404/stub-repo-404", stub, lib, _dir, WriteDoc());
        Assert.NotEqual(0, exit);
        Assert.Contains("HTTP 404", err);
    }

    [Fact]
    public void EnvOverrideBadFailsWithEcho()
    {
        // GIGATOKEN_NATIVE pointing at garbage: fail-fast, no silent fallback.
        var staged = StagedCliDir();
        var bad = Path.Combine(_dir, "nope.so");
        var (exit, _, err) = RunDotnet(staged,
            ["bench", Fixtures.Gpt2Json, WriteDoc()], _dir,
            new() { ["GIGATOKEN_NATIVE"] = bad });
        Assert.Equal(1, exit);
        Assert.Contains($"GIGATOKEN_NATIVE='{bad}'", err);
        Assert.Contains("no fallback probing was attempted", err);
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
    public void EnvMissFailsFast()
    {
        // Bad env path is a hard error: no fallback to the cargo walk.
        var ex = Assert.Throws<DllNotFoundException>(() =>
            Gigatoken.Interop.NativeLib.LoadLibrary(AppContext.BaseDirectory, "/nonexistent/x.so"));
        Assert.Contains("GIGATOKEN_NATIVE='/nonexistent/x.so'", ex.Message);
        Assert.Contains("no fallback probing was attempted", ex.Message);
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
        Assert.Contains($"GIGATOKEN_NATIVE='{bad}'", ex.Message);
        Assert.Contains("no fallback probing was attempted", ex.Message);
    }
}
