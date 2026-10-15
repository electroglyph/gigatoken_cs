using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

namespace Gigatoken.Interop;

/// <summary>Locates and loads the native <c>gigatoken_cabi</c> library.</summary>
internal static class NativeLib
{
    private static int _loaded;
    private static nint _handle;

    internal static void EnsureLoaded()
    {
        if (Volatile.Read(ref _loaded) != 0) return;
        if (Interlocked.CompareExchange(ref _loaded, 1, 0) != 0) return;
        try
        {
            // Eager single resolution: the DllImportResolver fires lazily per
            // P/Invoke method, so without pinning, a mid-process environment or
            // filesystem change could split bindings across two live library
            // objects with independent thread-local error state. Resolve once
            // here; Resolve() just replays the pinned handle. The assignment
            // runs after SetDllImportResolver so a failed registration keeps
            // the previous handle (if any) instead of zeroing it.
            var handle = LoadLibrary();
            NativeLibrary.SetDllImportResolver(
                typeof(NativeLib).Assembly, Resolve);
            _handle = handle;
        }
        catch
        {
            Interlocked.Exchange(ref _loaded, 0);
            throw;
        }
    }

    private static nint Resolve(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath) =>
        libraryName != NativeMethods.LibraryName ? nint.Zero : _handle;
    private static nint LoadLibrary() =>
        LoadLibrary(AppContext.BaseDirectory, Environment.GetEnvironmentVariable("GIGATOKEN_NATIVE"));

    // (baseDir, env) parameters exist for testability: the suite drives every
    // arm in-process with temp dirs without touching process-global loader
    // state (the pinned _handle is only assigned in EnsureLoaded).
    internal static nint LoadLibrary(string baseDir, string? env)
    {
        // 1. Explicit override for development / custom layouts.
        string? badEnv = null;
        if (env is not null)
        {
            if (NativeLibrary.TryLoad(env, out var envHandle))
                return envHandle;
            badEnv = env;
        }

        var rid = RuntimeInformation.RuntimeIdentifier;
        string[] candidates =
        [
            Path.Combine(baseDir, "runtimes", rid, "native", MapName()),
            Path.Combine(baseDir, MapName()),
        ];
        foreach (var c in candidates)
        {
            if (NativeLibrary.TryLoad(c, out var h))
                return h;
        }

        // 2. Cargo workspace layouts (dev inner-loop: tests run without build.sh).
        var dir = new DirectoryInfo(baseDir);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            foreach (var profile in new[] { "debug", "release" })
            {
                var p = Path.Combine(dir.FullName, "target", profile, MapName());
                if (File.Exists(p) && NativeLibrary.TryLoad(p, out var h))
                    return h;
            }
        }

        // 3. Default probing (system library paths).
        if (NativeLibrary.TryLoad(MapName(), typeof(NativeLib).Assembly, null, out var sys))
            return sys;

        string budgetMsg;
        if (badEnv is null) budgetMsg = "GIGATOKEN_NATIVE is unset. ";
        else budgetMsg = $"GIGATOKEN_NATIVE='{badEnv}' failed to load. ";
        throw new DllNotFoundException(
            $"Unable to load native library '{MapName()}'. " + budgetMsg +
            $"Searched runtimes/{rid}/native next to {baseDir}, cargo target/ dirs, and system paths. " +
            "Build it with build.sh or set GIGATOKEN_NATIVE to the library path.");
    }

    // Excluded: Windows/macOS library names are untestable on Linux CI.
    [ExcludeFromCodeCoverage]
    private static string MapName() =>
        OperatingSystem.IsWindows() ? "gigatoken_cabi.dll" :
        OperatingSystem.IsMacOS() ? "libgigatoken_cabi.dylib" :
        "libgigatoken_cabi.so";
}
