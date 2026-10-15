using System.Runtime.InteropServices;

namespace Gigatoken.Interop;

/// <summary>Locates and loads the native <c>gigatoken_cabi</c> library.</summary>
internal static class NativeLib
{
    private static int _loaded;

    internal static void EnsureLoaded()
    {
        if (Volatile.Read(ref _loaded) != 0) return;
        if (Interlocked.CompareExchange(ref _loaded, 1, 0) != 0) return;
        try
        {
            NativeLibrary.SetDllImportResolver(
                typeof(NativeLib).Assembly, Resolve);
        }
        catch
        {
            Interlocked.Exchange(ref _loaded, 0);
            throw;
        }
    }

    private static nint Resolve(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != NativeMethods.LibraryName)
            return nint.Zero;

        // 1. Explicit override for development / custom layouts.
        var env = Environment.GetEnvironmentVariable("GIGATOKEN_NATIVE");
        string? badEnv = null;
        if (env is not null)
        {
            if (NativeLibrary.TryLoad(env, out var envHandle))
                return envHandle;
            badEnv = env;
        }

        var baseDir = AppContext.BaseDirectory;
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
            // Nested crate-root checkout: <root>/target/... when baseDir is under src/Gigatoken/bin.
            if (File.Exists(Path.Combine(dir.FullName, "Cargo.toml")))
            {
                foreach (var profile in new[] { "debug", "release" })
                {
                    var p = Path.Combine(dir.FullName, "target", profile, MapName());
                    if (File.Exists(p) && NativeLibrary.TryLoad(p, out var h))
                        return h;
                }
            }
        }

        // 3. Default probing (system library paths).
        if (NativeLibrary.TryLoad(MapName(), assembly, searchPath, out var sys))
            return sys;

        throw new DllNotFoundException(
            $"Unable to load native library '{MapName()}'. " +
            (badEnv is null
                ? "GIGATOKEN_NATIVE is unset. "
                : $"GIGATOKEN_NATIVE='{badEnv}' failed to load. ") +
            $"Searched runtimes/{rid}/native next to {baseDir}, cargo target/ dirs, and system paths. " +
            "Build it with build.sh or set GIGATOKEN_NATIVE to the library path.");
    }

    private static string MapName() =>
        OperatingSystem.IsWindows() ? "gigatoken_cabi.dll" :
        OperatingSystem.IsMacOS() ? "libgigatoken_cabi.dylib" :
        "libgigatoken_cabi.so";
}
