using System.Text;
using Gigatoken.Interop;

namespace Gigatoken;

/// <summary>A file-backed document source for <c>EncodeFiles</c> / training. Holds a native source handle.</summary>
public interface IFileSource : IDisposable
{
    internal SourceHandle NativeHandle { get; }
}

/// <summary>Base class for file sources (path-list marshalling shared).</summary>
public abstract class FileSourceBase : IFileSource
{
    private SourceHandle? _native;
    private int _disposed;

    /// <summary>Initializes the native handle via <paramref name="create"/>.</summary>
    internal FileSourceBase(Func<IReadOnlyList<byte[]>, SourceHandle> create, IEnumerable<string> paths)
    {
        ArgumentNullException.ThrowIfNull(create);
        ArgumentNullException.ThrowIfNull(paths);
        var list = paths as IList<string> ?? paths.ToList();
        var bufs = new byte[list.Count][];
        for (var i = 0; i < list.Count; i++)
            bufs[i] = NativeBridge.Utf8Bytes(Path.GetFullPath(list[i] ?? throw new ArgumentNullException($"{nameof(paths)}[{i}]")));
        _native = create(bufs);
    }

    SourceHandle IFileSource.NativeHandle =>
        Volatile.Read(ref _disposed) != 0 || _native is null
            ? throw new ObjectDisposedException(GetType().Name)
            : _native;

    /// <summary>Releases the native source.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _native?.Dispose();
            _native = null;
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>Plain-text files; one document per file unless <c>Separator</c> is set.</summary>
/// <param name="Paths">Files to read.</param>
/// <param name="Separator">Optional UTF-8 document separator (null = one doc per file).</param>
public sealed class TextFileSource(string[] Paths, byte[]? Separator = null)
    : FileSourceBase(bufs => NativeBridge.TextSource(bufs, Separator), Paths)
{
    /// <summary>Creates a source with a string separator (UTF-8 encoded).</summary>
    public TextFileSource(string[] paths, string? separator)
        : this(paths, separator is null ? null : Encoding.UTF8.GetBytes(separator)) { }
}

/// <summary>JSONL files (one JSON object per line, text taken from <c>Field</c>).</summary>
/// <param name="Paths">Files to read.</param>
/// <param name="Field">JSON field holding the text (default <c>"text"</c>).</param>
public sealed class JsonlFileSource(string[] Paths, string Field = "text")
    : FileSourceBase(bufs => NativeBridge.JsonlSource(bufs, NativeBridge.Utf8Bytes(Field)), Paths);

/// <summary>Parquet files (text taken from <c>Column</c>; null cells decode as empty docs).</summary>
/// <param name="Paths">Files to read.</param>
/// <param name="Column">String/binary column holding the text (default <c>"text"</c>).</param>
public sealed class ParquetFileSource(string[] Paths, string Column = "text")
    : FileSourceBase(bufs => NativeBridge.ParquetSource(bufs, NativeBridge.Utf8Bytes(Column)), Paths);

/// <summary>
/// In-memory documents. Buffers are passed whole to Rust with the separator —
/// Rust splits (never pre-split in C#). A single <c>byte[]</c> maps to a
/// one-element buffer list (mirrors Python <c>bytes | list[bytes]</c>).
/// </summary>
/// <remarks>The separator array is snapshotted at construction; later caller
/// mutation does not affect encoding.</remarks>
public sealed class BytesSource : IDisposable
{
    internal IReadOnlyList<byte[]> Buffers { get; }
    internal byte[]? Separator { get; }

    /// <summary>Creates a source from buffers with an optional raw-byte separator.</summary>
    /// <exception cref="ArgumentNullException">Any buffer array or element is null.</exception>
    public BytesSource(ReadOnlyMemory<byte>[] buffers, byte[]? separator = null)
    {
        ArgumentNullException.ThrowIfNull(buffers);
        Buffers = buffers.Select(b => b.ToArray()).ToArray();
        Separator = separator?.ToArray();
    }

    /// <summary>Creates a source from a single buffer.</summary>
    public BytesSource(ReadOnlyMemory<byte> singleBuffer, byte[]? separator = null)
        : this([singleBuffer], separator) { }

    /// <summary>Creates a source from byte-array buffers.</summary>
    /// <exception cref="ArgumentNullException">Any buffer element is null.</exception>
    public BytesSource(byte[][] buffers, byte[]? separator = null)
        : this(Checked(buffers), separator) { }

    private static ReadOnlyMemory<byte>[] Checked(byte[][] buffers)
    {
        ArgumentNullException.ThrowIfNull(buffers);
        var out_ = new ReadOnlyMemory<byte>[buffers.Length];
        for (var i = 0; i < buffers.Length; i++)
            out_[i] = buffers[i] ?? throw new ArgumentNullException($"{nameof(buffers)}[{i}]");
        return out_;
    }

    /// <summary>Creates a source from buffers with a string separator (UTF-8 encoded).</summary>
    public BytesSource(ReadOnlyMemory<byte>[] buffers, string? separator)
        : this(buffers, separator is null ? null : Encoding.UTF8.GetBytes(separator)) { }

    /// <summary>Releases managed state (buffers are copied per call; nothing native to free).</summary>
    public void Dispose() { }
}
