using System.Text;

namespace Gigatoken.Tests;

public sealed class FileSourceTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("gt_files_").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Write(string name, string content)
    {
        var p = Path.Combine(_dir, name);
        File.WriteAllText(p, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        return p;
    }

    [Fact]
    public void TextSourceOneDocPerFile()
    {
        using var tok = Fixtures.LoadGpt2();
        var a = Write("a.txt", "Hello, world!");
        var b = Write("b.txt", "Second doc.");
        using var src = new TextFileSource([a, b]);
        var rows = tok.EncodeFiles(src);
        Assert.Equal(2, rows.Length);
        Assert.Equal(tok.Encode("Hello, world!"), rows[0]);
        Assert.Equal(tok.Encode("Second doc."), rows[1]);
    }

    [Fact]
    public void TextSourceWithSeparator()
    {
        using var tok = Fixtures.LoadGpt2();
        var a = Write("a.txt", "one<|sep|>two<|sep|>");
        using var src = new TextFileSource([a], "<|sep|>");
        var rows = tok.EncodeFiles(src);
        Assert.Equal(2, rows.Length);
        Assert.Equal(tok.Encode("one"), rows[0]);
        Assert.Equal(tok.Encode("two"), rows[1]);
    }

    [Fact]
    public void JsonlSource()
    {
        using var tok = Fixtures.LoadGpt2();
        var f = Write("docs.jsonl", "{\"text\": \"Hello\"}\n{\"text\": \"world\"}\n");
        using var src = new JsonlFileSource([f]);
        var rows = tok.EncodeFiles(src);
        Assert.Equal(2, rows.Length);
        Assert.Equal(tok.Encode("Hello"), rows[0]);
    }

    [Fact]
    public void JsonlGzipSource()
    {
        using var tok = Fixtures.LoadGpt2();
        var f = Path.Combine(_dir, "docs.jsonl.gz");
        using (var fs = File.Create(f))
        using (var gz = new System.IO.Compression.GZipStream(fs, System.IO.Compression.CompressionLevel.Fastest))
            gz.Write(Encoding.UTF8.GetBytes("{\"text\": \"Hello\"}\n"));
        using var src = new JsonlFileSource([f]);
        var rows = tok.EncodeFiles(src);
        Assert.Single(rows);
        Assert.Equal(tok.Encode("Hello"), rows[0]);
    }

    [Fact]
    public void BytesSourceSplit()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = new BytesSource(
            [(ReadOnlyMemory<byte>)Encoding.UTF8.GetBytes("a<|sep|>b<|sep|>")], "<|sep|>");
        var rows = tok.EncodeBatch(src);
        Assert.Equal(2, rows.Length);
        Assert.Equal(tok.Encode("a"), rows[0]);
        Assert.Equal(tok.Encode("b"), rows[1]);
    }

    [Fact]
    public void BarePathOverloads()
    {
        using var tok = Fixtures.LoadGpt2();
        var a = Write("a.txt", "Hello");
        var rows = tok.EncodeFiles(a);
        Assert.Single(rows);
        Assert.Equal(tok.Encode("Hello"), rows[0]);
        var rows2 = tok.EncodeFiles((IEnumerable<string>)[a]);
        Assert.Equal(rows[0], rows2[0]);
    }

    [Fact]
    public void MissingFileThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        var missing = Path.Combine(_dir, "nope.txt");
        using var src = new TextFileSource([missing]);
        Assert.ThrowsAny<Exception>(() => tok.EncodeFiles(src));
    }
}
