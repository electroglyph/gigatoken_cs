namespace Gigatoken.Tests;

using Gigatoken.Examples;

/// <summary>
/// The <c>examples/</c> sources are linked into this assembly, so each fact
/// below executes the runnable example it names (mirroring upstream
/// <c>examples/{quickstart,tiktoken_drop_in,hf_drop_in,encode_files}.py</c>).
/// </summary>
public sealed class ExamplesTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("gt_examples_").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    [Fact]
    public void QuickstartExample()
    {
        var (ids, text) = Quickstart.Run(Fixtures.Gpt2Json);
        Assert.Equal([15496u, 11u, 995u, 0u], ids);
        Assert.Equal("Hello, world!", text);
    }

    [Fact]
    public void TiktokenDropInExample()
    {
        var path = Fixtures.WriteTinyTiktoken();
        try
        {
            var (ids, text) = TiktokenDropIn.Run(path, "gpt2", "AB");
            Assert.Equal([256u], ids);
            Assert.Equal("AB", text);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void HfDropInExample()
    {
        var (ids, tokens) = HfDropIn.Run(Fixtures.Gpt2Json, "Hello");
        Assert.Equal([15496u], ids);
        Assert.Single(tokens);
        Assert.Equal("Hello", tokens[0]);
    }

    [Fact]
    public void EncodeFilesExampleExample()
    {
        var file = Path.Combine(_dir, "doc.txt");
        File.WriteAllText(file, "Hello, world!");
        var rows = EncodeFilesExample.Run(Fixtures.Gpt2Json, [file]);
        Assert.Single(rows);
        Assert.Equal([15496u, 11u, 995u, 0u], rows[0]);
    }
}
