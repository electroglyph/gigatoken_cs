namespace Gigatoken.Tests;

/// <summary>
/// Parquet file-source tests against the checked-in <c>Fixtures/tiny.parquet</c>
/// (pyarrow, string column <c>"text"</c> plus an int column <c>"n"</c>; row 2 is
/// null, row 4 is empty). Regenerate with <c>tools/export_goldens.py --parquet</c>.
/// </summary>
public sealed class ParquetFileSourceTests
{
    private static string ParquetPath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny.parquet");

    [Fact]
    public void ReadsRowsInOrderWithNullAsEmpty()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = new ParquetFileSource([ParquetPath]);
        var rows = tok.EncodeFiles(src);
        Assert.Equal(5, rows.Length);
        Assert.Equal(tok.Encode("hello world"), rows[0]);
        Assert.Equal(tok.Encode("second doc with unicode caf\u00E9"), rows[1]);
        Assert.Empty(rows[2]); // null cell decodes as an empty doc
        Assert.Equal(tok.Encode("line1\nline2"), rows[3]);
        Assert.Empty(rows[4]);
    }

    [Fact]
    public void MissingColumnThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = new ParquetFileSource([ParquetPath], "nope");
        Assert.ThrowsAny<Exception>(() => tok.EncodeFiles(src));
    }

    [Fact]
    public void NonStringColumnThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = new ParquetFileSource([ParquetPath], "n");
        Assert.ThrowsAny<Exception>(() => tok.EncodeFiles(src));
    }

    [Fact]
    public void MissingFileThrows()
    {
        using var tok = Fixtures.LoadGpt2();
        using var src = new ParquetFileSource(
            [Path.Combine(Path.GetTempPath(), "no-such-file.parquet")]);
        Assert.ThrowsAny<Exception>(() => tok.EncodeFiles(src));
    }

    [Fact]
    public void TrainFromParquetSource()
    {
        using var src = new ParquetFileSource([ParquetPath]);
        var (vocab, merges) = BpeTrainer.Train(src, 260, []);
        Assert.Equal(260, vocab.Count);
        Assert.Equal(260 - 256, merges.Count);
        for (var b = 0; b < 256; b++)
            Assert.Equal([(byte)b], vocab[(uint)b]);
    }
}
