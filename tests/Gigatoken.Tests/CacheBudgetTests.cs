namespace Gigatoken.Tests;

public sealed class CacheBudgetTests
{
    [Fact]
    public void DefaultIs512MiB()
    {
        Assert.Equal(512L << 20, EncodeCache.GetMaxBytes());
    }

    [Fact]
    public void SetAndClear()
    {
        try
        {
            EncodeCache.SetMaxBytes(5L << 20);
            Assert.Equal(5L << 20, EncodeCache.GetMaxBytes());
            EncodeCache.SetMaxBytes(null);
            Assert.Null(EncodeCache.GetMaxBytes());
        }
        finally
        {
            EncodeCache.SetMaxBytes(512L << 20);
        }
    }

    [Fact]
    public void BudgetNeverChangesOutput()
    {
        using var tok = Fixtures.LoadGpt2();
        var expected = tok.Encode("Hello, world! Caching must not affect output.");
        try
        {
            EncodeCache.SetMaxBytes(1L << 20);
            using var small = Fixtures.LoadGpt2();
            Assert.Equal(expected, small.Encode("Hello, world! Caching must not affect output."));
        }
        finally
        {
            EncodeCache.SetMaxBytes(512L << 20);
        }
    }
}
