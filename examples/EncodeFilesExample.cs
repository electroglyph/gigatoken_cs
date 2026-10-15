// Encode files from disk without loading them into memory first.
// Runnable copy: executed by ExamplesTests.EncodeFilesExample.
namespace Gigatoken.Examples;

using Gigatoken;

/// <summary>Streaming file encode through <see cref="TextFileSource"/>.</summary>
public static class EncodeFilesExample
{
    /// <summary>Encodes each file as one document.</summary>
    /// <returns>One id array per file.</returns>
    public static uint[][] Run(string tokenizerPath, string[] files)
    {
        using var tok = Tokenizer.FromPretrained(tokenizerPath);
        using var src = new TextFileSource(files);
        return tok.EncodeFiles(src);
    }
}
