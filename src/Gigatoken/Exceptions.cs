namespace Gigatoken;

/// <summary>Base exception for all Gigatoken errors.</summary>
public class GigatokenException : Exception
{
    /// <summary>Initializes a new instance.</summary>
    public GigatokenException(string message) : base(message) { }
}

/// <summary>Tokenizer file / JSON / model parse failure (native code 5; includes WordPiece, unknown normalizers, non-BPE .model).</summary>
/// <remarks>Planned as <c>InvalidDataException</c>, which is sealed; derives from <c>IOException</c> instead.</remarks>
public sealed class TokenizerLoadException : IOException
{
    /// <summary>Initializes a new instance.</summary>
    public TokenizerLoadException(string message) : base(message) { }
}

/// <summary>Hub download 404 (native code 3 on a hub fetch).</summary>
public sealed class HubFileNotFoundException : FileNotFoundException
{
    /// <summary>Initializes a new instance.</summary>
    public HubFileNotFoundException(string message) : base(message) { }

    /// <summary>Initializes a new instance preserving the cause.</summary>
    public HubFileNotFoundException(string message, Exception innerException)
        : base(message, innerException) { }

    /// <summary>Initializes a new instance preserving the file name and cause.</summary>
    public HubFileNotFoundException(string message, string? fileName, Exception? innerException)
        : base(message, fileName, innerException) { }
}

/// <summary>A forbidden special token was found during a checked encode (native code 6).</summary>
public sealed class SpecialTokenFoundException : GigatokenException
{
    /// <summary>Sorted pattern indices that matched.</summary>
    public IReadOnlyList<int> MatchedIndices { get; }

    /// <summary>Initializes a new instance.</summary>
    public SpecialTokenFoundException(IReadOnlyList<int> matchedIndices, string message)
        : base(message) => MatchedIndices = matchedIndices;
}
