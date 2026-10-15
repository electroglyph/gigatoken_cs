namespace Gigatoken;

/// <summary>Backend kind of a <see cref="Tokenizer"/>.</summary>
public enum TokenizerKind
{
    /// <summary>Byte-pair encoding backend.</summary>
    Bpe = 0,
    /// <summary>SentencePiece BPE + byte-fallback backend.</summary>
    SentencePiece = 1,
}

/// <summary>Hugging Face Hub repository type.</summary>
public enum RepoType
{
    /// <summary>Model repository.</summary>
    Model = 0,
    /// <summary>Dataset repository.</summary>
    Dataset = 1,
    /// <summary>Space repository.</summary>
    Space = 2,
}

/// <summary>BPE training merge tie-breaking (mirrors upstream <c>TieBreaking</c>).</summary>
public enum TieBreaking
{
    /// <summary>Hugging Face ordering.</summary>
    HuggingFace = 0,
    /// <summary>Raw token-id ordering.</summary>
    RawTokenIds = 1,
    /// <summary>Assembled-bytes ordering.</summary>
    AssembledBytes = 2,
}
