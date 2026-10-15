using System.Runtime.InteropServices;

namespace Gigatoken.Interop;

internal sealed class TokenizerHandle : SafeHandle
{
    public TokenizerHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static TokenizerHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid tokenizer handle");
        var h = new TokenizerHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_tokenizer_free(handle);
        return true;
    }
}

internal sealed class SourceHandle : SafeHandle
{
    public SourceHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static SourceHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid source handle");
        var h = new SourceHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_source_free(handle);
        return true;
    }
}

internal sealed class MatcherHandle : SafeHandle
{
    public MatcherHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static MatcherHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid matcher handle");
        var h = new MatcherHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_matcher_free(handle);
        return true;
    }
}

internal sealed class WrapOptsHandle : SafeHandle
{
    public WrapOptsHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static WrapOptsHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid wrap-opts handle");
        var h = new WrapOptsHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_wrap_opts_free(handle);
        return true;
    }
}

internal sealed class PadOptsHandle : SafeHandle
{
    public PadOptsHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static PadOptsHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid pad-opts handle");
        var h = new PadOptsHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_pad_opts_free(handle);
        return true;
    }
}

internal sealed class PretokenIterHandle : SafeHandle
{
    public PretokenIterHandle() : base(nint.Zero, ownsHandle: true) { }
    internal static PretokenIterHandle FromRaw(nint raw)
    {
        if (raw == nint.Zero || raw == new nint(-1))
            throw new GigatokenException("native returned an invalid pretoken-iterator handle");
        var h = new PretokenIterHandle();
        h.SetHandle(raw);
        return h;
    }
    public override bool IsInvalid => handle == nint.Zero || handle == new nint(-1);
    protected override bool ReleaseHandle()
    {
        NativeMethods.gt_pretokenize_iter_free(handle);
        return true;
    }
}
