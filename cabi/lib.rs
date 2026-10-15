//! C ABI for the .NET port (`Gigatoken` NuGet): opaque handles + `int32`
//! error codes over the vendored upstream core.
//!
//! The upstream `gigatoken` crate cannot be used as a path dependency (its
//! manifest uses an unstable `profile-*` key without a `cargo-features`
//! header, and its modules are `pub(crate)`), so the Python-free core
//! modules are vendored here via `#[path]` includes — the same files,
//! compiled as part of *this* crate. `bindings/` (PyO3/numpy) is NOT
//! vendored; its portable logic (wrap/pad row assembly, matcher scan,
//! file scaffolding, train input handling, cache budget, hub error mapping)
//! is reimplemented below against the C calling convention.
//!
//! Error codes (all fns return `i32` unless noted):
//! `0 = OK`, `1 = null/invalid arg`, `2 = utf8 error`, `3 = io/not-found`,
//! `4 = unauthorized (401/403)`, `5 = load/parse error`,
//! `6 = special-token-found` (indices via `gt_special_hit_*`),
//! `7 = internal`.
//!
//! Buffers: callee allocates with `libc::malloc`, caller frees each pointer
//! with `gt_free`. Every entry point catches unwinds (panic -> code 7) and
//! records a thread-local message readable via `gt_last_error_*`.

#![feature(portable_simd)]

#[path = "../../gigatoken/src/token.rs"]
pub mod token;
#[path = "../../gigatoken/src/batch.rs"]
pub mod batch;
#[path = "../../gigatoken/src/bpe_train.rs"]
pub mod bpe_train;
#[path = "../../gigatoken/src/bpe/mod.rs"]
pub mod bpe;
#[path = "../../gigatoken/src/input/mod.rs"]
pub mod input;
#[path = "../../gigatoken/src/load_tokenizer/mod.rs"]
pub mod load_tokenizer;
#[path = "../../gigatoken/src/pretokenize/mod.rs"]
pub mod pretokenize;

pub use crate::bpe::Tokenizer;
pub use crate::bpe::sentencepiece::EncodeState;
pub use crate::batch::{WorkerPool, encode_docs_ragged, sp_encode_docs_ragged};

mod sp_model;

use crate::bpe::sentencepiece::SentencePieceBPE;
use crate::input::file_source::{DocFormat, FileSourceSpec, detect_default_format, load_file};
use crate::input::{MmappedFile, Resource};
use crate::load_tokenizer::hf::{HfTokenizer, load_hf_bpe, load_hf_sentencepiece, load_hf_slice};
use crate::load_tokenizer::hub::{self, FetchError, RepoType};
use crate::token::TokenId;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Error model + allocation helpers
// ---------------------------------------------------------------------------

pub const GT_OK: i32 = 0;
pub const GT_INVALID_ARG: i32 = 1;
pub const GT_UTF8: i32 = 2;
pub const GT_IO: i32 = 3;
pub const GT_UNAUTHORIZED: i32 = 4;
pub const GT_LOAD: i32 = 5;
pub const GT_SPECIAL: i32 = 6;
pub const GT_INTERNAL: i32 = 7;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static SPECIAL_HITS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn set_err(code: i32, msg: String) -> i32 {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
    code
}

fn clear_last_error() {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
}

fn clear_special_hits() {
    SPECIAL_HITS.with(|h| h.borrow_mut().clear());
}

/// Free a callee-malloc'd buffer published to the caller (used to reclaim a
/// first output when a later output fails, so partial publishes never leak).
unsafe fn free_published(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr) };
    }
}

fn guard<F: FnOnce() -> i32>(f: F) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(code) => {
            if code == GT_OK {
                // Success: stale thread-local state must not survive.
                clear_last_error();
                clear_special_hits();
            } else if code != GT_SPECIAL {
                // Any non-forbid result invalidates a previous forbid hit
                // list; the error message itself was just set via set_err.
                // (Note: code 1 from gt_get_hf_token means "has token"
                // success, not an error; that caller clears LAST_ERROR
                // explicitly before returning 1.)
                clear_special_hits();
            }
            code
        }
        Err(_) => {
            clear_special_hits();
            set_err(GT_INTERNAL, "internal panic".to_string())
        }
    }
}

/// Copy `v` into a `malloc`'d buffer; empty vecs yield a null pointer.
fn malloc_copy<T: Copy>(v: &[T]) -> *mut T {
    if v.is_empty() {
        return std::ptr::null_mut();
    }
    let bytes = v.len() * std::mem::size_of::<T>();
    let p = unsafe { libc::malloc(bytes) } as *mut T;
    if p.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
    p
}

/// malloc `v` and publish through `out_ptr`/`out_len`; code 7 on OOM.
fn out_vec<T: Copy>(v: Vec<T>, out_ptr: *mut *mut T, out_len: *mut usize) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return set_err(GT_INVALID_ARG, "null out pointer".to_string());
    }
    let len = v.len();
    let p = malloc_copy(&v);
    if p.is_null() && !v.is_empty() {
        return set_err(GT_INTERNAL, "out of memory".to_string());
    }
    unsafe {
        *out_ptr = p;
        *out_len = len;
    }
    GT_OK
}

unsafe fn read_bytes<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], i32> {
    if ptr.is_null() {
        if len == 0 {
            return Ok(&[]);
        }
        return Err(set_err(GT_INVALID_ARG, "null pointer".to_string()));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, i32> {
    let b = unsafe { read_bytes(ptr, len) }?;
    std::str::from_utf8(b).map_err(|_| set_err(GT_UTF8, "invalid UTF-8".to_string()))
}

unsafe fn read_u32s<'a>(ptr: *const u32, len: usize) -> Result<&'a [u32], i32> {
    if ptr.is_null() {
        if len == 0 {
            return Ok(&[]);
        }
        return Err(set_err(GT_INVALID_ARG, "null pointer".to_string()));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Read `n` byte slices from parallel ptr/len arrays.
unsafe fn read_byte_slices<'a>(
    ptrs: *const *const u8,
    lens: *const usize,
    n: usize,
) -> Result<Vec<&'a [u8]>, i32> {
    if n > 0 && (ptrs.is_null() || lens.is_null()) {
        return Err(set_err(GT_INVALID_ARG, "null array pointer".to_string()));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = unsafe { *ptrs.add(i) };
        let l = unsafe { *lens.add(i) };
        out.push(unsafe { read_bytes(p, l) }?);
    }
    Ok(out)
}

unsafe fn read_path(ptr: *const u8, len: usize) -> Result<PathBuf, i32> {
    Ok(PathBuf::from(unsafe { read_str(ptr, len) }?))
}

fn eyre_code(err: &eyre::Report) -> (i32, String) {
    if err.downcast_ref::<std::io::Error>().is_some() {
        return (GT_IO, format!("{err:#}"));
    }
    (GT_LOAD, format!("{err:#}"))
}

// ---------------------------------------------------------------------------
// Process-global cache budget (ports bindings/cache.rs)
// ---------------------------------------------------------------------------

static MAX_CACHE_BYTES: Mutex<Option<usize>> =
    Mutex::new(Some(Tokenizer::DEFAULT_MAX_CACHE_BYTES));

fn apply_max_cache_bytes(mut tokenizer: Tokenizer) -> Tokenizer {
    let configured = *MAX_CACHE_BYTES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if configured != tokenizer.max_cache_bytes() {
        tokenizer.set_max_cache_bytes(configured);
    }
    tokenizer
}

fn apply_max_cache_bytes_sp(mut model: SentencePieceBPE) -> SentencePieceBPE {
    model.set_max_cache_bytes(
        *MAX_CACHE_BYTES
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
    );
    model
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Tokenizer handle: BPE (pooled workers + prototype + serial worker for
/// single encodes) or SentencePiece (model + serial encode state).
pub struct TokenizerHandle {
    bpe: Option<BpeState>,
    sp: Option<SpState>,
    vocab: Mutex<Option<VocabCache>>,
}

struct BpeState {
    proto: Tokenizer,
    workers: WorkerPool,
    single: Mutex<Tokenizer>,
}

struct SpState {
    model: SentencePieceBPE,
    single: Mutex<EncodeState>,
}

impl TokenizerHandle {
    fn bpe(tokenizer: Tokenizer) -> Self {
        let tokenizer = apply_max_cache_bytes(tokenizer);
        let single = tokenizer.fork();
        Self {
            bpe: Some(BpeState {
                proto: tokenizer,
                workers: WorkerPool::new(),
                single: Mutex::new(single),
            }),
            sp: None,
            vocab: Mutex::new(None),
        }
    }

    fn sp(model: SentencePieceBPE) -> Self {
        let model = apply_max_cache_bytes_sp(model);
        let state = EncodeState::with_budget(model.max_cache_bytes());
        Self {
            bpe: None,
            sp: Some(SpState {
                model,
                single: Mutex::new(state),
            }),
            vocab: Mutex::new(None),
        }
    }

    fn kind(&self) -> i32 {
        if self.bpe.is_some() {
            0
        } else {
            1
        }
    }
}

/// Source handle: file paths + document format, or owned memory buffers.
pub struct SourceHandle {
    data: SourceData,
}

enum SourceData {
    Files { paths: Vec<PathBuf>, format: DocFormat },
    Bytes { buffers: Vec<Vec<u8>>, format: DocFormat },
}

/// Compiled multi-pattern substring matcher (ports bindings/matcher.rs).
#[derive(Clone)]
pub struct MatcherHandle {
    automaton: aho_corasick::AhoCorasick,
    n_patterns: usize,
}

impl MatcherHandle {
    fn scan(&self, haystack: &[u8]) -> Vec<bool> {
        let mut seen = vec![false; self.n_patterns];
        let mut found = 0;
        for m in self.automaton.find_overlapping_iter(haystack) {
            let idx = m.pattern().as_usize();
            if !seen[idx] {
                seen[idx] = true;
                found += 1;
                if found == self.n_patterns {
                    break;
                }
            }
        }
        seen
    }

    fn scan_docs(&self, docs: &[&[u8]], parallel: bool) -> Vec<usize> {
        use rayon::prelude::*;
        let merge = |mut a: Vec<bool>, b: Vec<bool>| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x |= y);
            a
        };
        let seen = if parallel {
            docs.par_iter()
                .map(|d| self.scan(d))
                .reduce(|| vec![false; self.n_patterns], merge)
        } else {
            docs.iter()
                .map(|d| self.scan(d))
                .fold(vec![false; self.n_patterns], merge)
        };
        seen.iter()
            .enumerate()
            .filter_map(|(i, &s)| s.then_some(i))
            .collect()
    }
}

/// Row-assembly options for batch encode (ports WrapTruncate). `forbid`
/// owns a clone of the caller's matcher taken at `gt_wrap_opts_new` time
/// (null = none), so the caller may free its matcher immediately after
/// constructing the options without risking use-after-free.
pub struct WrapOptsHandle {
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    max_tokens: Option<usize>,
    truncate_left: bool,
    forbid: Option<MatcherHandle>,
}

/// Padded-matrix options for batch encode (ports PadTruncate).
pub struct PadOptsHandle {
    pad_id: u32,
    max_length: Option<usize>,
    pad_to_max_length: bool,
    truncate: bool,
    pad_left: bool,
    truncate_left: bool,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

/// Owned-bytes pretokenizer iterator (FastR50k scheme, like the Python one).
/// `pos` is behind a `Mutex` so concurrent `next` calls on one iterator are
/// serialized instead of `&mut`-aliasing (which would be UB).
pub struct PretokenIterHandle {
    bytes: Vec<u8>,
    pos: Mutex<usize>,
}

// ---------------------------------------------------------------------------
// Version (Phase 0 smoke test)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_version_major() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_version_minor() -> i32 {
    10
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_version_patch() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Memory + error inspection
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_last_error_len() -> usize {
    LAST_ERROR.with(|e| e.borrow().len())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_last_error_write(buf: *mut u8, len: usize) -> usize {
    // Null buffer yields 0 by design (no code to return; cannot report rc 1
    // through a usize return). Null+len>0 is treated the same as null+0.
    if buf.is_null() {
        return 0;
    }
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        let n = e.len().min(len);
        unsafe { std::ptr::copy_nonoverlapping(e.as_ptr(), buf, n) };
        n
    })
}

/// Matched pattern indices from the last code-6 return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_special_hit_count() -> usize {
    SPECIAL_HITS.with(|h| h.borrow().len())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_special_hit_get(index: usize, out: *mut usize) -> i32 {
    // Bypass `guard`: its success path clears SPECIAL_HITS, which would erase
    // the hit list after the first successful read. Hit reads must preserve
    // the list until the next non-6 guarded call clears it.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out.is_null() {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let v = SPECIAL_HITS.with(|h| h.borrow().get(index).copied());
        match v {
            Some(i) => {
                unsafe { *out = i };
                GT_OK
            }
            None => set_err(GT_INVALID_ARG, "special hit index out of range".to_string()),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_err(GT_INTERNAL, "internal panic".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 0 = BPE, 1 = SentencePiece, -1 = null handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_kind(handle: *const TokenizerHandle) -> i32 {
    let r = std::panic::catch_unwind(|| {
        if handle.is_null() {
            return -1;
        }
        unsafe { (*handle).kind() }
    });
    r.unwrap_or(-1)
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

fn load_boxed(handle: TokenizerHandle, out: *mut *mut TokenizerHandle) -> i32 {
    if out.is_null() {
        return set_err(GT_INVALID_ARG, "null out pointer".to_string());
    }
    unsafe { *out = Box::into_raw(Box::new(handle)) };
    GT_OK
}

/// Load `tokenizer.json` from memory; byte_fallback dispatches to SP.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_load_hf_json(
    data_ptr: *const u8,
    data_len: usize,
    out: *mut *mut TokenizerHandle,
) -> i32 {
    guard(|| {
        let data = match unsafe { read_bytes(data_ptr, data_len) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        match load_hf_slice(data) {
            Ok(HfTokenizer::Bpe(t)) => load_boxed(TokenizerHandle::bpe(t), out),
            Ok(HfTokenizer::SentencePiece(m)) => load_boxed(TokenizerHandle::sp(m), out),
            Err(e) => {
                let (c, m) = eyre_code(&e);
                set_err(c, m)
            }
        }
    })
}

/// Load a `.tiktoken` rank file. `specials_keys/lens` + `specials_vals`
/// carry `n_specials` (content, id) pairs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_bpe_from_tiktoken(
    path_ptr: *const u8,
    path_len: usize,
    pretokenizer_ptr: *const u8,
    pretokenizer_len: usize,
    specials_keys: *const *const u8,
    specials_keys_lens: *const usize,
    specials_vals: *const u32,
    n_specials: usize,
    out: *mut *mut TokenizerHandle,
) -> i32 {
    guard(|| {
        let path = match unsafe { read_path(path_ptr, path_len) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let name = match unsafe { read_str(pretokenizer_ptr, pretokenizer_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let scheme = match pretokenize::PretokenizerType::from_name(name) {
            Some(s) => s,
            None => {
                return set_err(
                    GT_INVALID_ARG,
                    format!(
                        "unknown pretokenizer scheme {name:?}; expected one of {}",
                        pretokenize::PretokenizerType::NAMES.join(", ")
                    ),
                );
            }
        };
        let mut specials = Vec::with_capacity(n_specials);
        if n_specials > 0 {
            if specials_keys.is_null() || specials_keys_lens.is_null() || specials_vals.is_null() {
                return set_err(GT_INVALID_ARG, "null specials array".to_string());
            }
            for i in 0..n_specials {
                let k = unsafe { *specials_keys.add(i) };
                let kl = unsafe { *specials_keys_lens.add(i) };
                let v = unsafe { *specials_vals.add(i) };
                match unsafe { read_str(k, kl) } {
                    Ok(s) => specials.push((s.to_string(), v)),
                    Err(c) => return c,
                }
            }
        }
        match crate::load_tokenizer::tiktoken::load_tiktoken(&path, scheme, specials) {
            Ok(t) => load_boxed(TokenizerHandle::bpe(t), out),
            Err(e) => {
                let (c, m) = eyre_code(&e);
                set_err(c, m)
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_bpe_from_hf(
    path_ptr: *const u8,
    path_len: usize,
    out: *mut *mut TokenizerHandle,
) -> i32 {
    guard(|| {
        let path = match unsafe { read_path(path_ptr, path_len) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        match load_hf_bpe(&path) {
            Ok(t) => load_boxed(TokenizerHandle::bpe(t), out),
            Err(e) => {
                let (c, m) = eyre_code(&e);
                set_err(c, m)
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_sp_from_hf(
    path_ptr: *const u8,
    path_len: usize,
    out: *mut *mut TokenizerHandle,
) -> i32 {
    guard(|| {
        let path = match unsafe { read_path(path_ptr, path_len) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        match load_hf_sentencepiece(&path) {
            Ok(m) => load_boxed(TokenizerHandle::sp(m), out),
            Err(e) => {
                let (c, m) = eyre_code(&e);
                set_err(c, m)
            }
        }
    })
}

/// Build a tokenizer from in-memory `tokenizer.json` produced by
/// `gt_sentencepiece_model_to_json` (accepts either backend kind, like
/// `load_hf_json`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_sentencepiece_from_bytes(
    json_ptr: *const u8,
    json_len: usize,
    out: *mut *mut TokenizerHandle,
) -> i32 {
    guard(|| {
        let data = match unsafe { read_bytes(json_ptr, json_len) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        match load_hf_slice(data) {
            Ok(HfTokenizer::Bpe(t)) => load_boxed(TokenizerHandle::bpe(t), out),
            Ok(HfTokenizer::SentencePiece(m)) => load_boxed(TokenizerHandle::sp(m), out),
            Err(e) => {
                let (c, m) = eyre_code(&e);
                set_err(c, m)
            }
        }
    })
}

/// Convert a raw sentencepiece `.model` to `tokenizer.json` (BPE +
/// byte-fallback only; error 5 otherwise).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_sentencepiece_model_to_json(
    model_ptr: *const u8,
    model_len: usize,
    out_json: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        let data = match unsafe { read_bytes(model_ptr, model_len) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        match sp_model::sentencepiece_to_tokenizer_json(data) {
            Ok(json) => out_vec(json.into_bytes(), out_json, out_len),
            Err(m) => set_err(GT_LOAD, m),
        }
    })
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

fn lock_single<'a>(
    mutex: &'a Mutex<Tokenizer>,
) -> Result<std::sync::MutexGuard<'a, Tokenizer>, i32> {
    Ok(mutex.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Single encode: input is arbitrary bytes for BPE, UTF-8 text for SP.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_encode(
    handle: *const TokenizerHandle,
    input_ptr: *const u8,
    input_len: usize,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null tokenizer handle".to_string());
        }
        let input = match unsafe { read_bytes(input_ptr, input_len) } {
            Ok(b) => b,
            Err(c) => return c,
        };
        let h = unsafe { &*handle };
        if let Some(st) = h.bpe.as_ref() {
            let mut tok = match lock_single(&st.single) {
                Ok(g) => g,
                Err(c) => return c,
            };
            let mut ids = Vec::new();
            let mut lens = Vec::new();
            crate::batch::encode_into(&mut tok, input, &mut ids, &mut lens);
            out_vec(ids, out_ids, out_len)
        } else if let Some(st) = h.sp.as_ref() {
            let text = match std::str::from_utf8(input) {
                Ok(s) => s,
                Err(_) => return set_err(GT_UTF8, "input is not valid UTF-8".to_string()),
            };
            let mut state = st.single.lock().unwrap_or_else(|e| e.into_inner());
            let mut ids = Vec::new();
            st.model.encode_raw_cb(&mut state, text, &mut |tokens: &[TokenId]| {
                ids.extend(tokens.iter().map(|&t| u32::from(t)));
            });
            out_vec(ids, out_ids, out_len)
        } else {
            set_err(GT_INTERNAL, "empty tokenizer handle".to_string())
        }
    })
}

/// SentencePiece encode of already-normalized text (mirrors
/// `encode_no_normalize`); error 1 on a BPE handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_sp_encode_no_normalize(
    handle: *const TokenizerHandle,
    str_ptr: *const u8,
    str_len: usize,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null tokenizer handle".to_string());
        }
        let text = match unsafe { read_str(str_ptr, str_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let h = unsafe { &*handle };
        match h.sp.as_ref() {
            Some(st) => {
                let mut state = st.single.lock().unwrap_or_else(|e| e.into_inner());
                let mut ids = Vec::new();
                st.model
                    .encode_normalized_cb(&mut state, text, &mut |tokens: &[TokenId]| {
                        ids.extend(tokens.iter().map(|&t| u32::from(t)));
                    });
                out_vec(ids, out_ids, out_len)
            }
            None => set_err(
                GT_INVALID_ARG,
                "gt_sp_encode_no_normalize requires a SentencePiece handle".to_string(),
            ),
        }
    })
}

struct BatchOut {
    flat: Vec<u32>,
    counts: Vec<i64>,
}

unsafe fn encode_batch_inner(
    h: &TokenizerHandle,
    docs: &[&[u8]],
    parallel: bool,
    wrap: Option<&WrapOptsHandle>,
) -> Result<BatchOut, i32> {
    if let Some(st) = h.bpe.as_ref() {
        if let Some(w) = wrap {
            if let Some(m) = w.forbid.as_ref() {
                let hits = m.scan_docs(docs, parallel);
                if !hits.is_empty() {
                    SPECIAL_HITS.with(|slot| *slot.borrow_mut() = hits.clone());
                    return Err(set_err(
                        GT_SPECIAL,
                        format!("forbidden special token found: {hits:?}"),
                    ));
                }
            }
        }
        let (flat, counts) = if parallel {
            encode_docs_ragged(&st.workers, &st.proto, docs)
        } else {
            crate::batch::encode_docs_ragged_serial(&st.workers, &st.proto, docs)
        };
        Ok(BatchOut { flat, counts })
    } else if let Some(st) = h.sp.as_ref() {
        if let Some(w) = wrap {
            if let Some(m) = w.forbid.as_ref() {
                let hits = m.scan_docs(docs, parallel);
                if !hits.is_empty() {
                    SPECIAL_HITS.with(|slot| *slot.borrow_mut() = hits.clone());
                    return Err(set_err(
                        GT_SPECIAL,
                        format!("forbidden special token found: {hits:?}"),
                    ));
                }
            }
        }
        let mut texts = Vec::with_capacity(docs.len());
        for d in docs {
            match std::str::from_utf8(d) {
                Ok(s) => texts.push(s),
                Err(_) => {
                    return Err(set_err(GT_UTF8, "input is not valid UTF-8".to_string()));
                }
            }
        }
        let (flat, counts) = if parallel {
            sp_encode_docs_ragged(&st.model, &texts)
        } else {
            crate::batch::sp_encode_docs_ragged_serial(&st.model, &texts)
        };
        Ok(BatchOut { flat, counts })
    } else {
        Err(set_err(GT_INTERNAL, "empty tokenizer handle".to_string()))
    }
}

/// Apply wrap options (prefix/suffix/max_tokens/truncate_left) to a ragged
/// batch — ports `encode_batch_pylist` row assembly.
fn apply_wrap(
    flat: &[u32],
    counts: &[i64],
    wrap: Option<&WrapOptsHandle>,
) -> Result<(Vec<u32>, Vec<i64>), (i32, String)> {
    let (prefix, suffix, cap, truncate_left) = match wrap {
        Some(w) => (
            w.prefix.as_slice(),
            w.suffix.as_slice(),
            w.max_tokens.unwrap_or(usize::MAX),
            w.truncate_left,
        ),
        None => (&[][..], &[][..], usize::MAX, false),
    };
    let wrap_on = !prefix.is_empty() || !suffix.is_empty();
    let mut out_flat = Vec::with_capacity(flat.len() + counts.len() * (prefix.len() + suffix.len()));
    let mut out_counts = Vec::with_capacity(counts.len());
    let mut pos = 0usize;
    for &n in counts {
        if n < 0 {
            return Err((GT_INTERNAL, format!("negative row count {n}")));
        }
        let n = n as usize;
        let end = pos.checked_add(n).ok_or_else(|| {
            (
                GT_INTERNAL,
                "row offset overflow".to_string(),
            )
        })?;
        if end > flat.len() {
            return Err((GT_INTERNAL, "row count exceeds flat buffer".to_string()));
        }
        let tokens = &flat[pos..end];
        pos = end;
        let kept = n.min(cap);
        let tokens = if truncate_left {
            &tokens[n - kept..]
        } else {
            &tokens[..kept]
        };
        if wrap_on {
            let total = kept
                .checked_add(prefix.len())
                .and_then(|v| v.checked_add(suffix.len()))
                .ok_or_else(|| (GT_INTERNAL, "wrapped row length overflow".to_string()))?;
            if total > i64::MAX as usize {
                return Err((GT_INTERNAL, "wrapped row length overflow".to_string()));
            }
            out_flat.extend_from_slice(prefix);
            out_flat.extend_from_slice(tokens);
            out_flat.extend_from_slice(suffix);
            out_counts.push(total as i64);
        } else {
            out_flat.extend_from_slice(tokens);
            if kept > i64::MAX as usize {
                return Err((GT_INTERNAL, "row length overflow".to_string()));
            }
            out_counts.push(kept as i64);
        }
    }
    Ok((out_flat, out_counts))
}

/// Batch encode whole documents (each input slice is one document).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_encode_batch(
    handle: *const TokenizerHandle,
    inputs_ptr_array: *const *const u8,
    inputs_len_array: *const usize,
    n: usize,
    parallel: i32,
    wrap_or_null: *const WrapOptsHandle,
    out_flat: *mut *mut u32,
    out_flat_len: *mut usize,
    out_counts: *mut *mut i64,
    out_nrows: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null tokenizer handle".to_string());
        }
        if out_flat.is_null() || out_flat_len.is_null() || out_counts.is_null() || out_nrows.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let docs = match unsafe { read_byte_slices(inputs_ptr_array, inputs_len_array, n) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        let wrap = if wrap_or_null.is_null() {
            None
        } else {
            Some(unsafe { &*wrap_or_null })
        };
        let h = unsafe { &*handle };
        let out = match unsafe { encode_batch_inner(h, &docs, parallel != 0, wrap) } {
            Ok(o) => o,
            Err(c) => return c,
        };
        let (flat, counts) = match apply_wrap(&out.flat, &out.counts, wrap) {
            Ok(t) => t,
            Err((c, m)) => return set_err(c, m),
        };
        let c = out_vec(flat, out_flat, out_flat_len);
        if c != GT_OK {
            return c;
        }
        let c = out_vec(counts, out_counts, out_nrows);
        if c != GT_OK {
            // Reclaim the first buffer: the caller never received a success
            // code, so it has no ownership to free.
            unsafe {
                free_published(*out_flat as *mut std::ffi::c_void);
                *out_flat = std::ptr::null_mut();
                *out_flat_len = 0;
            }
            return c;
        }
        GT_OK
    })
}

/// Batch encode with the BytesSource split-during-encode semantics: each
/// buffer is split into documents on the separator (null separator = each
/// buffer is one document). The SentencePiece backend requires a valid
/// UTF-8 separator (error 2 otherwise).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_encode_regions(
    handle: *const TokenizerHandle,
    buffers_ptr_array: *const *const u8,
    buffers_len_array: *const usize,
    n: usize,
    sep_ptr: *const u8,
    sep_len: usize,
    has_sep: i32,
    parallel: i32,
    wrap_or_null: *const WrapOptsHandle,
    out_flat: *mut *mut u32,
    out_flat_len: *mut usize,
    out_counts: *mut *mut i64,
    out_nrows: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null tokenizer handle".to_string());
        }
        if out_flat.is_null() || out_flat_len.is_null() || out_counts.is_null() || out_nrows.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let bufs = match unsafe { read_byte_slices(buffers_ptr_array, buffers_len_array, n) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        let sep: Option<Vec<u8>> = if has_sep != 0 {
            match unsafe { read_bytes(sep_ptr, sep_len) } {
                Ok(s) => Some(s.to_vec()),
                Err(c) => return c,
            }
        } else {
            None
        };
        let format = DocFormat::Text { separator: sep };
        let wrap = if wrap_or_null.is_null() {
            None
        } else {
            Some(unsafe { &*wrap_or_null })
        };
        let h = unsafe { &*handle };
        let out = match encode_regions_inner(h, &bufs, &format, parallel != 0) {
            Ok(o) => o,
            Err(c) => return c,
        };
        // Fused forbid scan applies to the raw buffers (mirrors the
        // BytesSource path, where the scan runs before splitting).
        if let Some(w) = wrap {
            if let Some(m) = w.forbid.as_ref() {
                let docs: Vec<&[u8]> = bufs;
                let hits = m.scan_docs(&docs, parallel != 0);
                if !hits.is_empty() {
                    SPECIAL_HITS.with(|slot| *slot.borrow_mut() = hits.clone());
                    return set_err(GT_SPECIAL, format!("forbidden special token found: {hits:?}"));
                }
            }
        }
        let (flat, counts) = match apply_wrap(&out.flat, &out.counts, wrap) {
            Ok(t) => t,
            Err((c, m)) => return set_err(c, m),
        };
        let c = out_vec(flat, out_flat, out_flat_len);
        if c != GT_OK {
            return c;
        }
        let c = out_vec(counts, out_counts, out_nrows);
        if c != GT_OK {
            unsafe {
                free_published(*out_flat as *mut std::ffi::c_void);
                *out_flat = std::ptr::null_mut();
                *out_flat_len = 0;
            }
            return c;
        }
        GT_OK
    })
}

fn encode_regions_inner(
    h: &TokenizerHandle,
    regions: &[&[u8]],
    format: &DocFormat,
    parallel: bool,
) -> Result<BatchOut, i32> {
    if let Some(st) = h.bpe.as_ref() {
        let (flat, counts) = if parallel {
            crate::batch::encode_files_docs(&st.workers, &st.proto, regions, format)
        } else {
            crate::batch::encode_files_docs_serial(&st.workers, &st.proto, regions, format)
        };
        Ok(BatchOut { flat, counts })
    } else if let Some(st) = h.sp.as_ref() {
        if let DocFormat::Text {
            separator: Some(sep),
        } = format
        {
            if std::str::from_utf8(sep).is_err() {
                return Err(set_err(
                    GT_UTF8,
                    "the SentencePiece backend requires a separator that is valid UTF-8"
                        .to_string(),
                ));
            }
        }
        // File/region bytes are trusted UTF-8 (same contract as upstream).
        for r in regions {
            if std::str::from_utf8(r).is_err() {
                return Err(set_err(GT_UTF8, "input is not valid UTF-8".to_string()));
            }
        }
        let (flat, counts) = if parallel {
            crate::batch::sp_encode_files_docs(&st.model, regions, format)
        } else {
            crate::batch::sp_encode_files_docs_serial(&st.model, regions, format)
        };
        Ok(BatchOut { flat, counts })
    } else {
        Err(set_err(GT_INTERNAL, "empty tokenizer handle".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Padded batch (ports pad_truncate_matrix)
// ---------------------------------------------------------------------------

fn pad_truncate_matrix(
    flat: &[u32],
    counts: &[i64],
    opts: &PadOptsHandle,
) -> Result<(Vec<u32>, Vec<i64>, usize), (i32, String)> {
    let extra = opts.prefix.len() + opts.suffix.len();
    let cap = if opts.truncate {
        let max = opts.max_length.ok_or_else(|| {
            (
                GT_INVALID_ARG,
                "truncate requires max_length".to_string(),
            )
        })?;
        max.checked_sub(extra).ok_or_else(|| {
            (
                GT_INVALID_ARG,
                format!(
                    "max_length={max} leaves no room for the {extra} special tokens added per sequence"
                ),
            )
        })?
    } else {
        usize::MAX
    };
    let mut offsets = Vec::with_capacity(counts.len());
    let mut lengths = Vec::with_capacity(counts.len());
    let mut longest = 0usize;
    let mut pos = 0usize;
    for &count in counts {
        if count < 0 {
            return Err((GT_INTERNAL, format!("negative row count {count}")));
        }
        let count = count as usize;
        let kept = count.min(cap);
        offsets.push(pos);
        pos = pos.checked_add(count).ok_or_else(|| {
            (
                GT_INTERNAL,
                "row offset overflow".to_string(),
            )
        })?;
        if pos > flat.len() {
            return Err((GT_INTERNAL, "row count exceeds flat buffer".to_string()));
        }
        let total = kept.checked_add(extra).ok_or_else(|| {
            (
                GT_INTERNAL,
                "padded row length overflow".to_string(),
            )
        })?;
        if total > i64::MAX as usize {
            return Err((GT_INTERNAL, "padded row length overflow".to_string()));
        }
        lengths.push(total as i64);
        longest = longest.max(total);
    }
    let width = if opts.pad_to_max_length {
        let max = opts.max_length.ok_or_else(|| {
            (
                GT_INVALID_ARG,
                "pad_to_max_length requires max_length".to_string(),
            )
        })?;
        if longest > max {
            return Err((
                GT_INVALID_ARG,
                format!(
                    "a sequence is {longest} ids long but padding to max_length={max} was requested \
                     without truncation; enable truncation or raise max_length"
                ),
            ));
        }
        max
    } else {
        longest
    };
    let n_elems = counts.len().checked_mul(width).ok_or_else(|| {
        (
            GT_INTERNAL,
            "padded matrix size overflow".to_string(),
        )
    })?;
    let mut out = vec![opts.pad_id; n_elems];
    if width > 0 {
        for (i, row) in out.chunks_mut(width).enumerate() {
            // Non-negativity was validated in the offsets pass above; re-check
            // defensively so a future caller change cannot turn this into a
            // wrap/panic.
            if counts[i] < 0 {
                return Err((GT_INTERNAL, format!("negative row count {}", counts[i])));
            }
            let count = counts[i] as usize;
            let kept = count.min(cap);
            let end = offsets[i].checked_add(count).ok_or_else(|| {
                (
                    GT_INTERNAL,
                    "row offset overflow".to_string(),
                )
            })?;
            if end > flat.len() {
                return Err((GT_INTERNAL, "row count exceeds flat buffer".to_string()));
            }
            let tokens = &flat[offsets[i]..end];
            let tokens = if opts.truncate_left {
                &tokens[count - kept..]
            } else {
                &tokens[..kept]
            };
            let len = kept + extra;
            // pad_left shifts the row right; pad_to_max_length uses full width.
            let start = if opts.pad_left { width - len } else { 0 };
            row[start..start + opts.prefix.len()].copy_from_slice(&opts.prefix);
            row[start + opts.prefix.len()..start + opts.prefix.len() + kept]
                .copy_from_slice(tokens);
            if !opts.suffix.is_empty() {
                row[start + len - opts.suffix.len()..start + len].copy_from_slice(&opts.suffix);
            }
        }
    }
    Ok((out, lengths, width))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_encode_batch_padded(
    handle: *const TokenizerHandle,
    inputs_ptr_array: *const *const u8,
    inputs_len_array: *const usize,
    n: usize,
    parallel: i32,
    pad_opts: *const PadOptsHandle,
    out_matrix: *mut *mut u32,
    out_nrows: *mut usize,
    out_ncols: *mut usize,
    out_lengths: *mut *mut i64,
) -> i32 {
    guard(|| {
        if handle.is_null() || pad_opts.is_null() {
            return set_err(GT_INVALID_ARG, "null handle or options".to_string());
        }
        if out_matrix.is_null() || out_nrows.is_null() || out_ncols.is_null() || out_lengths.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let docs = match unsafe { read_byte_slices(inputs_ptr_array, inputs_len_array, n) } {
            Ok(d) => d,
            Err(c) => return c,
        };
        let h = unsafe { &*handle };
        let opts = unsafe { &*pad_opts };
        // Padded assembly never carries a forbid scan (matches upstream,
        // where only the list path takes WrapTruncate).
        let out = match unsafe { encode_batch_inner(h, &docs, parallel != 0, None) } {
            Ok(o) => o,
            Err(c) => return c,
        };
        let (matrix, lengths, width) = match pad_truncate_matrix(&out.flat, &out.counts, opts) {
            Ok(t) => t,
            Err((c, m)) => return set_err(c, m),
        };
        let mut _matrix_len = 0usize;
        let c = out_vec(matrix, out_matrix, &mut _matrix_len);
        if c != GT_OK {
            return c;
        }
        // out_vec wrote the element count into a dummy; publish rows/cols.
        unsafe {
            *out_nrows = lengths.len();
            *out_ncols = width;
        }
        let mut _lengths_len = 0usize;
        let c = out_vec(lengths, out_lengths, &mut _lengths_len);
        if c != GT_OK {
            unsafe {
                free_published(*out_matrix as *mut std::ffi::c_void);
                *out_matrix = std::ptr::null_mut();
                *out_nrows = 0;
                *out_ncols = 0;
            }
            return c;
        }
        GT_OK
    })
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

fn encode_source_inner(
    h: &TokenizerHandle,
    source: &SourceHandle,
    parallel: bool,
) -> Result<BatchOut, i32> {
    match &source.data {
        SourceData::Files { paths, format } => {
            if let DocFormat::Parquet { column } = format {
                // Parquet rows materialize as owned documents, then encode
                // through the whole-document path (mirrors upstream).
                let docs = load_parquet_docs(paths, column, parallel)?;
                let refs: Vec<&[u8]> = docs.iter().map(|d| d.as_slice()).collect();
                if let Some(st) = h.bpe.as_ref() {
                    let (flat, counts) = if parallel {
                        encode_docs_ragged(&st.workers, &st.proto, &refs)
                    } else {
                        crate::batch::encode_docs_ragged_serial(&st.workers, &st.proto, &refs)
                    };
                    Ok(BatchOut { flat, counts })
                } else if let Some(st) = h.sp.as_ref() {
                    let mut texts = Vec::with_capacity(refs.len());
                    for d in &refs {
                        match std::str::from_utf8(d) {
                            Ok(s) => texts.push(s),
                            Err(_) => {
                                return Err(set_err(
                                    GT_UTF8,
                                    "input is not valid UTF-8".to_string(),
                                ));
                            }
                        }
                    }
                    let (flat, counts) = if parallel {
                        sp_encode_docs_ragged(&st.model, &texts)
                    } else {
                        crate::batch::sp_encode_docs_ragged_serial(&st.model, &texts)
                    };
                    Ok(BatchOut { flat, counts })
                } else {
                    Err(set_err(GT_INTERNAL, "empty tokenizer handle".to_string()))
                }
            } else {
                let files = load_files_vec(paths, parallel)?;
                let refs: Vec<&[u8]> = files.iter().map(|f| f.as_bytes()).collect();
                encode_regions_inner(h, &refs, format, parallel)
            }
        }
        SourceData::Bytes { buffers, format } => {
            let refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
            encode_regions_inner(h, &refs, format, parallel)
        }
    }
}

fn load_files_vec(
    paths: &[PathBuf],
    parallel: bool,
) -> Result<Vec<crate::input::file_source::LoadedFile>, i32> {
    use rayon::prelude::*;
    if parallel {
        paths
            .par_iter()
            .map(|p| load_file(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<Vec<_>, String>>()
            .map_err(|m| set_err(GT_IO, m))
    } else {
        paths
            .iter()
            .map(|p| load_file(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<Vec<_>, String>>()
            .map_err(|m| set_err(GT_IO, m))
    }
}

fn load_parquet_docs(
    paths: &[PathBuf],
    column: &str,
    parallel: bool,
) -> Result<Vec<Vec<u8>>, i32> {
    use crate::input::parquet::read_docs;
    use rayon::prelude::*;
    if parallel {
        paths
            .par_iter()
            .map(|p| {
                read_docs(p, column, true)
                    .map_err(|e| format!("{}: {e}", p.display()))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|v| v.into_iter().flatten().collect())
            .map_err(|m| set_err(GT_IO, m))
    } else {
        paths
            .iter()
            .map(|p| {
                read_docs(p, column, false)
                    .map_err(|e| format!("{}: {e}", p.display()))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|v| v.into_iter().flatten().collect())
            .map_err(|m| set_err(GT_IO, m))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_encode_files(
    handle: *const TokenizerHandle,
    source: *const SourceHandle,
    parallel: i32,
    out_flat: *mut *mut u32,
    out_flat_len: *mut usize,
    out_counts: *mut *mut i64,
    out_nrows: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() || source.is_null() {
            return set_err(GT_INVALID_ARG, "null handle or source".to_string());
        }
        if out_flat.is_null() || out_flat_len.is_null() || out_counts.is_null() || out_nrows.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let h = unsafe { &*handle };
        let s = unsafe { &*source };
        let out = match encode_source_inner(h, s, parallel != 0) {
            Ok(o) => o,
            Err(c) => return c,
        };
        let c = out_vec(out.flat, out_flat, out_flat_len);
        if c != GT_OK {
            return c;
        }
        let c = out_vec(out.counts, out_counts, out_nrows);
        if c != GT_OK {
            unsafe {
                free_published(*out_flat as *mut std::ffi::c_void);
                *out_flat = std::ptr::null_mut();
                *out_flat_len = 0;
            }
            return c;
        }
        GT_OK
    })
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_decode(
    handle: *const TokenizerHandle,
    ids_ptr: *const u32,
    ids_len: usize,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null tokenizer handle".to_string());
        }
        let ids = match unsafe { read_u32s(ids_ptr, ids_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        // SAFETY: TokenId is #[repr(transparent)] over u32.
        let ids: &[TokenId] =
            unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len()) };
        let h = unsafe { &*handle };
        if let Some(st) = h.bpe.as_ref() {
            let bytes: Vec<u8> = st.proto.decode(ids).collect();
            out_vec(bytes, out_bytes, out_len)
        } else if let Some(st) = h.sp.as_ref() {
            out_vec(st.model.decode(ids), out_bytes, out_len)
        } else {
            set_err(GT_INTERNAL, "empty tokenizer handle".to_string())
        }
    })
}

// ---------------------------------------------------------------------------
// Vocab / merges / cache stats (ordinal iteration with memoized tables)
// ---------------------------------------------------------------------------

struct VocabCache {
    vocab: Vec<(u32, Vec<u8>)>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

fn build_vocab_cache(h: &TokenizerHandle) -> VocabCache {
    if let Some(st) = h.bpe.as_ref() {
        VocabCache {
            vocab: st
                .proto
                .vocab_entries()
                .map(|(id, b)| (id, b.to_vec()))
                .collect(),
            merges: st
                .proto
                .merge_entries()
                .into_iter()
                .map(|(a, b)| (a.to_vec(), b.to_vec()))
                .collect(),
        }
    } else if let Some(st) = h.sp.as_ref() {
        VocabCache {
            vocab: st
                .model
                .vocab_entries()
                .map(|(id, b)| (id, b.to_vec()))
                .collect(),
            merges: st
                .model
                .merge_entries()
                .into_iter()
                .map(|(a, b)| (a.to_vec(), b.to_vec()))
                .collect(),
        }
    } else {
        VocabCache {
            vocab: Vec::new(),
            merges: Vec::new(),
        }
    }
}

/// Lazily-built, memoized per handle: vocab/merge tables are built once
/// (building sorts ~50k merges) and then borrowed per call; only the single
/// requested entry's bytes are copied out.
fn vocab_cache(h: &TokenizerHandle) -> std::sync::MutexGuard<'_, Option<VocabCache>> {
    let mut slot = h.vocab.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        *slot = Some(build_vocab_cache(h));
    }
    slot
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_vocab_size(handle: *const TokenizerHandle) -> usize {
    // Cannot use guard (returns i32); still must not let panics cross FFI.
    let r = std::panic::catch_unwind(|| {
        if handle.is_null() {
            return 0;
        }
        let h = unsafe { &*handle };
        if let Some(st) = h.bpe.as_ref() {
            st.proto.vocab_size()
        } else if let Some(st) = h.sp.as_ref() {
            st.model.vocab_size()
        } else {
            0
        }
    });
    r.unwrap_or(0)
}

/// Ordinal `index` over ID-ordered vocab entries (empty ids skipped, like
/// upstream). `out_id` gets the token id, `out_bytes`/`out_len` its bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_vocab_entry(
    handle: *const TokenizerHandle,
    index: usize,
    out_id: *mut u32,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() || out_id.is_null() {
            return set_err(GT_INVALID_ARG, "null handle or out pointer".to_string());
        }
        let h = unsafe { &*handle };
        let cache = vocab_cache(h);
        match cache.as_ref().unwrap().vocab.get(index) {
            Some((id, bytes)) => {
                unsafe { *out_id = *id };
                out_vec(bytes.clone(), out_bytes, out_len)
            }
            None => set_err(GT_INVALID_ARG, "vocab index out of range".to_string()),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_merge_count(handle: *const TokenizerHandle) -> usize {
    // Builds + sorts the vocab cache; lock/malloc can panic — contain it.
    let r = std::panic::catch_unwind(|| {
        if handle.is_null() {
            return 0;
        }
        vocab_cache(unsafe { &*handle }).as_ref().unwrap().merges.len()
    });
    r.unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_merge_entry(
    handle: *const TokenizerHandle,
    index: usize,
    out_left_ptr: *mut *mut u8,
    out_left_len: *mut usize,
    out_right_ptr: *mut *mut u8,
    out_right_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null handle".to_string());
        }
        if out_left_ptr.is_null()
            || out_left_len.is_null()
            || out_right_ptr.is_null()
            || out_right_len.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let h = unsafe { &*handle };
        let cache = vocab_cache(h);
        match cache.as_ref().unwrap().merges.get(index) {
            Some((left, right)) => {
                let c = out_vec(left.clone(), out_left_ptr, out_left_len);
                if c != GT_OK {
                    return c;
                }
                let c = out_vec(right.clone(), out_right_ptr, out_right_len);
                if c != GT_OK {
                    unsafe {
                        free_published(*out_left_ptr as *mut std::ffi::c_void);
                        *out_left_ptr = std::ptr::null_mut();
                        *out_left_len = 0;
                    }
                    return c;
                }
                GT_OK
            }
            None => set_err(GT_INVALID_ARG, "merge index out of range".to_string()),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_tokenizer_cache_entries(handle: *const TokenizerHandle) -> usize {
    let r = std::panic::catch_unwind(|| {
        if handle.is_null() {
            return 0;
        }
        let h = unsafe { &*handle };
        if let Some(st) = h.bpe.as_ref() {
            st.single.lock().unwrap_or_else(|e| e.into_inner()).cache_entries()
        } else if let Some(st) = h.sp.as_ref() {
            st.single.lock().unwrap_or_else(|e| e.into_inner()).cache_size()
        } else {
            0
        }
    });
    r.unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

unsafe fn read_paths(
    ptrs: *const *const u8,
    lens: *const usize,
    n: usize,
) -> Result<Vec<PathBuf>, i32> {
    let mut out = Vec::with_capacity(n);
    for s in unsafe { read_byte_slices(ptrs, lens, n) }? {
        match std::str::from_utf8(s) {
            Ok(p) => out.push(PathBuf::from(p)),
            Err(_) => return Err(set_err(GT_UTF8, "path is not valid UTF-8".to_string())),
        }
    }
    Ok(out)
}

fn box_source(data: SourceData, out: *mut *mut SourceHandle) -> i32 {
    if out.is_null() {
        return set_err(GT_INVALID_ARG, "null out pointer".to_string());
    }
    unsafe { *out = Box::into_raw(Box::new(SourceHandle { data })) };
    GT_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_text_source_new(
    paths_ptr_array: *const *const u8,
    paths_len_array: *const usize,
    n_paths: usize,
    sep_ptr: *const u8,
    sep_len: usize,
    has_sep: i32,
    out: *mut *mut SourceHandle,
) -> i32 {
    guard(|| {
        let paths = match unsafe { read_paths(paths_ptr_array, paths_len_array, n_paths) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let sep = if has_sep != 0 {
            match unsafe { read_bytes(sep_ptr, sep_len) } {
                Ok(s) => Some(s.to_vec()),
                Err(c) => return c,
            }
        } else {
            None
        };
        box_source(
            SourceData::Files {
                paths,
                format: DocFormat::Text { separator: sep },
            },
            out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_jsonl_source_new(
    paths_ptr_array: *const *const u8,
    paths_len_array: *const usize,
    n_paths: usize,
    field_ptr: *const u8,
    field_len: usize,
    out: *mut *mut SourceHandle,
) -> i32 {
    guard(|| {
        let paths = match unsafe { read_paths(paths_ptr_array, paths_len_array, n_paths) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let field = match unsafe { read_str(field_ptr, field_len) } {
            Ok(s) => s.to_string(),
            Err(c) => return c,
        };
        box_source(
            SourceData::Files {
                paths,
                format: DocFormat::Jsonl { field },
            },
            out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_parquet_source_new(
    paths_ptr_array: *const *const u8,
    paths_len_array: *const usize,
    n_paths: usize,
    column_ptr: *const u8,
    column_len: usize,
    out: *mut *mut SourceHandle,
) -> i32 {
    guard(|| {
        let paths = match unsafe { read_paths(paths_ptr_array, paths_len_array, n_paths) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let column = match unsafe { read_str(column_ptr, column_len) } {
            Ok(s) => s.to_string(),
            Err(c) => return c,
        };
        box_source(
            SourceData::Files {
                paths,
                format: DocFormat::Parquet { column },
            },
            out,
        )
    })
}

/// Bare paths with per-first-path default format (mirrors upstream bare-path
/// handling: `.jsonl` -> Jsonl[text], `.parquet` -> Parquet[text], else
/// whole-file text). Used by `EncodeFiles(path)` overloads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_path_source_new(
    paths_ptr_array: *const *const u8,
    paths_len_array: *const usize,
    n_paths: usize,
    out: *mut *mut SourceHandle,
) -> i32 {
    guard(|| {
        let paths = match unsafe { read_paths(paths_ptr_array, paths_len_array, n_paths) } {
            Ok(p) => p,
            Err(c) => return c,
        };
        let format = paths
            .first()
            .map(|p| detect_default_format(p))
            .unwrap_or(DocFormat::Text { separator: None });
        box_source(SourceData::Files { paths, format }, out)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_bytes_source_new(
    buffers_ptr_array: *const *const u8,
    buffers_len_array: *const usize,
    n_buffers: usize,
    sep_ptr: *const u8,
    sep_len: usize,
    has_sep: i32,
    out: *mut *mut SourceHandle,
) -> i32 {
    guard(|| {
        let bufs = match unsafe { read_byte_slices(buffers_ptr_array, buffers_len_array, n_buffers) }
        {
            Ok(b) => b,
            Err(c) => return c,
        };
        let sep = if has_sep != 0 {
            match unsafe { read_bytes(sep_ptr, sep_len) } {
                Ok(s) => Some(s.to_vec()),
                Err(c) => return c,
            }
        } else {
            None
        };
        box_source(
            SourceData::Bytes {
                buffers: bufs.into_iter().map(|b| b.to_vec()).collect(),
                format: DocFormat::Text { separator: sep },
            },
            out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_source_free(handle: *mut SourceHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ---------------------------------------------------------------------------
// Matcher + options
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_matcher_new(
    patterns_ptr_array: *const *const u8,
    patterns_len_array: *const usize,
    n: usize,
    out: *mut *mut MatcherHandle,
) -> i32 {
    guard(|| {
        let slices = match unsafe { read_byte_slices(patterns_ptr_array, patterns_len_array, n) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let mut patterns = Vec::with_capacity(n);
        for s in slices {
            match std::str::from_utf8(s) {
                Ok(p) => patterns.push(p.to_string()),
                Err(_) => return set_err(GT_UTF8, "pattern is not valid UTF-8".to_string()),
            }
        }
        match aho_corasick::AhoCorasick::new(&patterns) {
            Ok(automaton) => {
                if out.is_null() {
                    return set_err(GT_INVALID_ARG, "null out pointer".to_string());
                }
                unsafe {
                    *out = Box::into_raw(Box::new(MatcherHandle {
                        automaton,
                        n_patterns: patterns.len(),
                    }))
                };
                GT_OK
            }
            Err(e) => set_err(GT_INVALID_ARG, format!("failed to build matcher: {e}")),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_matcher_free(handle: *mut MatcherHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Sorted indices of the patterns occurring in `text`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_matcher_present(
    handle: *const MatcherHandle,
    text_ptr: *const u8,
    text_len: usize,
    out_indices: *mut *mut usize,
    out_n: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return set_err(GT_INVALID_ARG, "null matcher handle".to_string());
        }
        let text = match unsafe { read_bytes(text_ptr, text_len) } {
            Ok(b) => b,
            Err(c) => return c,
        };
        let m = unsafe { &*handle };
        let hits: Vec<usize> = m
            .scan(text)
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| s.then_some(i))
            .collect();
        out_vec(hits, out_indices, out_n)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_wrap_opts_new(
    prefix_ptr: *const u32,
    prefix_len: usize,
    suffix_ptr: *const u32,
    suffix_len: usize,
    has_max: i32,
    max_tokens: usize,
    truncate_left: i32,
    matcher_or_null: *const MatcherHandle,
    out: *mut *mut WrapOptsHandle,
) -> i32 {
    guard(|| {
        let prefix = match unsafe { read_u32s(prefix_ptr, prefix_len) } {
            Ok(s) => s.to_vec(),
            Err(c) => return c,
        };
        let suffix = match unsafe { read_u32s(suffix_ptr, suffix_len) } {
            Ok(s) => s.to_vec(),
            Err(c) => return c,
        };
        if out.is_null() {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        // Clone the matcher so the options own their forbid set: the caller
        // may free (or concurrently free) its handle right after this call
        // without creating a dangling pointer (H11).
        let forbid = if matcher_or_null.is_null() {
            None
        } else {
            Some(unsafe { (*matcher_or_null).clone() })
        };
        unsafe {
            *out = Box::into_raw(Box::new(WrapOptsHandle {
                prefix,
                suffix,
                max_tokens: if has_max != 0 { Some(max_tokens) } else { None },
                truncate_left: truncate_left != 0,
                forbid,
            }))
        };
        GT_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_wrap_opts_free(handle: *mut WrapOptsHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gt_pad_opts_new(
    pad_id: u32,
    has_max: i32,
    max_length: usize,
    pad_to_max_length: i32,
    truncate: i32,
    pad_left: i32,
    truncate_left: i32,
    prefix_ptr: *const u32,
    prefix_len: usize,
    suffix_ptr: *const u32,
    suffix_len: usize,
    out: *mut *mut PadOptsHandle,
) -> i32 {
    guard(|| {
        let prefix = match unsafe { read_u32s(prefix_ptr, prefix_len) } {
            Ok(s) => s.to_vec(),
            Err(c) => return c,
        };
        let suffix = match unsafe { read_u32s(suffix_ptr, suffix_len) } {
            Ok(s) => s.to_vec(),
            Err(c) => return c,
        };
        if out.is_null() {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        unsafe {
            *out = Box::into_raw(Box::new(PadOptsHandle {
                pad_id,
                max_length: if has_max != 0 { Some(max_length) } else { None },
                pad_to_max_length: pad_to_max_length != 0,
                truncate: truncate != 0,
                pad_left: pad_left != 0,
                truncate_left: truncate_left != 0,
                prefix,
                suffix,
            }))
        };
        GT_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_pad_opts_free(handle: *mut PadOptsHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ---------------------------------------------------------------------------
// train_bpe (returns vocab+merges data, no handle)
// ---------------------------------------------------------------------------

/// `input_tag`: 0 = memory blob, 1 = filesystem path, 2 = `gt_source_t*`.
/// `tie_breaking`: 0 = huggingface, 1 = raw_token_ids, 2 = assembled_bytes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gt_train_bpe(
    input_tag: i32,
    input_ptr: *const u8,
    input_len: usize,
    source_or_null: *const SourceHandle,
    vocab_size: usize,
    specials_ptr_array: *const *const u8,
    specials_len_array: *const usize,
    n_specials: usize,
    tie_breaking: i32,
    sep_ptr: *const u8,
    sep_len: usize,
    has_sep: i32,
    out_vocab_ids: *mut *mut u32,
    out_vocab_ptrs: *mut *mut *mut u8,
    out_vocab_lens: *mut *mut usize,
    out_vocab_n: *mut usize,
    out_merge_left: *mut *mut *mut u8,
    out_merge_left_lens: *mut *mut usize,
    out_merge_right: *mut *mut *mut u8,
    out_merge_right_lens: *mut *mut usize,
    out_merge_n: *mut usize,
) -> i32 {
    guard(|| {
        // Validate all out pointers before doing any work or mallocs (late
        // validation after the train would waste work and leak).
        if out_vocab_ids.is_null()
            || out_vocab_ptrs.is_null()
            || out_vocab_lens.is_null()
            || out_vocab_n.is_null()
            || out_merge_left.is_null()
            || out_merge_left_lens.is_null()
            || out_merge_right.is_null()
            || out_merge_right_lens.is_null()
            || out_merge_n.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        // Mirror upstream `assert!(vocab_size <= 2^32)` (bindings/train.rs:63)
        // as rc 1 instead of a panic/OOM. This also rejects wrapped C#
        // negatives (e.g. -1 -> usize::MAX).
        if (vocab_size as u64) > (1u64 << 32) {
            return set_err(
                GT_INVALID_ARG,
                format!("vocab_size {vocab_size} exceeds 2^32"),
            );
        }
        let tie = match tie_breaking {
            0 => crate::bpe_train::TieBreaking::HuggingFace,
            1 => crate::bpe_train::TieBreaking::RawTokenIds,
            2 => crate::bpe_train::TieBreaking::AssembledBytes,
            _ => {
                return set_err(
                    GT_INVALID_ARG,
                    format!("tie_breaking must be 0, 1, or 2, got {tie_breaking}"),
                );
            }
        };
        let mut specials = Vec::with_capacity(n_specials);
        if n_specials > 0 {
            match unsafe { read_byte_slices(specials_ptr_array, specials_len_array, n_specials) } {
                Ok(ss) => {
                    for s in ss {
                        match std::str::from_utf8(s) {
                            Ok(p) => specials.push(p.to_string()),
                            Err(_) => {
                                return set_err(
                                    GT_UTF8,
                                    "special token is not valid UTF-8".to_string(),
                                );
                            }
                        }
                    }
                }
                Err(c) => return c,
            }
        }
        let sep: &[u8] = if has_sep != 0 {
            match unsafe { read_bytes(sep_ptr, sep_len) } {
                Ok(s) => s,
                Err(c) => return c,
            }
        } else {
            pretokenize::DEFAULT_SEPARATOR
        };

        let result = match input_tag {
            0 => {
                let bytes = match unsafe { read_bytes(input_ptr, input_len) } {
                    Ok(b) => b,
                    Err(c) => return c,
                };
                let counts = pretokenize::pretokenize_par_bytes(bytes, sep);
                crate::bpe_train::train_bpe(counts, vocab_size, specials, tie)
            }
            1 => {
                let path = match unsafe { read_path(input_ptr, input_len) } {
                    Ok(p) => p,
                    Err(c) => return c,
                };
                if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                    let spec = FileSourceSpec {
                        paths: vec![path],
                        format: DocFormat::Parquet {
                            column: "text".to_string(),
                        },
                    };
                    match spec.pretokenize() {
                        Ok(counts) => crate::bpe_train::train_bpe(counts, vocab_size, specials, tie),
                        Err(e) => return set_err(GT_IO, e.to_string()),
                    }
                } else {
                    let mmap = match MmappedFile::open(&path) {
                        Ok(m) => m,
                        Err(e) => return set_err(GT_IO, format!("{}: {e}", path.display())),
                    };
                    // Borrowed map must outlive the counts: copy the train
                    // input through an owned buffer instead.
                    let owned = mmap.as_bytes().to_vec();
                    let counts = pretokenize::pretokenize_par_bytes(&owned, sep);
                    crate::bpe_train::train_bpe(counts, vocab_size, specials, tie)
                }
            }
            2 => {
                if source_or_null.is_null() {
                    return set_err(GT_INVALID_ARG, "null source handle".to_string());
                }
                let s = unsafe { &*source_or_null };
                match &s.data {
                    SourceData::Files { paths, format } => {
                        let spec = FileSourceSpec {
                            paths: paths.clone(),
                            format: format.clone(),
                        };
                        match spec.pretokenize() {
                            Ok(counts) => crate::bpe_train::train_bpe(counts, vocab_size, specials, tie),
                            Err(e) => return set_err(GT_IO, e.to_string()),
                        }
                    }
                    SourceData::Bytes { buffers, format } => {
                        // M17: train from a Bytes source must use the source's
                        // own separator, not the fn-level `sep` param (files
                        // already do via spec.pretokenize()). Each buffer is
                        // pretokenized separately then merged (like per-file
                        // pretokenize), so a None separator keeps buffers as
                        // distinct documents instead of fusing them.
                        let src_sep: &[u8] = match format {
                            DocFormat::Text {
                                separator: Some(s),
                            } => s.as_slice(),
                            DocFormat::Text { separator: None } => b"",
                            _ => sep,
                        };
                        let mut acc = pretokenize::pretokenize_par_bytes(
                            buffers.first().map(|b| b.as_slice()).unwrap_or(&[]),
                            src_sep,
                        );
                        for b in buffers.iter().skip(1) {
                            let c = pretokenize::pretokenize_par_bytes(b, src_sep);
                            for (k, v) in c {
                                *acc.entry(k).or_default() += v;
                            }
                        }
                        crate::bpe_train::train_bpe(acc, vocab_size, specials, tie)
                    }
                }
            }
            _ => {
                return set_err(
                    GT_INVALID_ARG,
                    format!("input_tag must be 0, 1, or 2, got {input_tag}"),
                );
            }
        };

        // Vocab sorted by id (mirrors the Python binding).
        let mut vocab: Vec<(u32, Vec<u8>)> = result.vocab.into_iter().collect();
        vocab.sort_by_key(|(id, _)| *id);
        // Helper to free already-malloc'd inners on mid-loop OOM.
        unsafe fn free_ptr_list(ptrs: &[*mut u8]) {
            for &p in ptrs {
                if !p.is_null() {
                    unsafe { libc::free(p as *mut std::ffi::c_void) };
                }
            }
        }
        let mut ids = Vec::with_capacity(vocab.len());
        let mut ptrs = Vec::with_capacity(vocab.len());
        let mut lens = Vec::with_capacity(vocab.len());
        for (id, bytes) in &vocab {
            ids.push(*id);
            let p = malloc_copy(bytes);
            lens.push(bytes.len());
            if p.is_null() && !bytes.is_empty() {
                unsafe { free_ptr_list(&ptrs) };
                return set_err(GT_INTERNAL, "out of memory".to_string());
            }
            ptrs.push(p);
        }
        let mut mleft = Vec::with_capacity(result.merges.len());
        let mut mleft_lens = Vec::with_capacity(result.merges.len());
        let mut mright = Vec::with_capacity(result.merges.len());
        let mut mright_lens = Vec::with_capacity(result.merges.len());
        for (l, r) in &result.merges {
            let pl = malloc_copy(l);
            let pr = malloc_copy(r);
            if (pl.is_null() && !l.is_empty()) || (pr.is_null() && !r.is_empty()) {
                unsafe {
                    if !pl.is_null() {
                        libc::free(pl as *mut std::ffi::c_void);
                    }
                    if !pr.is_null() {
                        libc::free(pr as *mut std::ffi::c_void);
                    }
                    free_ptr_list(&ptrs);
                    free_ptr_list(&mleft);
                    free_ptr_list(&mright);
                }
                return set_err(GT_INTERNAL, "out of memory".to_string());
            }
            mleft.push(pl);
            mleft_lens.push(l.len());
            mright.push(pr);
            mright_lens.push(r.len());
        }
        let vocab_n = vocab.len();
        let merge_n = result.merges.len();
        // Retain inner pointer lists: out_vec moves its Vec arg, so keep a
        // copy of the malloc'd element pointers to free on later OOM.
        let ptrs_inner = ptrs.clone();
        let mleft_inner = mleft.clone();
        let mright_inner = mright.clone();
        unsafe fn free_all_inners(
            a: &[*mut u8],
            b: &[*mut u8],
            c: &[*mut u8],
        ) {
            for &p in a.iter().chain(b.iter()).chain(c.iter()) {
                if !p.is_null() {
                    unsafe { libc::free(p as *mut std::ffi::c_void) };
                }
            }
        }
        let mut _n = 0usize;
        let c = out_vec(ids, out_vocab_ids, &mut _n);
        if c != GT_OK {
            unsafe { free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner) };
            return c;
        }
        // Publish pointer/len arrays (elements already malloc'd above).
        let c = out_vec(ptrs, out_vocab_ptrs, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(lens, out_vocab_lens, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
                free_published(*out_vocab_ptrs as *mut std::ffi::c_void);
                *out_vocab_ptrs = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(mleft, out_merge_left, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
                free_published(*out_vocab_ptrs as *mut std::ffi::c_void);
                *out_vocab_ptrs = std::ptr::null_mut();
                free_published(*out_vocab_lens as *mut std::ffi::c_void);
                *out_vocab_lens = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(mleft_lens, out_merge_left_lens, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
                free_published(*out_vocab_ptrs as *mut std::ffi::c_void);
                *out_vocab_ptrs = std::ptr::null_mut();
                free_published(*out_vocab_lens as *mut std::ffi::c_void);
                *out_vocab_lens = std::ptr::null_mut();
                free_published(*out_merge_left as *mut std::ffi::c_void);
                *out_merge_left = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(mright, out_merge_right, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
                free_published(*out_vocab_ptrs as *mut std::ffi::c_void);
                *out_vocab_ptrs = std::ptr::null_mut();
                free_published(*out_vocab_lens as *mut std::ffi::c_void);
                *out_vocab_lens = std::ptr::null_mut();
                free_published(*out_merge_left as *mut std::ffi::c_void);
                *out_merge_left = std::ptr::null_mut();
                free_published(*out_merge_left_lens as *mut std::ffi::c_void);
                *out_merge_left_lens = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(mright_lens, out_merge_right_lens, &mut _n);
        if c != GT_OK {
            unsafe {
                free_all_inners(&ptrs_inner, &mleft_inner, &mright_inner);
                free_published(*out_vocab_ids as *mut std::ffi::c_void);
                *out_vocab_ids = std::ptr::null_mut();
                free_published(*out_vocab_ptrs as *mut std::ffi::c_void);
                *out_vocab_ptrs = std::ptr::null_mut();
                free_published(*out_vocab_lens as *mut std::ffi::c_void);
                *out_vocab_lens = std::ptr::null_mut();
                free_published(*out_merge_left as *mut std::ffi::c_void);
                *out_merge_left = std::ptr::null_mut();
                free_published(*out_merge_left_lens as *mut std::ffi::c_void);
                *out_merge_left_lens = std::ptr::null_mut();
                free_published(*out_merge_right as *mut std::ffi::c_void);
                *out_merge_right = std::ptr::null_mut();
            }
            return c;
        }
        unsafe {
            *out_vocab_n = vocab_n;
            *out_merge_n = merge_n;
        }
        GT_OK
    })
}

// ---------------------------------------------------------------------------
// Pretokenizer inspection (always the FastR50k debug scheme, like upstream)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_pretokenize_iter_new(
    bytes_ptr: *const u8,
    bytes_len: usize,
    out: *mut *mut PretokenIterHandle,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let bytes = match unsafe { read_bytes(bytes_ptr, bytes_len) } {
            Ok(b) => b.to_vec(),
            Err(c) => return c,
        };
        unsafe {
            *out = Box::into_raw(Box::new(PretokenIterHandle {
                bytes,
                pos: Mutex::new(0),
            }));
        }
        GT_OK
    })
}

/// Next piece: borrows the iterator's buffer (valid until free/reset).
/// Returns 1 with `out_ptr`/`out_len` set, 0 when exhausted, -1 on error.
/// Never returns 7: panics are recorded via `gt_last_error_*` but reported
/// as -1 to stay inside the documented 1/0/-1 protocol. Concurrent `next`
/// calls are serialized through the position mutex (single logical cursor).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_pretokenize_iter_next(
    iter: *mut PretokenIterHandle,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if iter.is_null() || out_ptr.is_null() || out_len.is_null() {
        clear_special_hits();
        set_err(GT_INVALID_ARG, "null pointer".to_string());
        // Must NOT return 1: 1 means "piece available" and would be
        // misread as success by callers. Return -1 (handled as error).
        return -1;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let h = unsafe { &*iter };
        let pos = *h
            .pos
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut it = pretokenize::FastR50kPretokenizer::with_pos(&h.bytes, pos);
        match it.next() {
            Some(piece) => {
                let new_pos = it.pos();
                // Re-borrow to update: lock again (short critical section).
                *h.pos.lock().unwrap_or_else(|e| e.into_inner()) = new_pos;
                // piece borrows h.bytes, which outlives the handle.
                let (ptr, len) = (piece.0.as_ptr(), piece.0.len());
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                1
            }
            None => {
                unsafe {
                    *out_ptr = std::ptr::null();
                    *out_len = 0;
                }
                0
            }
        }
    })) {
        Ok(code) => {
            if code == 1 || code == 0 {
                clear_last_error();
                clear_special_hits();
            }
            code
        }
        Err(_) => {
            clear_special_hits();
            set_err(GT_INTERNAL, "internal panic".to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_pretokenize_iter_free(iter: *mut PretokenIterHandle) {
    if !iter.is_null() {
        unsafe { drop(Box::from_raw(iter)) };
    }
}

/// Pretoken counts: keys as parallel ptr/len arrays plus counts.
/// (Keys are variable-length byte strings, so the flat-u32 batch layout
/// does not apply — see port.md §2 Notes.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_pretokenized_counts(
    bytes_ptr: *const u8,
    bytes_len: usize,
    sep_ptr: *const u8,
    sep_len: usize,
    has_sep: i32,
    out_keys: *mut *mut *mut u8,
    out_keys_lens: *mut *mut usize,
    out_counts: *mut *mut usize,
    out_n: *mut usize,
) -> i32 {
    guard(|| {
        // Validate outs before copying input or training (late validation
        // would leak the per-key mallocs below on the null-out path).
        if out_keys.is_null() || out_keys_lens.is_null() || out_counts.is_null() || out_n.is_null()
        {
            return set_err(GT_INVALID_ARG, "null out pointer".to_string());
        }
        let bytes = match unsafe { read_bytes(bytes_ptr, bytes_len) } {
            Ok(b) => b,
            Err(c) => return c,
        };
        let sep: &[u8] = if has_sep != 0 {
            match unsafe { read_bytes(sep_ptr, sep_len) } {
                Ok(s) => s,
                Err(c) => return c,
            }
        } else {
            pretokenize::DEFAULT_SEPARATOR
        };
        // Own the input: counts borrow it, then keys are copied out.
        let owned = bytes.to_vec();
        let counts_map = pretokenize::pretokenize_par_bytes(&owned, sep);
        // Deterministic key order: HashMap iteration is nondeterministic
        // across calls, which makes C# snapshot tests order-sensitive.
        // Sort by key bytes (mirrors the sorted vocab publish in train).
        let mut entries: Vec<(&[u8], usize)> = Vec::with_capacity(counts_map.len());
        for (k, v) in &counts_map {
            let b: &[u8] = k.as_ref();
            entries.push((b, *v));
        }
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut keys: Vec<*mut u8> = Vec::with_capacity(entries.len());
        let mut lens: Vec<usize> = Vec::with_capacity(entries.len());
        let mut counts: Vec<usize> = Vec::with_capacity(entries.len());
        for (b, v) in &entries {
            let p = malloc_copy(b);
            if p.is_null() && !b.is_empty() {
                unsafe {
                    for &q in &keys {
                        if !q.is_null() {
                            libc::free(q as *mut std::ffi::c_void);
                        }
                    }
                }
                return set_err(GT_INTERNAL, "out of memory".to_string());
            }
            keys.push(p);
            lens.push(b.len());
            counts.push(*v);
        }
        let keys_inner = keys.clone();
        let n = counts.len();
        let mut _m = 0usize;
        let c = out_vec(keys, out_keys, &mut _m);
        if c != GT_OK {
            unsafe {
                for &q in &keys_inner {
                    if !q.is_null() {
                        libc::free(q as *mut std::ffi::c_void);
                    }
                }
            }
            return c;
        }
        let c = out_vec(lens, out_keys_lens, &mut _m);
        if c != GT_OK {
            unsafe {
                for &q in &keys_inner {
                    if !q.is_null() {
                        libc::free(q as *mut std::ffi::c_void);
                    }
                }
                free_published(*out_keys as *mut std::ffi::c_void);
                *out_keys = std::ptr::null_mut();
            }
            return c;
        }
        let c = out_vec(counts, out_counts, &mut _m);
        if c != GT_OK {
            unsafe {
                for &q in &keys_inner {
                    if !q.is_null() {
                        libc::free(q as *mut std::ffi::c_void);
                    }
                }
                free_published(*out_keys as *mut std::ffi::c_void);
                *out_keys = std::ptr::null_mut();
                free_published(*out_keys_lens as *mut std::ffi::c_void);
                *out_keys_lens = std::ptr::null_mut();
            }
            return c;
        }
        unsafe { *out_n = n };
        GT_OK
    })
}

// ---------------------------------------------------------------------------
// Hub + cache budget
// ---------------------------------------------------------------------------

/// Download-or-cache `filename` of `repo_id` at `revision`.
/// `repo_type`: 0 = model, 1 = dataset, 2 = space.
/// NotFound -> 3, Unauthorized -> 4, plain I/O (e.g. cache-write EACCES) -> 3,
/// other failures (HTTP 5xx/429, timeouts, network errors) -> 7.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_hub_file(
    repo_ptr: *const u8,
    repo_len: usize,
    filename_ptr: *const u8,
    filename_len: usize,
    repo_type: i32,
    revision_ptr: *const u8,
    revision_len: usize,
    out_path: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        let repo = match unsafe { read_str(repo_ptr, repo_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let filename = match unsafe { read_str(filename_ptr, filename_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let revision = match unsafe { read_str(revision_ptr, revision_len) } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let rt = match repo_type {
            0 => RepoType::Model,
            1 => RepoType::Dataset,
            2 => RepoType::Space,
            _ => {
                return set_err(
                    GT_INVALID_ARG,
                    format!("repo_type must be 0, 1, or 2, got {repo_type}"),
                );
            }
        };
        match hub::hub_file_in(rt, repo, filename, revision) {
            Ok(path) => {
                let bytes = path.as_os_str().as_encoded_bytes().to_vec();
                out_vec(bytes, out_path, out_len)
            }
            Err(e) => match e.downcast_ref::<FetchError>() {
                Some(FetchError::NotFound { .. }) => set_err(GT_IO, format!("{e:#}")),
                Some(FetchError::Unauthorized { .. }) => {
                    set_err(GT_UNAUTHORIZED, format!("{e:#}"))
                }
                None => {
                    // I/O-flavored failures (cache writes, local FS) map to 3
                    // like elsewhere; network/server errors stay 7.
                    if e.downcast_ref::<std::io::Error>().is_some() {
                        set_err(GT_IO, format!("{e:#}"))
                    } else {
                        set_err(GT_INTERNAL, format!("{e:#}"))
                    }
                }
            },
        }
    })
}

/// 1 = shaped like a Hub repo id, 0 = not (or null/invalid UTF-8).
/// A null pointer yields 0 regardless of `str_len` (by design: null is "not
/// a repo id", unlike `read_bytes` which demands rc 1 for null+len>0 —
/// this predicate cannot return a code, only a bool).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_looks_like_repo_id(str_ptr: *const u8, str_len: usize) -> i32 {
    let r = std::panic::catch_unwind(|| {
        if str_ptr.is_null() {
            return 0;
        }
        let bytes = unsafe { std::slice::from_raw_parts(str_ptr, str_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => {
                if hub::looks_like_repo_id(s) {
                    1
                } else {
                    0
                }
            }
            Err(_) => 0,
        }
    });
    r.unwrap_or(0)
}

/// Discovered HF token: 1 + malloc'd bytes, or 0 with null/0 outputs.
/// 1 is success ("has token"), not GT_INVALID_ARG: the guard's success
/// clearing only triggers on 0, so clear the stale error explicitly here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_get_hf_token(out_buf: *mut *mut u8, out_len: *mut usize) -> i32 {
    guard(|| match hub::get_hf_token() {
        Some(token) => {
            let c = out_vec(token.into_bytes(), out_buf, out_len);
            if c != GT_OK {
                return c;
            }
            clear_last_error();
            1
        }
        None => {
            if out_buf.is_null() || out_len.is_null() {
                return set_err(GT_INVALID_ARG, "null out pointer".to_string());
            }
            unsafe {
                *out_buf = std::ptr::null_mut();
                *out_len = 0;
            }
            0
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_set_max_cache_bytes(has_value: i32, value: usize) {
    let _ = guard(|| {
        *MAX_CACHE_BYTES
            .lock()
            .unwrap_or_else(|e| e.into_inner()) =
            if has_value != 0 { Some(value) } else { None };
        GT_OK
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gt_get_max_cache_bytes(out_has: *mut i32, out_value: *mut usize) {
    // Null outs are silently ignored (void fn, no code to return).
    if out_has.is_null() || out_value.is_null() {
        return;
    }
    // Lock poisoning must not unwind across FFI; recover the inner value,
    // falling back to defaults only if the lock itself panics.
    let guard =
        std::panic::catch_unwind(|| MAX_CACHE_BYTES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().copied());
    let opt = guard.unwrap_or(Some(Tokenizer::DEFAULT_MAX_CACHE_BYTES));
    match opt {
        Some(v) => unsafe {
            *out_has = 1;
            *out_value = v;
        },
        None => unsafe {
            *out_has = 0;
            *out_value = 0;
        },
    }
}
