//! Integration tests for the .NET port's own Rust code: the SentencePiece
//! `.model` -> `tokenizer.json` converter (`sp_model`, a port of
//! `gigatoken/_load/sentencepiece.py`). The vendored upstream core is covered
//! through the C# suite (same engine, exercised via P/Invoke); upstream's own
//! `#[cfg(test)]` modules are not compiled here (they assume the full
//! upstream crate context).
//!
//! Run: `cargo test -p gigatoken-cabi --test sp_model`

use gigatoken_cabi::sp_model::sentencepiece_to_tokenizer_json;

fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut b = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
    out
}

fn tag(field: u32, wire: u32) -> Vec<u8> {
    varint(((field << 3) | wire) as u64)
}

fn len_field(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = tag(field, 2);
    out.extend(varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn varint_field(field: u32, v: u64) -> Vec<u8> {
    let mut out = tag(field, 0);
    out.extend(varint(v));
    out
}

// Field numbers (sentencepiece_model.proto).
const MODEL_PIECES: u32 = 1;
const MODEL_TRAINER_SPEC: u32 = 2;
const MODEL_NORMALIZER_SPEC: u32 = 3;
const PIECE_PIECE: u32 = 1;
const PIECE_SCORE: u32 = 2;
const PIECE_TYPE: u32 = 3;
const TRAINER_MODEL_TYPE: u32 = 3;
const TRAINER_TREAT_WHITESPACE_AS_SUFFIX: u32 = 24;
const TRAINER_BYTE_FALLBACK: u32 = 35;
const TRAINER_UNK_PIECE: u32 = 45;
const NORM_PRECOMPILED_CHARSMAP: u32 = 2;
const NORM_ESCAPE_WHITESPACES: u32 = 5;

fn piece(content: &str, score: f32, ptype: u64) -> Vec<u8> {
    let mut inner = len_field(PIECE_PIECE, content.as_bytes());
    let mut key = tag(PIECE_SCORE, 5);
    key.extend_from_slice(&score.to_le_bytes());
    inner.extend(key);
    inner.extend(varint_field(PIECE_TYPE, ptype));
    len_field(MODEL_PIECES, &inner)
}

fn trainer(pairs: &[(u32, u64)]) -> Vec<u8> {
    let mut inner = Vec::new();
    for (f, v) in pairs {
        inner.extend(varint_field(*f, *v));
    }
    len_field(MODEL_TRAINER_SPEC, &inner)
}

fn valid_model(extra_trainer: &[(u32, u64)], extra_pieces: &[Vec<u8>]) -> Vec<u8> {
    let mut m = piece("a", 1.0, 1);
    m.extend(piece("b", 1.0, 1));
    m.extend(piece("ab", 2.0, 1));
    for p in extra_pieces {
        m.extend(p);
    }
    let mut t = vec![(TRAINER_MODEL_TYPE, 2), (TRAINER_BYTE_FALLBACK, 1)];
    t.extend_from_slice(extra_trainer);
    m.extend(trainer(&t));
    m
}

#[test]
fn truncated_and_overflow_varints() {
    // Lone continuation byte at top level: truncated varint.
    assert!(
        sentencepiece_to_tokenizer_json(&[0x80])
            .unwrap_err()
            .contains("truncated varint")
    );
    // 10 continuation bytes: shift reaches 64 -> overflow.
    assert!(
        sentencepiece_to_tokenizer_json(&vec![0x80u8; 10])
            .unwrap_err()
            .contains("varint overflow")
    );
}

#[test]
fn truncated_and_bad_wire_types() {
    // Truncated fixed64.
    let mut t = tag(1, 1);
    t.extend([0, 0, 0]);
    assert!(
        sentencepiece_to_tokenizer_json(&t)
            .unwrap_err()
            .contains("truncated fixed64")
    );
    // Truncated fixed32.
    let mut t = tag(1, 5);
    t.extend([0, 0]);
    assert!(
        sentencepiece_to_tokenizer_json(&t)
            .unwrap_err()
            .contains("truncated fixed32")
    );
    // Declared LEN longer than the buffer.
    let mut t = tag(1, 2);
    t.extend(varint(100));
    t.extend([0; 3]);
    assert!(
        sentencepiece_to_tokenizer_json(&t)
            .unwrap_err()
            .contains("truncated field")
    );
    // Huge LEN varint (would wrap i+len without checked_add).
    let mut t = tag(1, 2);
    t.extend(varint(u64::MAX));
    assert!(
        sentencepiece_to_tokenizer_json(&t)
            .unwrap_err()
            .contains("truncated field")
    );
    // Unsupported wire type 3.
    let bad = tag(1, 3);
    assert!(
        sentencepiece_to_tokenizer_json(&bad)
            .unwrap_err()
            .contains("unsupported protobuf wire type 3")
    );
}

#[test]
fn rejects_empty_and_non_bpe() {
    assert!(
        sentencepiece_to_tokenizer_json(&[])
            .unwrap_err()
            .contains("no pieces found")
    );
    // Unigram model_type=1, and missing model_type (defaults to unigram).
    for opt in [vec![(TRAINER_MODEL_TYPE, 1)], vec![]] {
        let mut extra = opt;
        extra.push((TRAINER_BYTE_FALLBACK, 1));
        let mut m = piece("a", 1.0, 1);
        m.extend(trainer(&extra));
        assert!(
            sentencepiece_to_tokenizer_json(&m)
                .unwrap_err()
                .contains("got model_type \"unigram\"")
        );
    }
    // Word/char/other model kinds name themselves.
    for (v, name) in [(3, "word"), (4, "char"), (9, "9")] {
        let mut m = piece("a", 1.0, 1);
        m.extend(trainer(&[(TRAINER_MODEL_TYPE, v), (TRAINER_BYTE_FALLBACK, 1)]));
        assert!(
            sentencepiece_to_tokenizer_json(&m)
                .unwrap_err()
                .contains(&format!("got model_type {name:?}")),
            "{v}"
        );
    }
}

#[test]
fn rejects_missing_byte_fallback_and_bad_flags() {
    // byte_fallback unset.
    let mut m = piece("a", 1.0, 1);
    m.extend(trainer(&[(TRAINER_MODEL_TYPE, 2)]));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("byte_fallback")
    );
    // escape_whitespaces=false.
    let mut m = valid_model(&[], &[]);
    let norm = varint_field(NORM_ESCAPE_WHITESPACES, 0);
    m.extend(len_field(MODEL_NORMALIZER_SPEC, &norm));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("escape_whitespaces=false")
    );
    // treat_whitespace_as_suffix=true.
    let m = valid_model(&[(TRAINER_TREAT_WHITESPACE_AS_SUFFIX, 1)], &[]);
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("treat_whitespace_as_suffix")
    );
}

#[test]
fn rejects_bad_pieces_and_specs() {
    // Piece field with varint instead of bytes.
    let mut m = varint_field(MODEL_PIECES, 5);
    m.extend(trainer(&[(TRAINER_MODEL_TYPE, 2), (TRAINER_BYTE_FALLBACK, 1)]));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("bad piece field")
    );
    // Bad score length (not 4 bytes).
    let mut inner = len_field(PIECE_PIECE, b"a");
    inner.extend(len_field(PIECE_SCORE, &[0, 0]));
    inner.extend(varint_field(PIECE_TYPE, 1));
    let mut m = len_field(MODEL_PIECES, &inner);
    m.extend(trainer(&[(TRAINER_MODEL_TYPE, 2), (TRAINER_BYTE_FALLBACK, 1)]));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("bad piece score")
    );
    // Non-UTF8 piece content.
    let mut inner = len_field(PIECE_PIECE, &[0xFF, 0xFE]);
    inner.extend(varint_field(PIECE_TYPE, 1));
    let mut m = len_field(MODEL_PIECES, &inner);
    m.extend(trainer(&[(TRAINER_MODEL_TYPE, 2), (TRAINER_BYTE_FALLBACK, 1)]));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("piece is not UTF-8")
    );
    // Bad trainer-spec field type.
    let mut m = piece("a", 1.0, 1);
    m.extend(varint_field(MODEL_TRAINER_SPEC, 5));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("bad trainer spec")
    );
    // Bad normalizer-spec field type.
    let mut m = piece("a", 1.0, 1);
    m.extend(trainer(&[(TRAINER_MODEL_TYPE, 2), (TRAINER_BYTE_FALLBACK, 1)]));
    m.extend(varint_field(MODEL_NORMALIZER_SPEC, 5));
    assert!(
        sentencepiece_to_tokenizer_json(&m)
            .unwrap_err()
            .contains("bad normalizer spec")
    );
    // Non-UTF8 unk piece (last duplicate wins, still validated).
    let mut m = piece("a", 1.0, 1);
    let mut inner = varint_field(TRAINER_MODEL_TYPE, 2);
    inner.extend(varint_field(TRAINER_BYTE_FALLBACK, 1));
    let mut unk = tag(TRAINER_UNK_PIECE, 2);
    unk.extend(varint(1));
    unk.push(0xFF);
    inner.extend(unk);
    inner.extend(len_field(TRAINER_UNK_PIECE, b"<unknown>"));
    m.extend(len_field(MODEL_TRAINER_SPEC, &inner));
    // Last value wins: valid "<unknown>" is used, conversion succeeds.
    let json = sentencepiece_to_tokenizer_json(&m).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["model"]["unk_token"], "<unknown>");
}

#[test]
fn valid_model_produces_bpe_json() {
    let m = valid_model(&[], &[]);
    let json = sentencepiece_to_tokenizer_json(&m).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["model"]["type"], "BPE");
    assert_eq!(v["model"]["vocab"]["a"], 0);
    assert_eq!(v["model"]["vocab"]["ab"], 2);
    assert_eq!(v["model"]["unk_token"], "<unk>");
    assert_eq!(v["model"]["byte_fallback"], true);
    let merges = v["model"]["merges"].as_array().unwrap();
    assert!(merges.iter().any(|x| x[0] == "a" && x[1] == "b"));
    assert_eq!(v["added_tokens"].as_array().unwrap().len(), 0);
    // Default normalizers: Strip + Replace + Prepend + Replace.
    let norms = v["normalizer"]["normalizers"].as_array().unwrap();
    assert_eq!(norms.len(), 4);
}

#[test]
fn control_and_user_defined_pieces() {
    let extra = vec![piece("<ctrl>", 0.0, 3), piece("<user>", 0.0, 4)];
    let m = valid_model(&[], &extra);
    let json = sentencepiece_to_tokenizer_json(&m).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let added = v["added_tokens"].as_array().unwrap();
    assert_eq!(added.len(), 2);
    assert_eq!(added[0]["content"], "<ctrl>");
    assert_eq!(added[0]["special"], true);
    assert_eq!(added[1]["content"], "<user>");
    assert_eq!(added[1]["special"], false);
}

#[test]
fn precompiled_charsmap_normalizer() {
    // Non-empty precompiled charmap -> Precompiled entry with base64 payload.
    let mut m = valid_model(&[], &[]);
    let norm = len_field(NORM_PRECOMPILED_CHARSMAP, &[1, 2, 3, 4]);
    m.extend(len_field(MODEL_NORMALIZER_SPEC, &norm));
    let json = sentencepiece_to_tokenizer_json(&m).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let norms = v["normalizer"]["normalizers"].as_array().unwrap();
    assert_eq!(norms[0]["type"], "Precompiled");
    assert_eq!(norms[0]["precompiled_charsmap"], "AQIDBA==");
}
