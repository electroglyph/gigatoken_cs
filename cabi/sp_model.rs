//! Port of `gigatoken/_load/sentencepiece.py::sentencepiece_to_tokenizer_json`
//! using only this crate's dependencies (no `sentencepiece`/protobuf).
//!
//! Only BPE models (`model_type: BPE`) with byte fallback are supported —
//! the same class the vendored `SentencePieceBPE` backend implements.

use std::collections::HashMap;

use base64::Engine as _;

// sentencepiece_model.proto field numbers.
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
const NORM_ADD_DUMMY_PREFIX: u32 = 3;
const NORM_REMOVE_EXTRA_WHITESPACES: u32 = 4;
const NORM_ESCAPE_WHITESPACES: u32 = 5;
// SentencePiece.Type values.
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

#[derive(Clone)]
enum FieldValue {
    Varint(u64),
    Bytes(Vec<u8>),
}

fn read_varint(buf: &[u8], mut i: usize) -> Result<(u64, usize), String> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *buf
            .get(i)
            .ok_or_else(|| "not a sentencepiece model: truncated varint".to_string())?;
        i += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((result, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err("not a sentencepiece model: varint overflow".to_string());
        }
    }
}

fn fields(buf: &[u8]) -> Result<Vec<(u32, FieldValue)>, String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        let (key, ni) = read_varint(buf, i)?;
        i = ni;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        match wire {
            0 => {
                let (v, ni) = read_varint(buf, i)?;
                i = ni;
                out.push((field, FieldValue::Varint(v)));
            }
            1 => {
                let end = i + 8;
                if end > buf.len() {
                    return Err("not a sentencepiece model: truncated fixed64".to_string());
                }
                out.push((field, FieldValue::Bytes(buf[i..end].to_vec())));
                i = end;
            }
            2 => {
                let (len, ni) = read_varint(buf, i)?;
                i = ni;
                // H13: `i + (len as usize)` wrapped on huge LEN varints
                // (debug: add-overflow panic at this line; release: wrap then
                // slice panic) and surfaced as rc 7 instead of rc 5.
                let len_usize =
                    usize::try_from(len).map_err(|_| "not a sentencepiece model: truncated field".to_string())?;
                let end = i
                    .checked_add(len_usize)
                    .ok_or_else(|| "not a sentencepiece model: truncated field".to_string())?;
                if end > buf.len() {
                    return Err("not a sentencepiece model: truncated field".to_string());
                }
                out.push((field, FieldValue::Bytes(buf[i..end].to_vec())));
                i = end;
            }
            5 => {
                let end = i + 4;
                if end > buf.len() {
                    return Err("not a sentencepiece model: truncated fixed32".to_string());
                }
                out.push((field, FieldValue::Bytes(buf[i..end].to_vec())));
                i = end;
            }
            _ => {
                return Err(format!(
                    "not a sentencepiece model: unsupported protobuf wire type {wire}"
                ));
            }
        }
    }
    Ok(out)
}

/// Last-value-wins view of a message's fields (protobuf semantics).
fn scalars(buf: &[u8]) -> Result<HashMap<u32, FieldValue>, String> {
    let mut map = HashMap::new();
    for (f, v) in fields(buf)? {
        map.insert(f, v);
    }
    Ok(map)
}

fn varint_or(model: &HashMap<u32, FieldValue>, field: u32, default: u64) -> u64 {
    match model.get(&field) {
        Some(FieldValue::Varint(v)) => *v,
        _ => default,
    }
}

fn bytes_or<'a>(model: &'a HashMap<u32, FieldValue>, field: u32, default: &'a [u8]) -> &'a [u8] {
    match model.get(&field) {
        Some(FieldValue::Bytes(b)) => b,
        _ => default,
    }
}

pub fn sentencepiece_to_tokenizer_json(data: &[u8]) -> Result<String, String> {
    let mut pieces: Vec<(String, f32, i64)> = Vec::new();
    let mut trainer: HashMap<u32, FieldValue> = HashMap::new();
    let mut normspec: HashMap<u32, FieldValue> = HashMap::new();
    for (field, value) in fields(data)? {
        match field {
            MODEL_PIECES => {
                let bytes = match value {
                    FieldValue::Bytes(b) => b,
                    _ => return Err("not a sentencepiece model: bad piece field".to_string()),
                };
                let piece = scalars(&bytes)?;
                let content = bytes_or(&piece, PIECE_PIECE, b"").to_vec();
                let raw_score = bytes_or(&piece, PIECE_SCORE, &[0, 0, 0, 0]).to_vec();
                if raw_score.len() != 4 {
                    return Err("not a sentencepiece model: bad piece score".to_string());
                }
                let piece_type = match piece.get(&PIECE_TYPE) {
                    Some(FieldValue::Varint(v)) => *v as i64,
                    _ => 1,
                };
                let score = f32::from_le_bytes([raw_score[0], raw_score[1], raw_score[2], raw_score[3]]);
                let content = String::from_utf8(content)
                    .map_err(|_| "not a sentencepiece model: piece is not UTF-8".to_string())?;
                pieces.push((content, score, piece_type));
            }
            MODEL_TRAINER_SPEC => {
                let bytes = match value {
                    FieldValue::Bytes(b) => b,
                    _ => return Err("not a sentencepiece model: bad trainer spec".to_string()),
                };
                trainer = scalars(&bytes)?;
            }
            MODEL_NORMALIZER_SPEC => {
                let bytes = match value {
                    FieldValue::Bytes(b) => b,
                    _ => {
                        return Err("not a sentencepiece model: bad normalizer spec".to_string());
                    }
                };
                normspec = scalars(&bytes)?;
            }
            _ => {}
        }
    }

    if pieces.is_empty() {
        return Err("not a sentencepiece model: no pieces found".to_string());
    }
    let model_type = varint_or(&trainer, TRAINER_MODEL_TYPE, 1);
    if model_type != 2 {
        let kind = match model_type {
            1 => "unigram".to_string(),
            2 => "BPE".to_string(),
            3 => "word".to_string(),
            4 => "char".to_string(),
            other => format!("{other}"),
        };
        return Err(format!(
            "only BPE sentencepiece models are supported, got model_type {kind:?}"
        ));
    }
    if varint_or(&trainer, TRAINER_BYTE_FALLBACK, 0) == 0 {
        return Err("only byte_fallback sentencepiece models are supported".to_string());
    }
    if varint_or(&normspec, NORM_ESCAPE_WHITESPACES, 1) == 0 {
        return Err(
            "sentencepiece models with escape_whitespaces=false are not supported".to_string(),
        );
    }
    if varint_or(&trainer, TRAINER_TREAT_WHITESPACE_AS_SUFFIX, 0) != 0 {
        return Err(
            "sentencepiece models with treat_whitespace_as_suffix are not supported".to_string(),
        );
    }

    let mut vocab: HashMap<&str, usize> = HashMap::with_capacity(pieces.len());
    for (i, (piece, _, _)) in pieces.iter().enumerate() {
        vocab.insert(piece.as_str(), i);
    }
    let unk_piece = bytes_or(&trainer, TRAINER_UNK_PIECE, b"<unk>");
    let unk_piece = std::str::from_utf8(unk_piece)
        .map_err(|_| "not a sentencepiece model: unk piece is not UTF-8".to_string())?;

    let mut normalizers = Vec::new();
    let charsmap = bytes_or(&normspec, NORM_PRECOMPILED_CHARSMAP, b"");
    if !charsmap.is_empty() {
        normalizers.push(serde_json::json!({
            "type": "Precompiled",
            "precompiled_charsmap": base64::engine::general_purpose::STANDARD.encode(charsmap),
        }));
    }
    if varint_or(&normspec, NORM_REMOVE_EXTRA_WHITESPACES, 1) != 0 {
        normalizers.push(serde_json::json!({"type": "Strip", "strip_left": true, "strip_right": true}));
        normalizers.push(serde_json::json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}));
    }
    if varint_or(&normspec, NORM_ADD_DUMMY_PREFIX, 1) != 0 {
        normalizers.push(serde_json::json!({"type": "Prepend", "prepend": "▁"}));
    }
    normalizers.push(serde_json::json!({"type": "Replace", "pattern": {"String": " "}, "content": "▁"}));

    let mut added_tokens = Vec::new();
    for (i, (piece, _, piece_type)) in pieces.iter().enumerate() {
        if *piece_type == TYPE_CONTROL || *piece_type == TYPE_USER_DEFINED {
            added_tokens.push(serde_json::json!({
                "id": i,
                "content": piece,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": *piece_type == TYPE_CONTROL,
            }));
        }
    }

    // vocab map: piece -> id (insertion order not significant for JSON objects).
    let vocab_map: serde_json::Map<String, serde_json::Value> = pieces
        .iter()
        .enumerate()
        .map(|(i, (piece, _, _))| (piece.clone(), serde_json::json!(i)))
        .collect();
    let merges = generate_merges(&pieces, &vocab);
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk_piece,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": vocab_map,
            "merges": merges,
        },
    });
    serde_json::to_string(&tokenizer_json)
        .map_err(|e| format!("failed to serialize tokenizer JSON: {e}"))
}

/// BPE merges recovered from the vocab, mirroring `_generate_merges`:
/// every in-vocab split of a piece is a merge, ranked by the merged piece's
/// score (descending). Local per-piece order is by `(vocab[left],
/// vocab[right])`; the global order is by `(score, len(left), len(right))`
/// descending, both sorts stable like Python's.
fn generate_merges(pieces: &[(String, f32, i64)], vocab: &HashMap<&str, usize>) -> Vec<(String, String)> {
    let mut merges: Vec<(String, String, f32)> = Vec::new();
    for (merged, score, _) in pieces.iter() {
        let merged_chars: Vec<char> = merged.chars().collect();
        // Byte-equivalent split positions: Python splits on str indices, which
        // for non-ASCII pieces can cut inside a char; such splits never match
        // vocab entries, so char-boundary splits are equivalent.
        let mut local: Vec<(String, String, f32)> = Vec::new();
        let mut byte_pos = 0usize;
        let mut char_indices: Vec<usize> = Vec::with_capacity(merged_chars.len() + 1);
        for ch in &merged_chars {
            char_indices.push(byte_pos);
            byte_pos += ch.len_utf8();
        }
        for &split in char_indices.iter().skip(1) {
            let left = &merged[..split];
            let right = &merged[split..];
            if vocab.contains_key(left) && vocab.contains_key(right) {
                local.push((left.to_string(), right.to_string(), *score));
            }
        }
        local.sort_by(|a, b| {
            (vocab[a.0.as_str()], vocab[a.1.as_str()]).cmp(&(vocab[b.0.as_str()], vocab[b.1.as_str()]))
        });
        merges.extend(local);
    }
    merges.sort_by(|a, b| {
        // Python `len()` counts chars, not bytes.
        let ka = (a.2, a.0.chars().count(), a.1.chars().count());
        let kb = (b.2, b.0.chars().count(), b.1.chars().count());
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    merges
        .into_iter()
        .map(|(l, r, _)| (l, r))
        .collect()
}
