use model::{KvFormat, PAGE_TOKENS};

use super::Reject;
use super::format::{
    HEADER_BYTES, Header, Identity, end_is_valid, end_mark, hash_parts, ids_hash, page_hash,
};

fn identity() -> Identity {
    Identity {
        format: KvFormat::F16,
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        weight_bits: 8,
        weight_group: 128,
        weights: 0x1234_5678_9abc_def0,
        config: 0x0fed_cba9_8765_4321,
    }
}

fn header(tokens: u32) -> Header {
    Header {
        identity: identity(),
        tokens,
        ids_hash: ids_hash(&[1, 2, 3]),
    }
}

#[test]
fn a_header_survives_a_round_trip() {
    for tokens in [1, 63, 64, 65, 4096] {
        let h = header(tokens);
        assert_eq!(Header::decode(&h.encode()), Ok(h));
    }
    assert_eq!(header(1).encode().len(), HEADER_BYTES);
}

#[test]
fn bf16_int8_and_different_int8_groups_cannot_share_a_snapshot() {
    let identities: Vec<_> = [(0, 0), (8, 32), (8, 128)]
        .into_iter()
        .map(|(weight_bits, weight_group)| Identity {
            weight_bits,
            weight_group,
            ..identity()
        })
        .collect();
    for saved in &identities {
        let h = Header {
            identity: *saved,
            ..header(100)
        };
        let decoded = Header::decode(&h.encode()).unwrap();
        for now in &identities {
            let expected = if saved == now {
                Ok(())
            } else if saved.weight_bits != now.weight_bits {
                Err(Reject::Foreign("weight quantization"))
            } else {
                Err(Reject::Foreign("weight quantization group"))
            };
            assert_eq!(decoded.check_against(now), expected);
        }
    }
}

#[test]
fn flipping_any_header_byte_is_noticed() {
    let bytes = header(100).encode();
    for at in 0..bytes.len() {
        let mut bad = bytes;
        bad[at] ^= 0x01;
        assert!(Header::decode(&bad).is_err(), "byte {at}");
    }
    assert!(Header::decode(&bytes[..HEADER_BYTES - 1]).is_err());
    assert!(Header::decode(&[]).is_err());
}

#[test]
fn every_part_of_the_identity_must_match() {
    let h = header(10);
    assert_eq!(h.check_against(&identity()), Ok(()));
    let others: [(Identity, &str); 8] = [
        (
            Identity {
                format: KvFormat::F32,
                ..identity()
            },
            "KV type",
        ),
        (
            Identity {
                layers: 27,
                ..identity()
            },
            "layer count",
        ),
        (
            Identity {
                kv_heads: 4,
                ..identity()
            },
            "kv heads",
        ),
        (
            Identity {
                head_dim: 64,
                ..identity()
            },
            "head size",
        ),
        (
            Identity {
                weight_bits: 4,
                ..identity()
            },
            "weight quantization",
        ),
        (
            Identity {
                weight_group: 32,
                ..identity()
            },
            "weight quantization group",
        ),
        (
            Identity {
                config: 1,
                ..identity()
            },
            "model configuration or tokenizer",
        ),
        (
            Identity {
                weights: 1,
                ..identity()
            },
            "model weights",
        ),
    ];
    for (now, what) in others {
        assert_eq!(h.check_against(&now), Err(Reject::Foreign(what)));
    }
}

#[test]
fn sizes_follow_the_page_layout() {
    let h = header(2 * PAGE_TOKENS as u32 + 6);
    assert_eq!(h.rows_of(0), 64);
    assert_eq!(h.rows_of(1), 64);
    assert_eq!(h.rows_of(2), 6);
    assert_eq!(h.rows_of(3), 0);
    let row = 8 * 128 * 2;
    assert_eq!(h.page_bytes(64), Some(28 * 2 * 64 * row));
    let pages: usize = [64, 64, 6].iter().map(|r| 28 * 2 * r * row + 8).sum();
    assert_eq!(h.file_bytes(), Some(80 + 4 * 134 + pages + 16));
    assert_eq!(header(0).file_bytes(), Some(80 + 16));
}

#[test]
fn hashes_depend_on_every_input() {
    assert_ne!(ids_hash(&[1, 2]), ids_hash(&[2, 1]));
    assert_ne!(ids_hash(&[1, 2]), ids_hash(&[1, 2, 0]));
    assert_ne!(page_hash(0, b"abc"), page_hash(1, b"abc"));
    assert_ne!(page_hash(0, b"abc"), page_hash(0, b"abd"));
    assert_ne!(hash_parts(&[b"ab", b"c"]), hash_parts(&[b"abd"]));
}

#[test]
fn the_end_mark_names_the_total_length() {
    assert!(end_is_valid(&end_mark(1234), 1234));
    assert!(!end_is_valid(&end_mark(1234), 1235));
    assert!(!end_is_valid(&end_mark(1234)[..15], 1234));
}
