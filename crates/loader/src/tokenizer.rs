use std::path::Path;

use tokenizers::tokenizer::DecodeStream;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
};

use crate::LoaderError;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

type Stream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Turns generated ids into text, holding back bytes of a character that is not complete yet.
pub struct TokenStream<'a> {
    stream: Stream<'a>,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, LoaderError> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| LoaderError::Tokenizer {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
        Ok(Self { inner })
    }

    /// Number of ids that decode to text; model logits beyond this are padding.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LoaderError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| LoaderError::TokenizerRuntime(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, LoaderError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| LoaderError::TokenizerRuntime(e.to_string()))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Text of every special token, e.g. `<|im_start|>`.
    pub fn special_tokens(&self) -> Vec<String> {
        let mut specials: Vec<String> = self
            .inner
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect();
        specials.sort();
        specials
    }

    pub fn token_text(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn stream(&self) -> TokenStream<'_> {
        TokenStream {
            stream: self.inner.decode_stream(false),
        }
    }
}

impl TokenStream<'_> {
    /// Text that became printable after adding `id`, if any.
    pub fn push(&mut self, id: u32) -> Result<Option<String>, LoaderError> {
        self.stream
            .step(id)
            .map_err(|e| LoaderError::TokenizerRuntime(e.to_string()))
    }
}
