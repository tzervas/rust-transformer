use nalgebra::DMatrix;
use crate::layers::{DecoderLayer, PositionalEncoding, SinusoidalPositionalEncoding};
use crate::Result;

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    positional_encoding: Box<dyn PositionalEncoding>,
    input_embedding: DMatrix<f64>,
    output_projection: DMatrix<f64>,
    dropout_rate: f64,
}

impl Decoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dropout_rate: f64,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DecoderLayer::new(d_model, num_heads, d_ff, dropout_rate)?);
        }
        
        let positional_encoding = Box::new(SinusoidalPositionalEncoding::new(max_seq_len));
        let input_embedding = Self::initialize_embeddings(vocab_size, d_model);
        let output_projection = Self::initialize_weights(d_model, vocab_size);
        
        Ok(Self {
            layers,
            positional_encoding,
            input_embedding,
            output_projection,
            dropout_rate,
        })
    }
    
    pub fn forward(
        &self,
        input_ids: &[usize],
        encoder_output: &DMatrix<f64>,
        self_attention_mask: Option<&DMatrix<bool>>,
        encoder_decoder_mask: Option<&DMatrix<bool>>,
    ) -> Result<DMatrix<f64>> {
        let seq_len = input_ids.len();
        let d_model = self.input_embedding.ncols();
        
        let mut embedded_input = DMatrix::zeros(seq_len, d_model);
        for (i, &token_id) in input_ids.iter().enumerate() {
            if token_id >= self.input_embedding.nrows() {
                return Err(format!("Token ID {} exceeds vocabulary size", token_id).into());
            }
            embedded_input.set_row(i, &self.input_embedding.row(token_id));
        }
        
        let positional_encodings = self.positional_encoding.encode_sequence(seq_len, d_model)?;
        let mut output = &embedded_input + &positional_encodings;
        
        for layer in &self.layers {
            output = layer.forward(
                &output,
                encoder_output,
                self_attention_mask,
                encoder_decoder_mask,
            )?;
        }
        
        let logits = &output * &self.output_projection;
        Ok(logits)
    }
    
    fn initialize_embeddings(vocab_size: usize, d_model: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f64).sqrt();
        
        DMatrix::from_fn(vocab_size, d_model, |_, _| {
            rng.gen_range(-scale..scale)
        })
    }
    
    fn initialize_weights(input_dim: usize, output_dim: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt();
        
        DMatrix::from_fn(input_dim, output_dim, |_, _| {
            rng.gen_range(-scale..scale)
        })
    }
}