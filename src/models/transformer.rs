use nalgebra::DMatrix;
use crate::models::{Encoder, Decoder};
use crate::utils::{create_padding_mask, create_causal_mask, combine_masks};
use crate::Result;

pub struct TransformerConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub dropout_rate: f64,
    pub pad_token_id: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            max_seq_len: 512,
            d_model: 512,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            d_ff: 2048,
            dropout_rate: 0.1,
            pad_token_id: 0,
        }
    }
}

pub struct Transformer {
    encoder: Encoder,
    decoder: Decoder,
    config: TransformerConfig,
}

impl Transformer {
    pub fn new(config: TransformerConfig) -> Result<Self> {
        let encoder = Encoder::new(
            config.num_encoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.vocab_size,
            config.max_seq_len,
            config.dropout_rate,
        )?;
        
        let decoder = Decoder::new(
            config.num_decoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.vocab_size,
            config.max_seq_len,
            config.dropout_rate,
        )?;
        
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }
    
    pub fn forward(
        &self,
        encoder_input: &[usize],
        decoder_input: &[usize],
    ) -> Result<DMatrix<f64>> {
        let encoder_padding_mask = create_padding_mask(encoder_input, self.config.pad_token_id);
        let encoder_output = self.encoder.forward(encoder_input, Some(&encoder_padding_mask))?;
        
        let decoder_padding_mask = create_padding_mask(decoder_input, self.config.pad_token_id);
        let causal_mask = create_causal_mask(decoder_input.len());
        let decoder_self_mask = combine_masks(&decoder_padding_mask, &causal_mask)
            .map_err(|e| format!("Failed to combine masks: {}", e))?;
        
        let decoder_output = self.decoder.forward(
            decoder_input,
            &encoder_output,
            Some(&decoder_self_mask),
            Some(&encoder_padding_mask),
        )?;
        
        Ok(decoder_output)
    }
    
    pub fn encode(&self, input: &[usize]) -> Result<DMatrix<f64>> {
        let padding_mask = create_padding_mask(input, self.config.pad_token_id);
        self.encoder.forward(input, Some(&padding_mask))
    }
    
    pub fn decode(
        &self,
        input: &[usize],
        encoder_output: &DMatrix<f64>,
        encoder_input: &[usize],
    ) -> Result<DMatrix<f64>> {
        let decoder_padding_mask = create_padding_mask(input, self.config.pad_token_id);
        let causal_mask = create_causal_mask(input.len());
        let decoder_self_mask = combine_masks(&decoder_padding_mask, &causal_mask)
            .map_err(|e| format!("Failed to combine masks: {}", e))?;
        
        let encoder_padding_mask = create_padding_mask(encoder_input, self.config.pad_token_id);
        
        self.decoder.forward(
            input,
            encoder_output,
            Some(&decoder_self_mask),
            Some(&encoder_padding_mask),
        )
    }
    
    pub fn generate(
        &self,
        encoder_input: &[usize],
        start_token: usize,
        max_length: usize,
    ) -> Result<Vec<usize>> {
        let encoder_output = self.encode(encoder_input)?;
        let mut generated = vec![start_token];
        
        for _ in 0..max_length {
            let decoder_output = self.decode(&generated, &encoder_output, encoder_input)?;
            let last_logits = decoder_output.row(decoder_output.nrows() - 1);
            
            let mut logits_vec = Vec::new();
            for j in 0..last_logits.ncols() {
                logits_vec.push(last_logits[(0, j)]);
            }
            let next_token = self.sample_from_vec(&logits_vec)?;
            generated.push(next_token);
            
            if next_token == self.config.pad_token_id {
                break;
            }
        }
        
        Ok(generated)
    }
    
    fn sample_from_vec(&self, logits: &[f64]) -> Result<usize> {
        let probabilities = self.softmax_vec(logits)?;
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();
        
        let mut cumulative_prob = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return Ok(i);
            }
        }
        
        Ok(probabilities.len() - 1)
    }
    
    fn softmax_vec(&self, logits: &[f64]) -> Result<Vec<f64>> {
        let max_val = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        
        if sum_exp == 0.0 {
            return Err("Softmax denominator is zero".into());
        }
        
        Ok(exp_logits.iter().map(|&x| x / sum_exp).collect())
    }
}