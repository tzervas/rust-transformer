use nalgebra::DMatrix;
use crate::attention::MultiHeadAttention;
use crate::layers::{FeedForward, LayerNorm, ResidualConnection, ActivationType};
use crate::Result;

pub struct DecoderLayer {
    masked_multi_head_attention: MultiHeadAttention,
    encoder_decoder_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: LayerNorm,
    dropout_rate: f64,
}

impl DecoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
    ) -> Result<Self> {
        let masked_multi_head_attention = MultiHeadAttention::new(num_heads, d_model, dropout_rate)?;
        let encoder_decoder_attention = MultiHeadAttention::new(num_heads, d_model, dropout_rate)?;
        let feed_forward = FeedForward::new(d_model, d_ff, ActivationType::ReLU, dropout_rate);
        let layer_norm1 = LayerNorm::new(d_model, 1e-6);
        let layer_norm2 = LayerNorm::new(d_model, 1e-6);
        let layer_norm3 = LayerNorm::new(d_model, 1e-6);
        
        Ok(Self {
            masked_multi_head_attention,
            encoder_decoder_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            layer_norm3,
            dropout_rate,
        })
    }
    
    pub fn forward(
        &self,
        input: &DMatrix<f64>,
        encoder_output: &DMatrix<f64>,
        self_attention_mask: Option<&DMatrix<bool>>,
        encoder_decoder_mask: Option<&DMatrix<bool>>,
    ) -> Result<DMatrix<f64>> {
        let self_attention_output = self.masked_multi_head_attention
            .forward(input, input, input, self_attention_mask)?;
        let output1 = ResidualConnection::forward(input, &self_attention_output, &self.layer_norm1)?;
        
        let encoder_decoder_output = self.encoder_decoder_attention
            .forward(&output1, encoder_output, encoder_output, encoder_decoder_mask)?;
        let output2 = ResidualConnection::forward(&output1, &encoder_decoder_output, &self.layer_norm2)?;
        
        let ff_output = self.feed_forward.forward(&output2)?;
        let output3 = ResidualConnection::forward(&output2, &ff_output, &self.layer_norm3)?;
        
        Ok(output3)
    }
}