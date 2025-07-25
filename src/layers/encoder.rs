use nalgebra::DMatrix;
use crate::attention::MultiHeadAttention;
use crate::layers::{FeedForward, LayerNorm, ResidualConnection, ActivationType};
use crate::Result;

pub struct EncoderLayer {
    multi_head_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    dropout_rate: f64,
}

impl EncoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
    ) -> Result<Self> {
        let multi_head_attention = MultiHeadAttention::new(num_heads, d_model, dropout_rate)?;
        let feed_forward = FeedForward::new(d_model, d_ff, ActivationType::ReLU, dropout_rate);
        let layer_norm1 = LayerNorm::new(d_model, 1e-6);
        let layer_norm2 = LayerNorm::new(d_model, 1e-6);
        
        Ok(Self {
            multi_head_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            dropout_rate,
        })
    }
    
    pub fn forward(
        &self,
        input: &DMatrix<f64>,
        mask: Option<&DMatrix<bool>>,
    ) -> Result<DMatrix<f64>> {
        let attention_output = self.multi_head_attention.forward(input, input, input, mask)?;
        let output1 = ResidualConnection::forward(input, &attention_output, &self.layer_norm1)?;
        
        let ff_output = self.feed_forward.forward(&output1)?;
        let output2 = ResidualConnection::forward(&output1, &ff_output, &self.layer_norm2)?;
        
        Ok(output2)
    }
}