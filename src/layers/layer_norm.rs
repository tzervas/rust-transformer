use nalgebra::DMatrix;
use crate::Result;

pub struct LayerNorm {
    gamma: DMatrix<f64>,
    beta: DMatrix<f64>,
    epsilon: f64,
}

impl LayerNorm {
    pub fn new(d_model: usize, epsilon: f64) -> Self {
        let gamma = DMatrix::from_element(1, d_model, 1.0);
        let beta = DMatrix::zeros(1, d_model);
        
        Self {
            gamma,
            beta,
            epsilon,
        }
    }
    
    pub fn forward(&self, input: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let (batch_size, d_model) = (input.nrows(), input.ncols());
        let mut normalized = DMatrix::zeros(batch_size, d_model);
        
        for i in 0..batch_size {
            let row = input.row(i);
            let mean = row.sum() / d_model as f64;
            
            let variance = row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / d_model as f64;
            
            let std_dev = (variance + self.epsilon).sqrt();
            
            for j in 0..d_model {
                let normalized_val = (input[(i, j)] - mean) / std_dev;
                normalized[(i, j)] = self.gamma[(0, j)] * normalized_val + self.beta[(0, j)];
            }
        }
        
        Ok(normalized)
    }
}

pub struct ResidualConnection;

impl ResidualConnection {
    pub fn forward(
        input: &DMatrix<f64>,
        sublayer_output: &DMatrix<f64>,
        layer_norm: &LayerNorm,
    ) -> Result<DMatrix<f64>> {
        if input.shape() != sublayer_output.shape() {
            return Err("Input and sublayer output shapes must match for residual connection".into());
        }
        
        let residual = input + sublayer_output;
        layer_norm.forward(&residual)
    }
}