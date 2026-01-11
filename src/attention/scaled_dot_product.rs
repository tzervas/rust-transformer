use nalgebra::{DMatrix, DVector};
use crate::Result;

pub struct ScaledDotProductAttention {
    dropout_rate: f64,
}

impl ScaledDotProductAttention {
    pub fn new(dropout_rate: f64) -> Self {
        Self { dropout_rate }
    }

    pub fn forward(
        &self,
        query: &DMatrix<f64>,
        key: &DMatrix<f64>,
        value: &DMatrix<f64>,
        mask: Option<&DMatrix<bool>>,
    ) -> Result<DMatrix<f64>> {
        let d_k = query.ncols() as f64;
        let scale = 1.0 / d_k.sqrt();
        
        let scores = query * key.transpose() * scale;
        
        let masked_scores = if let Some(mask) = mask {
            self.apply_mask(&scores, mask)?
        } else {
            scores
        };
        
        let attention_weights = self.softmax(&masked_scores)?;
        
        let output = &attention_weights * value;
        
        Ok(output)
    }

    fn apply_mask(&self, scores: &DMatrix<f64>, mask: &DMatrix<bool>) -> Result<DMatrix<f64>> {
        if scores.shape() != mask.shape() {
            return Err("Mask shape doesn't match scores shape".into());
        }
        
        let mut masked_scores = scores.clone();
        for i in 0..scores.nrows() {
            for j in 0..scores.ncols() {
                if mask[(i, j)] {
                    masked_scores[(i, j)] = f64::NEG_INFINITY;
                }
            }
        }
        Ok(masked_scores)
    }

    fn softmax(&self, scores: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut result = DMatrix::zeros(scores.nrows(), scores.ncols());
        
        for i in 0..scores.nrows() {
            let row = scores.row(i);
            let max_val = row.max();
            
            let exp_row = row.map(|x| (x - max_val).exp());
            let sum_exp = exp_row.sum();
            
            if sum_exp == 0.0 {
                return Err("Softmax denominator is zero".into());
            }
            
            for j in 0..scores.ncols() {
                result[(i, j)] = exp_row[(0, j)] / sum_exp;
            }
        }
        
        Ok(result)
    }
}