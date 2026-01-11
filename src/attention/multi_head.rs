use nalgebra::DMatrix;
use crate::attention::ScaledDotProductAttention;
use crate::Result;

pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    w_q: Vec<DMatrix<f64>>,
    w_k: Vec<DMatrix<f64>>,
    w_v: Vec<DMatrix<f64>>,
    w_o: DMatrix<f64>,
    attention: ScaledDotProductAttention,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize, dropout_rate: f64) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err("d_model must be divisible by num_heads".into());
        }
        
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;
        
        let mut w_q = Vec::with_capacity(num_heads);
        let mut w_k = Vec::with_capacity(num_heads);
        let mut w_v = Vec::with_capacity(num_heads);
        
        for _ in 0..num_heads {
            w_q.push(Self::initialize_weights(d_model, d_k));
            w_k.push(Self::initialize_weights(d_model, d_k));
            w_v.push(Self::initialize_weights(d_model, d_v));
        }
        
        let w_o = Self::initialize_weights(d_model, d_model);
        let attention = ScaledDotProductAttention::new(dropout_rate);
        
        Ok(Self {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            attention,
        })
    }

    pub fn forward(
        &self,
        query: &DMatrix<f64>,
        key: &DMatrix<f64>,
        value: &DMatrix<f64>,
        mask: Option<&DMatrix<bool>>,
    ) -> Result<DMatrix<f64>> {
        let batch_size = query.nrows();
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        
        for i in 0..self.num_heads {
            let q = query * &self.w_q[i];
            let k = key * &self.w_k[i];
            let v = value * &self.w_v[i];
            
            let head_output = self.attention.forward(&q, &k, &v, mask)?;
            head_outputs.push(head_output);
        }
        
        let concatenated = self.concatenate_heads(&head_outputs)?;
        let output = &concatenated * &self.w_o;
        
        Ok(output)
    }

    fn initialize_weights(input_dim: usize, output_dim: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt();
        
        DMatrix::from_fn(input_dim, output_dim, |_, _| {
            rng.gen_range(-scale..scale)
        })
    }

    fn concatenate_heads(&self, head_outputs: &[DMatrix<f64>]) -> Result<DMatrix<f64>> {
        if head_outputs.is_empty() {
            return Err("No head outputs to concatenate".into());
        }
        
        let batch_size = head_outputs[0].nrows();
        let total_dim = self.num_heads * self.d_v;
        let mut concatenated = DMatrix::zeros(batch_size, total_dim);
        
        for (i, head_output) in head_outputs.iter().enumerate() {
            let start_col = i * self.d_v;
            let end_col = (i + 1) * self.d_v;
            
            for row in 0..batch_size {
                for col in 0..self.d_v {
                    concatenated[(row, start_col + col)] = head_output[(row, col)];
                }
            }
        }
        
        Ok(concatenated)
    }
}