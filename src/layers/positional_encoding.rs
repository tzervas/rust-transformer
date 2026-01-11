use nalgebra::DMatrix;
use crate::Result;

pub trait PositionalEncoding {
    fn encode(&self, position: usize, d_model: usize) -> Result<DMatrix<f64>>;
    fn encode_sequence(&self, seq_len: usize, d_model: usize) -> Result<DMatrix<f64>>;
}

pub struct SinusoidalPositionalEncoding {
    max_seq_len: usize,
}

impl SinusoidalPositionalEncoding {
    pub fn new(max_seq_len: usize) -> Self {
        Self { max_seq_len }
    }
}

impl PositionalEncoding for SinusoidalPositionalEncoding {
    fn encode(&self, position: usize, d_model: usize) -> Result<DMatrix<f64>> {
        let mut encoding = DMatrix::zeros(1, d_model);
        
        for i in 0..d_model {
            let angle = position as f64 / 10000_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
            
            if i % 2 == 0 {
                encoding[(0, i)] = angle.sin();
            } else {
                encoding[(0, i)] = angle.cos();
            }
        }
        
        Ok(encoding)
    }
    
    fn encode_sequence(&self, seq_len: usize, d_model: usize) -> Result<DMatrix<f64>> {
        if seq_len > self.max_seq_len {
            return Err(format!("Sequence length {} exceeds maximum {}", seq_len, self.max_seq_len).into());
        }
        
        let mut encodings = DMatrix::zeros(seq_len, d_model);
        
        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = pos as f64 / 10000_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
                
                if i % 2 == 0 {
                    encodings[(pos, i)] = angle.sin();
                } else {
                    encodings[(pos, i)] = angle.cos();
                }
            }
        }
        
        Ok(encodings)
    }
}

pub struct LearnedPositionalEncoding {
    embeddings: DMatrix<f64>,
    max_seq_len: usize,
    d_model: usize,
}

impl LearnedPositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f64).sqrt();
        
        let embeddings = DMatrix::from_fn(max_seq_len, d_model, |_, _| {
            rng.gen_range(-scale..scale)
        });
        
        Self {
            embeddings,
            max_seq_len,
            d_model,
        }
    }
}

impl PositionalEncoding for LearnedPositionalEncoding {
    fn encode(&self, position: usize, d_model: usize) -> Result<DMatrix<f64>> {
        if position >= self.max_seq_len {
            return Err(format!("Position {} exceeds maximum {}", position, self.max_seq_len).into());
        }
        
        if d_model != self.d_model {
            return Err(format!("d_model {} doesn't match initialized {}", d_model, self.d_model).into());
        }
        
        let mut result = DMatrix::zeros(1, d_model);
        result.set_row(0, &self.embeddings.row(position));
        Ok(result)
    }
    
    fn encode_sequence(&self, seq_len: usize, d_model: usize) -> Result<DMatrix<f64>> {
        if seq_len > self.max_seq_len {
            return Err(format!("Sequence length {} exceeds maximum {}", seq_len, self.max_seq_len).into());
        }
        
        if d_model != self.d_model {
            return Err(format!("d_model {} doesn't match initialized {}", d_model, self.d_model).into());
        }
        
        Ok(self.embeddings.rows(0, seq_len).into_owned())
    }
}