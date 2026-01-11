use nalgebra::DMatrix;
use crate::utils::activation::{Activation, ReLU, GELU};
use crate::Result;

pub enum ActivationType {
    ReLU,
    GELU,
}

pub struct FeedForward {
    w1: DMatrix<f64>,
    b1: DMatrix<f64>,
    w2: DMatrix<f64>,
    b2: DMatrix<f64>,
    activation: Box<dyn Activation>,
    dropout_rate: f64,
}

impl FeedForward {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        activation_type: ActivationType,
        dropout_rate: f64,
    ) -> Self {
        let w1 = Self::initialize_weights(d_model, d_ff);
        let b1 = DMatrix::zeros(1, d_ff);
        let w2 = Self::initialize_weights(d_ff, d_model);
        let b2 = DMatrix::zeros(1, d_model);
        
        let activation: Box<dyn Activation> = match activation_type {
            ActivationType::ReLU => Box::new(ReLU),
            ActivationType::GELU => Box::new(GELU),
        };
        
        Self {
            w1,
            b1,
            w2,
            b2,
            activation,
            dropout_rate,
        }
    }
    
    pub fn forward(&self, input: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let hidden = input * &self.w1 + &self.b1;
        let activated = self.activation.forward(&hidden);
        let output = &activated * &self.w2 + &self.b2;
        
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
}