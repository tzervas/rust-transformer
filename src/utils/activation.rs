use nalgebra::DMatrix;

pub trait Activation {
    fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64>;
    fn derivative(&self, input: &DMatrix<f64>) -> DMatrix<f64>;
}

pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        input.map(|x| if x > 0.0 { x } else { 0.0 })
    }
    
    fn derivative(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        input.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

pub struct GELU;

impl Activation for GELU {
    fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        input.map(|x| {
            0.5 * x * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
        })
    }
    
    fn derivative(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        input.map(|x| {
            let tanh_arg = std::f64::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3));
            let tanh_val = tanh_arg.tanh();
            let sech2 = 1.0 - tanh_val.powi(2);
            
            0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * std::f64::consts::FRAC_2_SQRT_PI * (1.0 + 3.0 * 0.044715 * x.powi(2))
        })
    }
}