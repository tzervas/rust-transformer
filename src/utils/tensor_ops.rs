use nalgebra::DMatrix;

pub fn dropout(input: &DMatrix<f64>, rate: f64, training: bool) -> DMatrix<f64> {
    if !training || rate == 0.0 {
        return input.clone();
    }
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let keep_prob = 1.0 - rate;
    let scale = 1.0 / keep_prob;
    
    input.map(|x| {
        if rng.gen::<f64>() < keep_prob {
            x * scale
        } else {
            0.0
        }
    })
}

pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
}

pub fn swish(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn layer_norm_stats(input: &DMatrix<f64>) -> Vec<(f64, f64)> {
    let mut stats = Vec::with_capacity(input.nrows());
    
    for i in 0..input.nrows() {
        let row = input.row(i);
        let mean = row.sum() / input.ncols() as f64;
        
        let variance = row.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / input.ncols() as f64;
        
        stats.push((mean, variance));
    }
    
    stats
}