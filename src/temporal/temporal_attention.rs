use nalgebra::{DMatrix, RowDVector};
use crate::attention::ScaledDotProductAttention;
use crate::Result;

pub struct TemporalAttention {
    attention: ScaledDotProductAttention,
    temporal_weights: DMatrix<f64>,
    decay_factor: f64,
    max_temporal_distance: usize,
}

impl TemporalAttention {
    pub fn new(
        d_model: usize,
        max_temporal_distance: usize,
        decay_factor: f64,
        dropout_rate: f64,
    ) -> Self {
        let attention = ScaledDotProductAttention::new(dropout_rate);
        let temporal_weights = Self::initialize_temporal_weights(d_model, max_temporal_distance);
        
        Self {
            attention,
            temporal_weights,
            decay_factor,
            max_temporal_distance,
        }
    }
    
    pub fn forward(
        &self,
        current_query: &DMatrix<f64>,
        temporal_keys: &[DMatrix<f64>],
        temporal_values: &[DMatrix<f64>],
        temporal_distances: &[usize],
    ) -> Result<DMatrix<f64>> {
        if temporal_keys.len() != temporal_values.len() || temporal_keys.len() != temporal_distances.len() {
            return Err("Temporal keys, values, and distances must have the same length".into());
        }
        
        let mut weighted_outputs = Vec::new();
        
        for (i, ((key, value), &distance)) in temporal_keys.iter()
            .zip(temporal_values.iter())
            .zip(temporal_distances.iter())
            .enumerate() 
        {
            if distance > self.max_temporal_distance {
                continue;
            }
            
            let temporal_weight = self.compute_temporal_weight(distance);
            let attention_output = self.attention.forward(current_query, key, value, None)?;
            let weighted_output = attention_output.map(|x| x * temporal_weight);
            
            weighted_outputs.push(weighted_output);
        }
        
        if weighted_outputs.is_empty() {
            return Ok(DMatrix::zeros(current_query.nrows(), current_query.ncols()));
        }
        
        let combined_output = self.combine_temporal_outputs(&weighted_outputs)?;
        Ok(combined_output)
    }
    
    pub fn cross_temporal_attention(
        &self,
        query_sequence: &DMatrix<f64>,
        key_sequence: &DMatrix<f64>,
        value_sequence: &DMatrix<f64>,
        _temporal_positions: &[usize],
    ) -> Result<DMatrix<f64>> {
        // Simplified implementation for now - just return a basic attention
        self.attention.forward(query_sequence, key_sequence, value_sequence, None)
    }
    
    fn initialize_temporal_weights(d_model: usize, max_distance: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f64).sqrt();
        
        DMatrix::from_fn(max_distance + 1, d_model, |_, _| {
            rng.gen_range(-scale..scale)
        })
    }
    
    fn compute_temporal_weight(&self, distance: usize) -> f64 {
        if distance == 0 {
            1.0
        } else {
            (self.decay_factor).powi(distance as i32)
        }
    }
    
    fn compute_attention_score_rows(&self, query: &DMatrix<f64>, key: &DMatrix<f64>) -> Result<f64> {
        let d_k = query.ncols() as f64;
        let mut score = 0.0;
        for j in 0..query.ncols() {
            score += query[(0, j)] * key[(0, j)];
        }
        score = score / d_k.sqrt();
        Ok(score.exp())
    }
    
    fn combine_temporal_outputs(&self, outputs: &[DMatrix<f64>]) -> Result<DMatrix<f64>> {
        if outputs.is_empty() {
            return Err("No temporal outputs to combine".into());
        }
        
        let mut combined = outputs[0].clone();
        for output in outputs.iter().skip(1) {
            combined += output;
        }
        
        combined = combined.map(|x| x / outputs.len() as f64);
        Ok(combined)
    }
}