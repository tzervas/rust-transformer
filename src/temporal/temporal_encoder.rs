use nalgebra::DMatrix;
use crate::models::Encoder;
use crate::temporal::{TemporalAttention, MemoryBank, MemoryConfig};
use crate::layers::{LayerNorm, ResidualConnection};
use crate::Result;

pub struct TemporalEncoder {
    base_encoder: Encoder,
    temporal_attention: TemporalAttention,
    memory_bank: MemoryBank,
    temporal_layer_norm: LayerNorm,
    memory_integration_weights: DMatrix<f64>,
    temporal_decay_factor: f64,
}

impl TemporalEncoder {
    pub fn new(
        base_encoder: Encoder,
        d_model: usize,
        max_temporal_distance: usize,
        temporal_decay_factor: f64,
        memory_config: MemoryConfig,
        dropout_rate: f64,
    ) -> Self {
        let temporal_attention = TemporalAttention::new(
            d_model,
            max_temporal_distance,
            temporal_decay_factor,
            dropout_rate,
        );
        
        let memory_bank = MemoryBank::new(memory_config);
        let temporal_layer_norm = LayerNorm::new(d_model, 1e-6);
        let memory_integration_weights = Self::initialize_weights(d_model, d_model);
        
        Self {
            base_encoder,
            temporal_attention,
            memory_bank,
            temporal_layer_norm,
            memory_integration_weights,
            temporal_decay_factor,
        }
    }
    
    pub fn forward(
        &mut self,
        input_ids: &[usize],
        mask: Option<&DMatrix<bool>>,
        use_temporal_context: bool,
    ) -> Result<DMatrix<f64>> {
        let base_output = self.base_encoder.forward(input_ids, mask)?;
        
        if !use_temporal_context {
            self.store_in_memory(&base_output)?;
            return Ok(base_output);
        }
        
        let temporal_enhanced = self.apply_temporal_attention(&base_output)?;
        self.store_in_memory(&temporal_enhanced)?;
        
        Ok(temporal_enhanced)
    }
    
    pub fn forward_with_continuity(
        &mut self,
        input_ids: &[usize],
        mask: Option<&DMatrix<bool>>,
        previous_states: &[DMatrix<f64>],
        temporal_positions: &[usize],
    ) -> Result<DMatrix<f64>> {
        let base_output = self.base_encoder.forward(input_ids, mask)?;
        
        let temporal_context = if !previous_states.is_empty() {
            self.compute_temporal_context(&base_output, previous_states, temporal_positions)?
        } else {
            self.retrieve_memory_context(&base_output)?
        };
        
        let enhanced_output = self.integrate_temporal_context(&base_output, &temporal_context)?;
        self.store_in_memory(&enhanced_output)?;
        
        Ok(enhanced_output)
    }
    
    fn apply_temporal_attention(&mut self, current_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let memory_context = self.memory_bank.get_temporal_context(current_output, 10)?;
        
        if memory_context.is_empty() {
            return Ok(current_output.clone());
        }
        
        let temporal_distances: Vec<usize> = (1..=memory_context.len()).collect();
        
        let temporal_output = self.temporal_attention.forward(
            current_output,
            &memory_context,
            &memory_context,
            &temporal_distances,
        )?;
        
        let combined = ResidualConnection::forward(
            current_output,
            &temporal_output,
            &self.temporal_layer_norm,
        )?;
        
        Ok(combined)
    }
    
    fn compute_temporal_context(
        &self,
        current_output: &DMatrix<f64>,
        previous_states: &[DMatrix<f64>],
        temporal_positions: &[usize],
    ) -> Result<DMatrix<f64>> {
        if previous_states.is_empty() {
            return Ok(DMatrix::zeros(current_output.nrows(), current_output.ncols()));
        }
        
        let seq_len = current_output.nrows();
        let d_model = current_output.ncols();
        
        let combined_states = self.combine_previous_states(previous_states)?;
        let current_positions: Vec<usize> = (0..seq_len).collect();
        
        self.temporal_attention.cross_temporal_attention(
            current_output,
            &combined_states,
            &combined_states,
            &current_positions,
        )
    }
    
    fn retrieve_memory_context(&mut self, query: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let retrieved_memories = self.memory_bank.retrieve(query, 5)?;
        
        if retrieved_memories.is_empty() {
            return Ok(DMatrix::zeros(query.nrows(), query.ncols()));
        }
        
        let temporal_distances: Vec<usize> = (1..=retrieved_memories.len()).collect();
        
        self.temporal_attention.forward(
            query,
            &retrieved_memories,
            &retrieved_memories,
            &temporal_distances,
        )
    }
    
    fn integrate_temporal_context(
        &self,
        base_output: &DMatrix<f64>,
        temporal_context: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>> {
        if temporal_context.norm() < 1e-8 {
            return Ok(base_output.clone());
        }
        
        let weighted_context = temporal_context * &self.memory_integration_weights;
        let integrated = base_output + &weighted_context;
        
        self.temporal_layer_norm.forward(&integrated)
    }
    
    fn combine_previous_states(&self, states: &[DMatrix<f64>]) -> Result<DMatrix<f64>> {
        if states.is_empty() {
            return Err("No previous states to combine".into());
        }
        
        let total_rows: usize = states.iter().map(|s| s.nrows()).sum();
        let d_model = states[0].ncols();
        
        let mut combined = DMatrix::zeros(total_rows, d_model);
        let mut current_row = 0;
        
        for state in states {
            let rows = state.nrows();
            combined.rows_mut(current_row, rows).copy_from(state);
            current_row += rows;
        }
        
        Ok(combined)
    }
    
    fn store_in_memory(&mut self, representation: &DMatrix<f64>) -> Result<()> {
        let pooled = self.global_average_pool(representation);
        self.memory_bank.store(pooled)?;
        Ok(())
    }
    
    fn global_average_pool(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        let seq_len = input.nrows();
        let d_model = input.ncols();
        
        let mut pooled = DMatrix::zeros(1, d_model);
        for j in 0..d_model {
            let sum: f64 = (0..seq_len).map(|i| input[(i, j)]).sum();
            pooled[(0, j)] = sum / seq_len as f64;
        }
        
        pooled
    }
    
    fn initialize_weights(input_dim: usize, output_dim: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt();
        
        DMatrix::from_fn(input_dim, output_dim, |_, _| {
            rng.gen_range(-scale..scale)
        })
    }
    
    pub fn clear_memory(&mut self) {
        self.memory_bank.clear();
    }
    
    pub fn memory_size(&self) -> usize {
        self.memory_bank.size()
    }
}