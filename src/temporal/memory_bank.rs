use nalgebra::DMatrix;
use std::collections::VecDeque;
use crate::Result;

#[derive(Clone)]
pub struct MemoryConfig {
    pub max_memory_size: usize,
    pub memory_decay_factor: f64,
    pub retention_threshold: f64,
    pub compression_ratio: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_size: 1000,
            memory_decay_factor: 0.95,
            retention_threshold: 0.1,
            compression_ratio: 0.5,
        }
    }
}

pub struct MemoryEntry {
    pub representation: DMatrix<f64>,
    pub timestamp: usize,
    pub importance_score: f64,
    pub access_count: usize,
}

impl MemoryEntry {
    pub fn new(representation: DMatrix<f64>, timestamp: usize) -> Self {
        Self {
            representation,
            timestamp,
            importance_score: 1.0,
            access_count: 0,
        }
    }
    
    pub fn update_importance(&mut self, decay_factor: f64, current_timestamp: usize) {
        let time_decay = decay_factor.powi((current_timestamp - self.timestamp) as i32);
        let access_boost = (self.access_count as f64).ln().max(0.0) * 0.1;
        self.importance_score = self.importance_score * time_decay + access_boost;
        self.access_count += 1;
    }
}

pub struct MemoryBank {
    memories: VecDeque<MemoryEntry>,
    config: MemoryConfig,
    current_timestamp: usize,
    compressed_memories: Vec<DMatrix<f64>>,
}

impl MemoryBank {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            memories: VecDeque::new(),
            config,
            current_timestamp: 0,
            compressed_memories: Vec::new(),
        }
    }
    
    pub fn store(&mut self, representation: DMatrix<f64>) -> Result<()> {
        self.current_timestamp += 1;
        let entry = MemoryEntry::new(representation, self.current_timestamp);
        
        self.memories.push_back(entry);
        
        if self.memories.len() > self.config.max_memory_size {
            self.evict_memories()?;
        }
        
        Ok(())
    }
    
    pub fn retrieve(&mut self, query: &DMatrix<f64>, k: usize) -> Result<Vec<DMatrix<f64>>> {
        let similarities = self.compute_similarities(query)?;
        let mut indexed_similarities: Vec<(usize, f64)> = similarities
            .into_iter()
            .enumerate()
            .collect();
        
        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut retrieved = Vec::new();
        for (idx, _) in indexed_similarities.into_iter().take(k) {
            if let Some(memory) = self.memories.get_mut(idx) {
                memory.update_importance(self.config.memory_decay_factor, self.current_timestamp);
                retrieved.push(memory.representation.clone());
            }
        }
        
        Ok(retrieved)
    }
    
    pub fn get_temporal_context(&self, query: &DMatrix<f64>, max_distance: usize) -> Result<Vec<DMatrix<f64>>> {
        let current_time = self.current_timestamp;
        let mut context = Vec::new();
        
        for memory in self.memories.iter().rev() {
            let distance = current_time - memory.timestamp;
            if distance <= max_distance && memory.importance_score > self.config.retention_threshold {
                context.push(memory.representation.clone());
            }
        }
        
        Ok(context)
    }
    
    pub fn compress_old_memories(&mut self) -> Result<()> {
        let compression_threshold = (self.config.max_memory_size as f64 * self.config.compression_ratio) as usize;
        
        if self.memories.len() <= compression_threshold {
            return Ok(());
        }
        
        let old_memories: Vec<_> = self.memories
            .drain(..compression_threshold)
            .collect();
        
        if !old_memories.is_empty() {
            let compressed = self.compress_memories(&old_memories)?;
            self.compressed_memories.push(compressed);
        }
        
        Ok(())
    }
    
    fn evict_memories(&mut self) -> Result<()> {
        self.update_all_importance_scores();
        
        let mut indexed_memories: Vec<(usize, f64)> = self.memories
            .iter()
            .enumerate()
            .map(|(i, m)| (i, m.importance_score))
            .collect();
        
        indexed_memories.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let num_to_remove = self.memories.len() - self.config.max_memory_size + 1;
        let mut indices_to_remove: Vec<_> = indexed_memories
            .into_iter()
            .take(num_to_remove)
            .map(|(i, _)| i)
            .collect();
        
        indices_to_remove.sort_by(|a, b| b.cmp(a));
        
        for &idx in &indices_to_remove {
            self.memories.remove(idx);
        }
        
        Ok(())
    }
    
    fn update_all_importance_scores(&mut self) {
        for memory in &mut self.memories {
            memory.update_importance(self.config.memory_decay_factor, self.current_timestamp);
        }
    }
    
    fn compute_similarities(&self, query: &DMatrix<f64>) -> Result<Vec<f64>> {
        let mut similarities = Vec::new();
        
        for memory in &self.memories {
            let similarity = self.cosine_similarity(query, &memory.representation)?;
            similarities.push(similarity);
        }
        
        Ok(similarities)
    }
    
    fn cosine_similarity(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<f64> {
        if a.shape() != b.shape() {
            return Err("Matrix shapes must match for cosine similarity".into());
        }
        
        let dot_product = a.dot(b);
        let norm_a = a.norm();
        let norm_b = b.norm();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }
    
    fn compress_memories(&self, memories: &[MemoryEntry]) -> Result<DMatrix<f64>> {
        if memories.is_empty() {
            return Err("Cannot compress empty memory set".into());
        }
        
        let first_shape = memories[0].representation.shape();
        let mut sum = DMatrix::zeros(first_shape.0, first_shape.1);
        let mut total_importance = 0.0;
        
        for memory in memories {
            let weight = memory.importance_score;
            sum += &memory.representation.map(|x| x * weight);
            total_importance += weight;
        }
        
        if total_importance > 0.0 {
            sum = sum.map(|x| x / total_importance);
        }
        
        Ok(sum)
    }
    
    pub fn clear(&mut self) {
        self.memories.clear();
        self.compressed_memories.clear();
        self.current_timestamp = 0;
    }
    
    pub fn size(&self) -> usize {
        self.memories.len()
    }
}