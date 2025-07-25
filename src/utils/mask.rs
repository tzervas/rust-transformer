use nalgebra::DMatrix;

pub fn create_padding_mask(sequence: &[usize], pad_token: usize) -> DMatrix<bool> {
    let seq_len = sequence.len();
    let mut mask = DMatrix::from_element(seq_len, seq_len, false);
    
    for (i, &token) in sequence.iter().enumerate() {
        if token == pad_token {
            for j in 0..seq_len {
                mask[(i, j)] = true;
                mask[(j, i)] = true;
            }
        }
    }
    
    mask
}

pub fn create_causal_mask(seq_len: usize) -> DMatrix<bool> {
    let mut mask = DMatrix::from_element(seq_len, seq_len, false);
    
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[(i, j)] = true;
        }
    }
    
    mask
}

pub fn combine_masks(mask1: &DMatrix<bool>, mask2: &DMatrix<bool>) -> Result<DMatrix<bool>, Box<dyn std::error::Error + Send + Sync>> {
    if mask1.shape() != mask2.shape() {
        return Err("Masks must have the same shape".into());
    }
    
    Ok(mask1.zip_map(mask2, |a, b| a || b))
}