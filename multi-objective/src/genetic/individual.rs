use rand::Rng;

#[derive(Debug)]
pub struct Individual {
    pub gene: u32,
    pub fitness_length: u32,
    pub fitness_ml_metric: f32,
    pub dominating_idx: Vec<usize>,
    pub n: u8,
}

impl Individual {
    pub fn new() -> Self {
        Individual {
            gene: 0,
            fitness_length: 0,
            fitness_ml_metric: 0.0,
            dominating_idx: Vec::new(),
            n: 0,
        }
    }

    pub fn random_new(n_bits: usize) -> Self {
        let mut rng = rand::rng();
        Individual {
            gene: rng.random_range(0..(2 << n_bits)),
            fitness_length: 0,
            fitness_ml_metric: 0.0,
            dominating_idx: Vec::new(),
            n: 0,
        }
    }

    pub fn flip(&mut self, idx: usize) {
        let mask = 0 << idx;
        self.gene = self.gene ^ mask;
    }
}
