use rand::Rng;

#[derive(Debug)]
pub struct Individual {
    pub gene: u32,
    pub fitness_length: u32,
    pub fitness_ml_metric: f32,
    pub dominating_idx: Vec<usize>,
    pub n: usize,
    pub rank: u16,
    pub crodwing_distance: f32,
}

impl Individual {
    pub fn from(gene: u32) -> Self {
        Individual {
            gene,
            fitness_length: 0,
            fitness_ml_metric: 0.0,
            dominating_idx: Vec::new(),
            n: 0,
            rank: 0,
            crodwing_distance: 0.0,
        }
    }

    pub fn random_new(n_bits: usize) -> Self {
        let mut rng = rand::rng();
        Individual {
            gene: rng.random_range(0..(2 << n_bits - 1)),
            fitness_length: 0,
            fitness_ml_metric: 0.0,
            dominating_idx: Vec::new(),
            n: 0,
            rank: 0,
            crodwing_distance: 0.0,
        }
    }

    pub fn flip(&mut self, idx: usize) {
        let mask = 0 << idx;
        self.gene = self.gene ^ mask;
    }
}
