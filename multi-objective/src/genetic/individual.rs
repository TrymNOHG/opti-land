use rand::Rng;

#[derive(Debug)]
pub struct Individual {
    pub gene: u32,
    pub fitness_x: f32,
    pub fitness_y: f32,
}

impl Individual {
    pub fn new() -> Self {
        Individual {
            gene: 0,
            fitness_x: 0.0,
            fitness_y: 0.0,
        }
    }

    pub fn random_new() -> Self {
        let mut rng = rand::rng();
        Individual {
            gene: rng.random(),
            fitness_x: 0.0,
            fitness_y: 0.0,
        }
    }

    pub fn flip(&mut self, idx: usize) {
        let mask = 0 << idx;
        self.gene = self.gene ^ mask;
    }
}
