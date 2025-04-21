use rand::{Rng, rng};

use super::individual::Individual;

pub fn mutate(population: &mut Vec<Individual>, n_bits: usize) {
    let mut rng = rng();
    for i in 0..population.len() {
        let idx = rng.random_range(0..n_bits);
        population[i].flip(idx);
    }
}
