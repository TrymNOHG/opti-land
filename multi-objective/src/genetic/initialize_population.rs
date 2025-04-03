use crate::Config;

use super::individual::Individual;

pub fn init(config: &Config, but_length: usize) -> Vec<Individual> {
    let mut population = Vec::new();
    for _ in 0..config.population_size {
        population.push(Individual::random_new(but_length))
    }
    population
}
