use std::collections::HashMap;

use super::individual::Individual;

pub fn fitness_population(population: &mut Vec<Individual>, table: &HashMap<u32, f32>) {
    for individual in population {
        let gene = individual.gene;
        individual.fitness_length = gene.count_ones();
        individual.fitness_ml_metric = *table
            .get(&gene)
            .expect(&format!("No entry from gene: {}", &gene));
    }
}
