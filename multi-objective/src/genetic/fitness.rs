use super::individual::Individual;

pub fn fitness_population(population: &mut Vec<Individual>, table: &Vec<f32>) {
    for individual in population {
        let gene = individual.gene;
        individual.fitness_length = gene.count_ones();
        individual.fitness_ml_metric = *table.get(gene as usize).unwrap();
    }
}
