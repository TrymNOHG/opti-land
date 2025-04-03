use super::individual::Individual;

pub fn fitness_population(population: &mut Vec<Individual>, table: &Vec<f32>) {
    for individual in population {
        individual.fitness_length = individual.gene.count_zeros();
        individual.fitness_ml_metric = *table.get(individual.gene as usize).unwrap();
    }
}
