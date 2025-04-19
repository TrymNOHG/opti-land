use super::individual::Individual;

pub fn destroy(population: &mut Vec<Individual>) {
    assert!(population.is_sorted_by_key(|i| i.rank));
    let pop_len = population.len();
    let front = population[pop_len / 2].rank;
    let mut front_indices = Vec::new();
    for i in 0..population.len() {
        if population[i].rank == front {
            front_indices.push(i);
        }
    }
    let mut front_individuals: Vec<Individual> = population.drain(front_indices[0]..).collect();
    front_individuals.drain(front_indices.len() - 1..);
    front_individuals.sort_by(|i1, i2| i2.crodwing_distance.total_cmp(&i1.crodwing_distance));
    let individuals_to_add = (pop_len / 2) - population.len();
    population.extend(front_individuals.drain(0..individuals_to_add));
}
