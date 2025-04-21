use super::individual::Individual;

pub fn fast_non_dominated_sort(population: &mut Vec<Individual>) {
    let len = population.len();
    let mut front1 = Vec::new();

    // First calculate domination relationships
    for p_i in 0..len {
        population[p_i].dominating_idx.clear();
        population[p_i].n = 0;

        for q_i in 0..len {
            let p = &population[p_i];
            let q = &population[q_i];

            if dominates(&p, &q) {
                population[p_i].dominating_idx.push(q_i);
            } else if dominates(&q, &p) {
                population[p_i].n += 1;
            }
        }

        if population[p_i].n == 0 {
            front1.push(p_i);
            population[p_i].rank = 1;
        }
    }

    // Second assign fronts
    let mut i = 1;
    let mut curr_front = front1;

    while !curr_front.is_empty() {
        let mut next_front = Vec::new();

        for &p_i in &curr_front {
            let s = population[p_i].dominating_idx.clone();
            for &q_i in &s {
                population[q_i].n -= 1;
                if population[q_i].n == 0 {
                    population[q_i].rank = i + 1;
                    next_front.push(q_i);
                }
            }
        }

        i += 1;
        curr_front = next_front;
    }
}

fn dominates(individual1: &Individual, individual2: &Individual) -> bool {
    individual1.fitness_length < individual2.fitness_length
        && individual1.fitness_ml_metric < individual2.fitness_ml_metric
}
