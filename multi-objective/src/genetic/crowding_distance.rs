use super::individual::Individual;

pub fn crowding_distance(population: &mut Vec<Individual>) {
    population.sort_by_key(|i| i.rank);
    let mut curr_rank = 1;
    let mut curr_front = Vec::new();
    for individual in population {
        if individual.rank > curr_rank {
            curr_rank = individual.rank;
            curr_front.push(individual);
            crowding_distance_front(&mut curr_front);
            curr_front.clear();
        } else {
            curr_front.push(individual);
        }
    }
}

fn crowding_distance_front(front: &mut Vec<&mut Individual>) {
    for individual in front.iter_mut() {
        individual.crodwing_distance = 0.0;
    }

    let len = front.len();

    // Objective 1:
    front.sort_by_key(|i| i.fitness_length);
    front[0].crodwing_distance = f32::MAX;
    front[len - 1].crodwing_distance = f32::MAX;
    for i in 1..front.len() - 1 {
        front[i].crodwing_distance = front[i].crodwing_distance
            + (front[i + 1].fitness_length - front[i - 1].fitness_length) as f32
                / (front[len - 1].fitness_length - front[0].fitness_length) as f32
    }

    // Objective 2:
    front.sort_by(|p1, p2| p2.fitness_ml_metric.total_cmp(&p1.fitness_ml_metric));
    front[0].crodwing_distance = f32::MAX;
    front[len - 1].crodwing_distance = f32::MAX;
    for i in 1..front.len() - 1 {
        front[i].crodwing_distance = front[i].crodwing_distance
            + (front[i + 1].fitness_ml_metric - front[i - 1].fitness_ml_metric)
                / (front[len - 1].fitness_ml_metric - front[0].fitness_ml_metric)
    }
}
