use super::individual::Individual;

pub fn sorting(popuation: &mut Vec<Individual>) {
    let mut front1 = Vec::new();

    for (pi, p) in popuation.iter_mut().enumerate() {
        p.dominating_idx = Vec::new();
        p.n = 0;
        for (qi, q) in popuation.iter().enumerate() {
            if dominates(p, q) {
                p.dominating_idx.push(qi);
            } else if dominates(q, p) {
                p.n += 1;
            }
        }
        if p.n == 0 {
            // Set rank of p to 1
            front1.push(pi);
        }
    }

    let mut i = 1;
    let mut curr_front = front1;
    while !curr_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for pi in curr_front {
            let p = &popuation[pi];
            for &q_idx in &p.dominating_idx {
                let q = &mut popuation[q_idx];
                q.n -= 1;
                if q.n == 0 {
                    next_front.push(q_idx);
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
