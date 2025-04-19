use rand::rng;
use rand::seq::IndexedRandom;

use crate::util::config::Config;

use super::individual::Individual;

pub fn binary_tournament(population: &Vec<Individual>, config: &Config) -> Vec<Individual> {
    let mut rng = rng();
    let mut parents: Vec<Individual> = Vec::new();
    for _ in 0..config.population_size {
        let p1 = population.choose(&mut rng).unwrap();
        let p2 = population.choose(&mut rng).unwrap();

        let parent;
        if p1.rank == p2.rank {
            if p1.crodwing_distance > p2.crodwing_distance {
                parent = p1;
            } else {
                parent = p2;
            }
        } else if p1.rank < p2.rank {
            parent = p1;
        } else {
            parent = p2;
        }
        parents.push(Individual::from(parent.gene));
    }
    parents
}
