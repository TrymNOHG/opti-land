use crate::genetic::crossover::crossover;
use crate::genetic::mutation::mutate;
use crate::genetic::selection::binary_tournament;
use crate::util::{config::Config, csv::read_csv};

use super::crowding_distance::crowding_distance;
use super::darwin::destroy;
use super::fitness::fitness_population;
use super::initialize_population::init;
use super::non_dominated_sort::fast_non_dominated_sort;
use super::save::save_fitness;

pub fn start(config: &Config) {
    let (bit_length, table) = read_csv(&config.file_name, config).unwrap();
    let mut population = init(config, bit_length);

    fitness_population(&mut population, &table);
    crowding_distance(&mut population);
    fast_non_dominated_sort(&mut population);

    for individual in &population {
        println!(
            "Gene: {}, n features: {}",
            individual.gene, individual.fitness_length
        );
    }
    let mut generation = 1;

    for _ in 0..config.n_generations {
        let mut parents = binary_tournament(&population, &config);

        let mut children = crossover(&mut parents, bit_length);

        mutate(&mut children, bit_length);

        population.extend(children);

        fitness_population(&mut population, &table);
        fast_non_dominated_sort(&mut population);
        crowding_distance(&mut population);
        destroy(&mut population);
        /*
        println!("{}", population[0].fitness_ml_metric);
        let mut n_individuals = 0;
        for individual in &population {
            if individual.rank != 1 {
                break;
            }
            n_individuals += 1;
            println!(
                "generation: {}, fitness_ml: {}, fitness: {}, genome: {:b}",
                generation,
                individual.fitness_ml_metric,
                individual.fitness_length,
                individual.gene
            );
        }
        println!("Number of individuals in first front: {}", n_individuals);*/
        generation += 1;
        if generation % config.log_frequency == 0 {
            println!("Generation: {}", generation);
        }
    }
    let _ = save_fitness(population, &config);
}
