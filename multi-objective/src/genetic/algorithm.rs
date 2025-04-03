use crate::util::{config::Config, csv::read_csv};

use super::fitness::fitness_population;
use super::initialize_population::init;

pub fn start(config: &Config) {
    let (bit_length, table) = read_csv(&config.file_name, config).unwrap();
    let mut population = init(config, bit_length);

    fitness_population(&mut population, &table);

    for i in 0..config.n_generations {}
}
