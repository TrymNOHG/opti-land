use std::error::Error;
use std::fs::File;
use std::io::Write;

use super::individual::Individual;

pub fn save_fitness(population: Vec<Individual>) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("fitness_log.csv")?;
    writeln!(file, "fitness_length,fitness_ml_metric")?;
    for individual in population {
        if individual.rank != 1 {
            break;
        }
        writeln!(
            file,
            "{},{}",
            individual.fitness_length, individual.fitness_ml_metric
        )?;
    }
    Ok(())
}
