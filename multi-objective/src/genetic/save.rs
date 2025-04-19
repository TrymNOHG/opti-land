use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn save_fitness(fitness_length: f64, fitness_ml_metric: f64) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("fitness_log.csv")?;
    writeln!(file, "fitness_length,fitness_ml_metric")?;
    writeln!(file, "{},{}", fitness_length, fitness_ml_metric)?;
    Ok(())
}
