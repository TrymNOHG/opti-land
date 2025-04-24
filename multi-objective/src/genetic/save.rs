use std::error::Error;
use std::fmt::format;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::util::config::Config;

use super::individual::Individual;

pub fn save_fitness(population: Vec<Individual>, config: &Config) -> Result<(), Box<dyn Error>> {
    let file_stem = Path::new(&config.file_name)
        .file_stem() // gets "svm_feature" as OsStr
        .and_then(|s| s.to_str()); // converts OsStr to Option<&str>

    if let Some(name) = file_stem {
        println!("File name without extension: {}", name);
    } else {
        println!("Could not extract file name");
    }
    let mut file = File::create(format!("fitness_{}.csv", file_stem.unwrap()))?;
    let mut file2 = File::create(format!("fitness_{}_dominated.csv", file_stem.unwrap()))?;

    writeln!(file, "fitness_length,fitness_ml_metric")?;
    writeln!(file2, "fitness_length,fitness_ml_metric")?;
    for individual in population {
        if individual.rank != 1 {
            writeln!(
                file2,
                "{},{}",
                individual.fitness_length, individual.fitness_ml_metric
            )?;
        } else {
            writeln!(
                file,
                "{},{}",
                individual.fitness_length, individual.fitness_ml_metric
            )?;
        }
    }
    Ok(())
}
