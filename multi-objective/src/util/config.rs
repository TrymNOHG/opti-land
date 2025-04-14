use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum ScrambleFN {
    Delete,
    Keep,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    // Run time
    pub run_time: i32,
    // Logging
    pub log_frequency: i32,
    // File
    pub file_name: String,
    pub ind: String,
    pub fitness_name: String,
    pub mimizie_fitness_name: bool,
    // Hyperparameters
    pub population_size: usize,
    pub n_generations: i32,
    pub tournament_size: i32,
}

impl Config {
    pub fn new(path: &str) -> Self {
        let file_content: String = fs::read_to_string(path).expect("Failed to read file");
        serde_yaml::from_str(&file_content).unwrap()
    }
}
