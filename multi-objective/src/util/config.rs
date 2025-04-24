use serde::Deserialize;
use serde_yaml;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    // Logging
    pub log_frequency: i32,
    // File
    pub file_name: String,
    pub ind: String,
    pub fitness_name: String,
    // Hyperparameters
    pub population_size: usize,
    pub n_generations: i32,
}

impl Config {
    pub fn new(path: &str) -> Self {
        let file_content: String = fs::read_to_string(path).expect("Failed to read file");
        serde_yaml::from_str(&file_content).unwrap()
    }
}
