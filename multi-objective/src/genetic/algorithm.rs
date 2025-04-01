use crate::util::config::Config;

use super::initialize_population::init;

pub fn start(config: &Config) {
    let mut population = init(&config);
}
