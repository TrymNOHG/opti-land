mod genetic;
mod util;

use genetic::algorithm::start;
use util::{config::Config, csv::read_csv};

fn main() {
    let config = Config::new("./config/config.yaml");
    start(&config);
}
