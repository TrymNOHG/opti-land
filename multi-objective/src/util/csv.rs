use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::StringRecord;

use super::config::Config;

pub struct Entry {
    fitness_x: f32,
    fitness_y: f32,
}

pub fn read_csv<P: AsRef<Path>>(
    filename: P,
    config: &Config,
) -> Result<(usize, Vec<Entry>), Box<dyn Error>> {
    let table = Vec::new();
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);
    let headers = rdr.headers()?.clone();

    let fitness_1_idx = headers.iter().position(|h| h == config.fitness_name_1);
    let fitness_2_idx = headers.iter().position(|h| h == config.fitness_name_2);
    let ind_idx = headers.iter().position(|h| h == config.ind);

    let mut bit_length = 0;

    for result in rdr.records() {
        let record: StringRecord = result?;
        bit_length = record.get(ind_idx).iter().len();
        table.push(Entry {
            fitness_x: record.get(fitness_1_idx).unwrap().parse().unwrap(),
            fitness_y: record.get(fitness_2_idx).unwrap().parse().unwrap(),
        })
    }

    Ok((bit_length, table))
}
