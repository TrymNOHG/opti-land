use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::StringRecord;

use super::config::Config;

pub fn read_csv<P: AsRef<Path>>(
    filename: P,
    config: &Config,
) -> Result<(usize, Vec<f32>), Box<dyn Error>> {
    let mut table = Vec::new();
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);
    let headers = rdr.headers()?.clone();

    let fitness_idx = headers
        .iter()
        .position(|h| h == config.fitness_name)
        .unwrap();
    let ind_idx = headers.iter().position(|h| h == config.ind).unwrap();

    let mut bit_length = 0;

    let mut i = 0;
    for result in rdr.records() {
        let record: StringRecord = result?;
        bit_length = record.get(ind_idx).unwrap().len();
        table.push(record.get(fitness_idx).unwrap().parse().unwrap());
        i += 1;
    }

    println!("Len of record: {}", i);

    Ok((bit_length, table))
}
