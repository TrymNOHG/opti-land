use rand::{Rng, rng};

use super::individual::Individual;

/// One point crossover
pub fn crossover(parents: &mut Vec<Individual>, n_bits: usize) -> Vec<Individual> {
    let mut rng = rng();
    let mut children: Vec<Individual> = Vec::new();
    while parents.len() > 1 {
        let p1 = parents.pop().unwrap().gene;
        let p2 = parents.pop().unwrap().gene;
        //println!("P1 gene: {:06b}", p1);
        //println!("P2 gene: {:06b}", p2);

        let crossover_point = rng.random_range(0..n_bits + 1);
        //println!("Crossover point: {}", crossover_point);

        let low_bits = n_bits - crossover_point;
        let low_mask = (1 << low_bits) - 1;
        //println!("low_mask: {:06b}", low_mask);
        let high_mask = ((1 << n_bits) - 1) & !low_mask;
        //println!("high_mask: {:06b}", high_mask);

        //println!("(p1 & high_mask) {:06b}", (p1 & high_mask));
        //println!("(p2 & high_mask) {:06b}", (p2 & high_mask));
        //println!("(p1 & low_mask) {:06b}", (p1 & low_mask));
        //println!("(p2 & low_mask) {:06b}", (p2 & low_mask));
        let child1_gene = (p1 & high_mask) | (p2 & low_mask);
        let child2_gene = (p2 & high_mask) | (p1 & low_mask);

        children.push(Individual::from(child1_gene));
        children.push(Individual::from(child2_gene));
    }
    children
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover() {
        let bitstring_1 = u32::from_str_radix("111111", 2).unwrap();
        let parent_1 = Individual::from(bitstring_1);
        let bitstring_2 = u32::from_str_radix("000000", 2).unwrap();
        let parent_2 = Individual::from(bitstring_2);

        let mut parents = vec![parent_1, parent_2];

        let children = crossover(&mut parents, 6);

        println!("Children genes:");
        for child in children {
            println!("{:06b}", child.gene);
        }
    }
}
