use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::Iterator;
use std::vec::Vec;

pub fn run() -> (Vec<Vec<i32>>, i32, i32) {
    let file = File::open("111.txt").expect("ñooo");
    let mut buf_reader = BufReader::new(file);

    // first line contains number of jobs, number of machines
    let mut first_line = String::new();

    buf_reader.read_line(&mut first_line).expect("No hay ni una línea");

    let numbers: Vec<i32> = first_line.split_whitespace().map(|x| x.parse().unwrap()).collect();

    let jobs = numbers[0];
    let machines = numbers[1];
    let mut data = Vec::with_capacity(jobs as usize);

    // second line is useless
    buf_reader.read_line(&mut first_line).expect("No hay segunda línea");

    for line in buf_reader.lines().map(|l| l.unwrap()) {
        let vec: Vec<i32> = line
            .trim()
            .split_whitespace()
            .enumerate()
            .filter(|(i, e)| i%2 == 1)
            .map(|(i, e)| e.parse().unwrap())
            .collect();

        data.push(vec);
    }

    (data, jobs, machines)
}
