use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::Iterator;
use std::vec::Vec;

pub fn run() -> (Vec<Vec<i32>>, i32, i32) {
    let file = File::open("111.txt").expect("ñooo");
    let buf_reader = BufReader::new(file);
    let mut c: i32 = -1;
    //let mut numbers: Vec = vec![];

    let mut data: Vec<Vec<i32>> = vec![];
    let mut jobs: i32 = 0;
    let mut machines: i32 = 0;
    for line in buf_reader.lines() {
        c += 1;
        let tupla = String::from(line.unwrap());
        if c == 0 {
            let vec: Vec<&str> = tupla.split(" ").collect();
            let numbers: Vec<i32> = vec.iter().filter_map(|x| x.parse().ok()).collect();

            jobs = numbers[0];
            machines = numbers[1];

        //println!("kaaaa {:?}", numbers);
        } else if c > 1 {
            let vec: Vec<&str> = tupla.split("\t").collect();
            let numbers: Vec<i32> = vec.iter().filter_map(|x| x.parse().ok()).collect();
            //Adapted
            let mut f_vector: Vec<i32> = vec![];
            let mut i = 0;
            loop {
                if i % 2 != 0 {
                    f_vector.push(numbers[i])
                }
                i += 1;
                //println!("Vector {:?}", &f_vector);
                if i == numbers.len() {
                    break;
                }
            }
            //println!("Ni siquierea acá");

            //Fin Adapted
            data.push(f_vector);
            //println!("Sooooo {:?}",numbers);
        }
    }
    /*for x in data{
        println!("{:?}", x);
    }*/
    (data, jobs, machines)
}
