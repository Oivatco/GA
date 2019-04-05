extern crate rand;

mod get_data;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::vec::Vec;

//static machines: i32 = 0;
//static jobs: i32 = 0;
//static datos: Vec<Vec<i32>> = vec![];

fn main() {
    //Get Data
    let tupla = get_data::run();
    let machines: i32 = tupla.2;
    let jobs: i32 = tupla.1;
    let data: Vec<Vec<i32>> = tupla.0;
    let x =0;

    //Get initial population
    //Return vectores, cargas y bool
    let population = get_initial_population(jobs, machines, &data);

    if population.3 == false {
        let mut population_vectors: Vec<Vec<Vec<i32>>> = population.0;
        let mut population_loads: Vec<Vec<i32>> = population.1;
        let mut cmaxs: Vec<i32> = population.2;
        let size_pop: i32 = 100;
        let n_parents: i32 = 20;
        let max_generations: i32 = 2;
        let mut g: i32 = 0;

        //generación
        while g < max_generations {
            //Slección;
            let tupla2 = order_population_selection(
                &population_loads,
                &cmaxs,
                size_pop,
                machines,
                n_parents,
                jobs,
            );
            let mut ordered_population: Vec<i32> = tupla2.0;
            let mut parents: Vec<i32> = tupla2.1;
            println!(
                "G: {} Best {:?}",
                g, population_loads[ordered_population[0] as usize]
            );
            println!(
                "G: {} Best {:?} \n",
                g, population_loads[ordered_population[1] as usize]
            );

            //Generar Descendencia.
            let tuple3 = crossover_translocation(
                &mut parents,
                machines,
                &mut population_loads,
                &mut population_vectors,
                &ordered_population,
                jobs,
                &data,
                &mut cmaxs,
            );
            let offspring_vectors = tuple3.0;
            let offspring_loads = tuple3.1;
            let final_parents_order = tuple3.2;
            //Remplazo
            let mut p: i32 = 0;
            for par in final_parents_order {
                println!(
                    "lililll, {} {} {} {}",
                    par,
                    offspring_vectors.len(),
                    offspring_loads.len(),
                    cmaxs.len()
                );
                population_vectors[par as usize] = offspring_vectors[p as usize].to_vec();
                cmaxs[par as usize] = offspring_loads[p as usize][machines as usize];
                population_loads[par as usize] = offspring_loads[p as usize].to_vec();
                p += 1;
            }

            //En vista del exito no obtenido, vamos atener qeu regresar una tupla con los padres ordenados
            g += 1;
        }
    } else {
        println!("****************************************** Solved");
    }
}

fn get_initial_population(
    jobs: i32,
    machines: i32,
    datos: &Vec<Vec<i32>>,
) -> (Vec<Vec<Vec<i32>>>, Vec<Vec<i32>>, Vec<i32>, bool) {
    let mut population_vectors: Vec<Vec<Vec<i32>>> = vec![];
    let mut population_loads: Vec<Vec<i32>> = vec![];
    let mut cmaxs: Vec<i32> = vec![];
    let mut base_vector: Vec<i32> = (0..(jobs)).collect();
    let mut cont: i32 = -1;

    for s in 0..100 {
        base_vector.shuffle(&mut thread_rng());
        let mut sol: Vec<Vec<i32>> = vec![];
        let mut load: Vec<i32> = vec![];
        //Inicialiar arreglos
        cont = 0;
        loop {
            cont += 1;
            sol.push(vec![]);
            load.push(0);
            if cont == machines {
                break;
            }
        }
        //Dejamos min() aquí, para no andar pasando de un lado para otro la matriz de datos
        let mut k = 0 as usize;
        loop {
            let b: usize = base_vector[k] as usize;
            let mut m: i32 = load[0] + datos[b][0];
            let mut h = 0 as usize;
            let mut i = 0 as usize;
            loop {
                i += 1;
                if i == machines as usize {
                    break;
                }
                if load[i] + datos[b][i] < m {
                    h = i;
                    m = load[i] + datos[b][i]
                }
            }
            sol[h].push(b as i32);
            load[h] = load[h] + datos[b][h];
            k += 1;
            if k == jobs as usize {
                break;
            }
        }
        let mut k = 1;
        let mut cmax: i32 = load[0];
        let mut cmin: i32 = load[0];
        loop {
            if load[k] > cmax {
                cmax = load[k];
            }
            if load[k] < cmin {
                cmin = load[k];
            }
            k += 1;
            if k == machines as usize {
                break;
            }
        }
        let mut o = 1;
        let mut cont = 0;
        loop {
            if load[o] == cmax {
                cont += 1;
            }
            o += 1;
            if o == machines as usize {
                break;
            }
        }
        //Generamos nuestra lista a ordenar
        //Agregamos a loads max y número de repetidas con max, y agregamos min
        cmaxs.push(cmax);
        load.push(cmax);
        load.push(cont);
        load.push(cmin);
        population_vectors.push(sol);
        population_loads.push(load);
    }

    //El booleano se utiliza para determinar si se encontró o no el óptimo, pero eso se dejará para el final. Leer el csv.
    (population_vectors, population_loads, cmaxs, false)
}

fn order_population_selection(
    population_loads: &Vec<Vec<i32>>,
    cmaxs: &Vec<i32>,
    size_pop: i32,
    machines: i32,
    n_parents: i32,
    jobs: i32,
) -> (Vec<i32>, Vec<i32>) {
    let mut population_index: Vec<i32> = (0..100).collect();
    let mut ordered_population: Vec<i32> = Vec::new();
    let ordered_loads: Vec<i32> = quick_sort(cmaxs.to_vec());

    for ele in 0..size_pop as usize {
        for aka in 0..size_pop as usize {
            if ordered_loads[ele] == cmaxs[aka]
                && ordered_population
                    .iter()
                    .filter(|&n| *n == aka as i32)
                    .count()
                    == 0
            {
                ordered_population.push(aka as i32);
                break;
            }
        }
    }
    //Ordenado en una característica, inicia segunda caracteristica.
    let mut equals_index: Vec<i32> = Vec::new();
    let mut equals_load: Vec<i32> = Vec::new();
    for sol in 1..ordered_population.len() {
        if population_loads[ordered_population[&sol - 1] as usize][machines as usize]
            == population_loads[ordered_population[sol as usize] as usize][machines as usize]
        {
            equals_index.push(ordered_population[&sol - 1]);
            equals_load.push(
                population_loads[ordered_population[&sol - 1] as usize]
                    [machines as usize + 1],
            );
        } else {
            if equals_index.len() != 0 {
                equals_index.push(ordered_population[sol as usize - 1]);
                equals_load.push(
                    population_loads[ordered_population[sol as usize - 1] as usize]
                        [machines as usize + 1],
                );
                let ordered_equals_load: Vec<i32> = quick_sort(equals_load.to_vec());
                let mut final_equals_load: Vec<i32> = Vec::new();
                for ele in 0..ordered_equals_load.len() as usize {
                    for aka in 0..ordered_equals_load.len() as usize {
                        if ordered_equals_load[ele]
                            == population_loads[equals_index[aka as usize] as usize]
                                [machines as usize + 1]
                            && final_equals_load
                                .iter()
                                .filter(|&n| *n == equals_index[aka as usize])
                                .count()
                                == 0
                        {
                            final_equals_load.push(equals_index[aka as usize]);
                            break;
                        }
                    }
                }
                let mut ind = 0;
                for repe in (sol - &equals_index.len())..sol {
                    ordered_population[repe as usize] = final_equals_load[ind as usize];
                    ind += 1;
                }
                equals_index = vec![];
                equals_load = vec![];
            } else {
                equals_index = vec![];
                equals_load = vec![];
            }
        }
    }
    //ordered_population
    //Inicia selección
    let mut sum: f64 = 0.0;
    let mut probs: Vec<f64> = Vec::new();
    for sol in population_loads {
        sum = sum
            + (population_loads[ordered_population[jobs as usize - 1] as usize]
                [machines as usize]
                - sol[machines as usize]) as f64;
    }
    for sol in &ordered_population {
        probs.push(
            (population_loads[ordered_population[jobs as usize - 1] as usize]
                [machines as usize]
                - population_loads[*sol as usize][machines as usize]) as f64
                / sum,
        )
    }
    let mut rng = rand::thread_rng();
    let mut parents: Vec<i32> = Vec::new();
    let mut sum_probs: Vec<f64> = Vec::new();
    for ind in 0..100 as usize {
        if ind == 0 {
            sum_probs.push(probs[ind])
        } else {
            sum_probs.push(sum_probs[ind - 1 as usize] + probs[ind as usize])
        }
    }
    for p in 0..n_parents {
        let pa: f64 = rng.gen::<f64>();
        for ind in 0..100 {
            if pa <= sum_probs[ind as usize] {
                parents.push(ordered_population[ind]);
                break;
            }
        }
    }
    parents.shuffle(&mut thread_rng());
    (ordered_population, parents)
}

fn quick_sort(mut vector: Vec<i32>) -> Vec<i32> {
    if vector.len() > 1 {
        let pivote = vector[0];
        let mut less: Vec<i32> = Vec::new();
        let mut equals: Vec<i32> = Vec::new();
        let mut greater: Vec<i32> = Vec::new();
        for value in vector {
            if value > pivote {
                greater.push(value)
            }
            if value < pivote {
                less.push(value)
            }
            if value == pivote {
                equals.push(value)
            }
        }
        let mut final_vector = Vec::new();
        final_vector.extend(quick_sort(less));
        final_vector.extend(equals);
        final_vector.extend(quick_sort(greater));
        return final_vector;
    } else {
        return vector;
    }
}

fn crossover_translocation(
    parents: &mut Vec<i32>,
    machines: i32,
    population_loads: &mut Vec<Vec<i32>>,
    population_vectors: &mut Vec<Vec<Vec<i32>>>,
    ordered_population: &Vec<i32>,
    jobs: i32,
    data: &Vec<Vec<i32>>,
    cmaxs: &mut Vec<i32>,
) -> (Vec<Vec<Vec<i32>>>, Vec<Vec<i32>>, Vec<i32>) {
    let mut offspring_vectors: Vec<Vec<Vec<i32>>> = Vec::new();
    let mut offspring_loads: Vec<Vec<i32>> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut parents_new_order: Vec<i32> = Vec::new();

    let mut f: usize = 0;
    let mut f1 = 0;
    let mut f2 = 0;
    while f < parents.len() as usize {
        f1 = parents[f];
        f += 1;
        if parents[f] != f1 {
            f2 = parents[f];
        } else {
            let mut fx: usize = &f + 1;
            let mut bn1: bool = false;
            loop {
                if parents[fx] != f1 {
                    let aux = parents[f];
                    parents[f] = parents[fx];
                    parents[fx] = aux;
                    f2 = parents[f];
                    bn1 = true;
                    break;
                }
                fx += 1;
            }
            if bn1 == false {
                f2 = rng.gen_range(0, jobs);
            }
        }
        f += 1;

        //Tenemos dos padres, Todo el show es para garantizar qu eno se vaya el mismo padre jojojo.
        let mut child1: Vec<Vec<i32>> = Vec::new();
        let mut child2: Vec<Vec<i32>> = Vec::new();
        let mut load_c1: Vec<i32> = Vec::new();
        let mut load_c2: Vec<i32> = Vec::new();
        let mut cont_j_c1_remove: Vec<i32> = (0..100).collect();
        let mut cont_j_c2_remove: Vec<i32> = (0..100).collect();

        let mut new_order_parents: Vec<i32> = Vec::new();

        let cp = rng.gen_range(1, machines - 1);
        for gene in 0..machines {
            child1.push(vec![]);
            child2.push(vec![]);
            load_c1.push(0);
            load_c2.push(0);
            if gene < cp {
                //Genes del padre 1 a hijo uno
                for j in &population_vectors[ordered_population[f1 as usize] as usize][gene as usize]
                {
                    if cont_j_c1_remove.iter().filter(|&n| *n == *j).count() == 1 as usize {
                        child1[gene as usize].push(*j);
                        //remove element form vector
                        cont_j_c1_remove.retain(|&x| x != *j);
                    }
                }
                load_c1[gene as usize] =
                    population_loads[ordered_population[f1 as usize] as usize][gene as usize];

                //Genes del padre 2 a hijo dos
                for j in &population_vectors[ordered_population[f2 as usize] as usize][gene as usize]
                {
                    if cont_j_c2_remove.iter().filter(|&n| *n == *j).count() == 1 as usize {
                        child2[gene as usize].push(*j);
                        //remove element form vector
                        cont_j_c2_remove.retain(|&x| x != *j);
                    }
                }
                load_c2[gene as usize] =
                    population_loads[ordered_population[f2 as usize] as usize][gene as usize];
            } else {
                let mut load1: i32 = 0;
                //Genes del padre 2 a padre uno
                for j in
                    &population_vectors[ordered_population[f2 as usize] as usize][gene as usize]
                {
                    if cont_j_c1_remove.iter().filter(|&n| *n == *j).count() == 1 as usize {
                        child1[gene as usize].push(*j);
                        cont_j_c1_remove.retain(|&x| x != *j);
                        load1 = load1 + data[*j as usize][gene as usize];
                    }
                }
                load_c1[gene as usize] = load1;
                let mut load2: i32 = 0;
                //Genes del padre 1 a padre dos
                for j in
                    &population_vectors[ordered_population[f1 as usize] as usize][gene as usize]
                {
                    if cont_j_c2_remove.iter().filter(|&n| *n == *j).count() == 1 as usize {
                        child2[gene as usize].push(*j);
                        //remove element form vector
                        cont_j_c2_remove.retain(|&x| x != *j);
                        load2 = load2 + data[*j as usize][gene as usize];
                    }
                }
                load_c2[gene as usize] = load2;
            }
        }
        //Reasignación de trabajos liberados
        cont_j_c1_remove.shuffle(&mut thread_rng());
        cont_j_c2_remove.shuffle(&mut thread_rng());
        for j in cont_j_c1_remove {
            let mut m: i32 = load_c1[0] + data[j as usize][0];
            let mut h = 0 as usize;
            for i in 1..machines {
                if load_c1[i as usize] + data[j as usize][i as usize] < m {
                    h = i as usize;
                    m = load_c1[i as usize] + data[j as usize][i as usize]
                }
            }
            child1[h as usize].push(j as i32);
            load_c1[h as usize] = load_c1[h as usize] + data[j as usize][h as usize];
        }

        for j in cont_j_c2_remove {
            let mut m: i32 = load_c2[0] + data[j as usize][0];
            let mut h = 0 as usize;
            for i in 1..machines {
                if load_c2[i as usize] + data[j as usize][i as usize] < m {
                    h = i as usize;
                    m = load_c2[i as usize] + data[j as usize][i as usize]
                }
            }
            child2[h as usize].push(j as i32);
            load_c2[h as usize] = load_c2[h as usize] + data[j as usize][h as usize];
        }

        //Evaluación para ver si la encontré cruza
        let mut cmax1: i32 = load_c1[0];
        let mut cmax2: i32 = load_c2[0];
        for k in 1..machines {
            if load_c1[k as usize] > cmax1 {
                cmax1 = load_c1[k as usize];
            }
            if load_c2[k as usize] > cmax2 {
                cmax2 = load_c2[k as usize];
            }
        }
        let mut cont1 = 0;
        let mut cont2 = 0;
        for o in 1..machines {
            if load_c1[o as usize] == cmax1 {
                cont1 += 1;
            }
            if load_c2[o as usize] == cmax2 {
                cont2 += 1;
            }
        }
        load_c1.push(cmax1);
        load_c1.push(cont1);
        load_c2.push(cmax2);
        load_c2.push(cont2);
        //Aqui termina una cruza.
        //Comienza la mutación.

        let base: i32 = (jobs / machines) / 2;
        let mut released_jobs_c1: Vec<i32> = Vec::new();
        let mut ind: i32 = 0;
        let mut gene_load: i32 = 0;

        for gene in &mut child1 {
            if gene.len() < base as usize {
                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c1.push(gene[ind as usize]);
                load_c1[gene_load as usize] = load_c1[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);
            } else {
                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c1.push(gene[ind as usize]);
                load_c1[gene_load as usize] = load_c1[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);

                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c1.push(gene[ind as usize]);
                load_c1[gene_load as usize] = load_c1[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);
            }
            gene_load += 1;
        }
        for j in &released_jobs_c1 {
            if rng.gen::<f64>() <= 0.005 {
                //Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
                let mut shuffle_machines: Vec<i32> = (0..machines).collect();
                shuffle_machines.shuffle(&mut thread_rng());
                let mut bn: bool = true;
                for i in shuffle_machines {
                    if load_c1[i as usize] + data[*j as usize][i as usize]
                        < load_c1[machines as usize]
                    {
                        child1[i as usize].push(*j as i32);
                        load_c1[i as usize] = load_c1[i as usize] + data[*j as usize][i as usize];
                        bn = false;
                    }
                }
                if bn == true {
                    let random_i = rng.gen_range(0, machines);
                    child1[random_i as usize].push(*j as i32);
                    load_c1[random_i as usize] =
                        load_c1[random_i as usize] + data[*j as usize][random_i as usize];
                }
            } else {
                let mut m: i32 = load_c1[0] + data[*j as usize][0];
                let mut h = 0 as usize;
                for i in 1..machines {
                    if load_c1[i as usize] + data[*j as usize][i as usize] < m {
                        h = i as usize;
                        m = load_c1[i as usize] + data[*j as usize][i as usize]
                    }
                }
                child1[h as usize].push(*j as i32);
                load_c1[h as usize] = load_c1[h as usize] + data[*j as usize][h as usize];
            }
        }
        //Secound mutation
        let mut released_jobs_c2: Vec<i32> = Vec::new();
        gene_load = 0;
        for gene in &mut child2 {
            if gene.len() < base as usize {
                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c2.push(gene[ind as usize]);
                load_c2[gene_load as usize] = load_c2[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);
            } else {
                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c2.push(gene[ind as usize]);
                load_c2[gene_load as usize] = load_c2[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);

                ind = rng.gen_range(0, gene.len() as i32);
                released_jobs_c2.push(gene[ind as usize]);
                load_c2[gene_load as usize] = load_c2[gene_load as usize]
                    - data[gene[ind as usize] as usize][gene_load as usize];
                gene.remove(ind as usize);
            }
            gene_load += 1;
        }

        for j in &released_jobs_c2 {
            if rng.gen::<f64>() <= 0.005 {
                //Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
                let mut shuffle_machines: Vec<i32> = (0..machines).collect();
                shuffle_machines.shuffle(&mut thread_rng());
                let mut bn: bool = true;
                for i in shuffle_machines {
                    if load_c2[i as usize] + data[*j as usize][i as usize]
                        < load_c2[machines as usize]
                    {
                        child2[i as usize].push(*j as i32);
                        load_c2[i as usize] = load_c2[i as usize] + data[*j as usize][i as usize];
                        bn = false;
                    }
                }
                if bn == true {
                    let random_i = rng.gen_range(0, machines);
                    child2[random_i as usize].push(*j as i32);
                    load_c2[random_i as usize] =
                        load_c2[random_i as usize] + data[*j as usize][random_i as usize];
                }
            } else {
                let mut m: i32 = load_c2[0] + data[*j as usize][0];
                let mut h = 0 as usize;
                for i in 1..machines {
                    if load_c2[i as usize] + data[*j as usize][i as usize] < m {
                        h = i as usize;
                        m = load_c2[i as usize] + data[*j as usize][i as usize]
                    }
                }
                child2[h as usize].push(*j as i32);
                load_c2[h as usize] = load_c2[h as usize] + data[*j as usize][h as usize];
            }
        }

        //Evaluación para ver si la encontré cruza
        cmax1 = load_c1[0];
        cmax2 = load_c2[0];
        for k in 1..machines {
            if load_c1[k as usize] > cmax1 {
                cmax1 = load_c1[k as usize];
            }
            if load_c2[k as usize] > cmax2 {
                cmax2 = load_c2[k as usize];
            }
        }
        cont1 = 0;
        cont2 = 0;
        for o in 1..machines {
            if load_c1[o as usize] == cmax1 {
                cont1 += 1;
            }
            if load_c2[o as usize] == cmax2 {
                cont2 += 1;
            }
        }
        load_c1[machines as usize] = cmax1;
        load_c1[machines as usize + 1] = cont1;
        load_c2[machines as usize] = cmax2;
        load_c2[machines as usize + 1] = cont2;
        offspring_vectors.push(child1);
        offspring_vectors.push(child2);
        offspring_loads.push(load_c1);
        offspring_loads.push(load_c2);

        parents_new_order.push(f1);
        parents_new_order.push(f2);
    }
    (offspring_vectors, offspring_loads, parents_new_order)
}

fn is_feasible(vector: Vec<Vec<i32>>, jobs: i32) -> bool {
    let mut sum: i32 = 0;
    for gene in vector {
        sum = sum + gene.len() as i32;
    }
    if sum == jobs {
        return true;
    } else {
        return false;
    }
}
