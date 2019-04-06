extern crate rand;

mod get_data;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::vec::Vec;

//static machines = 0;
//static jobs = 0;
//static datos = vec![];

fn main() {
    //Get Data
    let (data, jobs, machines) = get_data::run();

    //Get initial population
    //Return vectores, cargas y bool
    let population = get_initial_population(jobs, machines, &data);

    if population.3 == false {
        let mut population_vectors = population.0;
        let mut population_loads = population.1;
        let mut cmaxs = population.2;
        let size_pop = 100;
        let n_parents = 20;
        let max_generations = 2;
        let mut g = 0;

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
            let mut ordered_population = tupla2.0;
            let mut parents = tupla2.1;
            println!(
                "G: {} Best {:?}",
                g, population_loads[ordered_population[0]]
            );
            println!(
                "G: {} Best {:?} \n",
                g, population_loads[ordered_population[1]]
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
            let mut p = 0;
            for par in final_parents_order {
                println!(
                    "lililll, {} {} {} {}",
                    par,
                    offspring_vectors.len(),
                    offspring_loads.len(),
                    cmaxs.len()
                );
                population_vectors[par] = offspring_vectors[p].to_vec();
                cmaxs[par] = offspring_loads[p][machines];
                population_loads[par] = offspring_loads[p].to_vec();
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
    jobs: usize,
    machines: usize,
    datos: &Vec<Vec<i32>>,
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<i32>>, Vec<i32>, bool) {
    let mut population_vectors = vec![];
    let mut population_loads = vec![];
    let mut cmaxs = vec![];
    let mut base_vector: Vec<usize> = (0..jobs).collect();

    for s in 0..100 {
        base_vector.shuffle(&mut thread_rng());
        let mut sol = vec![];
        let mut load = vec![];

        // Inicialiar arreglos
        let mut cont = 0;
        loop {
            cont += 1;
            sol.push(vec![]);
            load.push(0);
            if cont == machines {
                break;
            }
        }

        //Dejamos min() aquí, para no andar pasando de un lado para otro la matriz de datos
        let mut k = 0;
        loop {
            let b = base_vector[k];
            let mut m = load[0] + datos[b][0];
            let mut h = 0;
            let mut i = 0;
            loop {
                i += 1;
                if i == machines{
                    break;
                }
                if load[i] + datos[b][i] < m {
                    h = i;
                    m = load[i] + datos[b][i]
                }
            }
            sol[h].push(b);
            load[h] = load[h] + datos[b][h];
            k += 1;
            if k == jobs{
                break;
            }
        }
        let mut k = 1;
        let mut cmax = load[0];
        let mut cmin = load[0];
        loop {
            if load[k] > cmax {
                cmax = load[k];
            }
            if load[k] < cmin {
                cmin = load[k];
            }
            k += 1;
            if k == machines{
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
            if o == machines{
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
    size_pop: usize,
    machines: usize,
    n_parents: i32,
    jobs: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut population_index: Vec<i32> = (0..100).collect();
    let mut ordered_population = Vec::new();
    let ordered_loads = quick_sort(cmaxs.to_vec());

    for ele in 0..size_pop {
        for aka in 0..size_pop {
            if ordered_loads[ele] == cmaxs[aka]
                && ordered_population
                    .iter()
                    .filter(|&n| *n == aka)
                    .count()
                    == 0
            {
                ordered_population.push(aka);
                break;
            }
        }
    }
    //Ordenado en una característica, inicia segunda caracteristica.
    let mut equals_index = Vec::new();
    let mut equals_load = Vec::new();
    for sol in 1..ordered_population.len() {
        if population_loads[ordered_population[&sol - 1]][machines]
            == population_loads[ordered_population[sol]][machines]
        {
            equals_index.push(ordered_population[&sol - 1]);
            equals_load.push(
                population_loads[ordered_population[&sol - 1]]
                    [machines+ 1],
            );
        } else {
            if equals_index.len() != 0 {
                equals_index.push(ordered_population[sol- 1]);
                equals_load.push(
                    population_loads[ordered_population[sol- 1]]
                        [machines+ 1],
                );
                let ordered_equals_load = quick_sort(equals_load.to_vec());
                let mut final_equals_load = Vec::new();
                for ele in 0..ordered_equals_load.len(){
                    for aka in 0..ordered_equals_load.len(){
                        if ordered_equals_load[ele]
                            == population_loads[equals_index[aka]]
                                [machines+ 1]
                            && final_equals_load
                                .iter()
                                .filter(|&n| *n == equals_index[aka])
                                .count()
                                == 0
                        {
                            final_equals_load.push(equals_index[aka]);
                            break;
                        }
                    }
                }
                let mut ind = 0;
                for repe in (sol - &equals_index.len())..sol {
                    ordered_population[repe] = final_equals_load[ind];
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
    let mut sum = 0.0;
    let mut probs = Vec::new();
    for sol in population_loads {
        sum = sum
            + (population_loads[ordered_population[jobs- 1]]
                [machines]
                - sol[machines]) as f64;
    }
    for sol in &ordered_population {
        probs.push(
            (population_loads[ordered_population[jobs- 1]]
                [machines]
                - population_loads[*sol][machines]) as f64
                / sum,
        )
    }
    let mut rng = rand::thread_rng();
    let mut parents = Vec::new();
    let mut sum_probs = Vec::new();
    for ind in 0..100{
        if ind == 0 {
            sum_probs.push(probs[ind])
        } else {
            sum_probs.push(sum_probs[ind - 1] + probs[ind])
        }
    }
    for p in 0..n_parents {
        let pa = rng.gen::<f64>();
        for ind in 0..100 {
            if pa <= sum_probs[ind] {
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
        let mut less = Vec::new();
        let mut equals = Vec::new();
        let mut greater = Vec::new();
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
    parents: &mut Vec<usize>,
    machines: usize,
    population_loads: &mut Vec<Vec<i32>>,
    population_vectors: &mut Vec<Vec<Vec<usize>>>,
    ordered_population: &Vec<usize>,
    jobs: usize,
    data: &Vec<Vec<i32>>,
    cmaxs: &mut Vec<i32>,
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<i32>>, Vec<usize>) {
    let mut offspring_vectors = Vec::new();
    let mut offspring_loads = Vec::new();
    let mut rng = rand::thread_rng();
    let mut parents_new_order = Vec::new();

    let mut f = 0;
    let mut f1 = 0;
    let mut f2 = 0;

    while f < parents.len(){
        f1 = parents[f];
        f += 1;
        if parents[f] != f1 {
            f2 = parents[f];
        } else {
            let mut fx = &f + 1;
            let mut bn1 = false;
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
        let mut child1 = Vec::new();
        let mut child2 = Vec::new();
        let mut load_c1 = Vec::new();
        let mut load_c2 = Vec::new();
        let mut cont_j_c1_remove: Vec<usize> = (0..100).collect();
        let mut cont_j_c2_remove: Vec<usize> = (0..100).collect();

        let cp = rng.gen_range(1, machines - 1);
        for gene in 0..machines {
            child1.push(vec![]);
            child2.push(vec![]);
            load_c1.push(0);
            load_c2.push(0);
            if gene < cp {
                //Genes del padre 1 a hijo uno
                for j in &population_vectors[ordered_population[f1]][gene]
                {
                    if cont_j_c1_remove.iter().filter(|&n| *n == *j).count() == 1{
                        child1[gene].push(*j);
                        //remove element form vector
                        cont_j_c1_remove.retain(|&x| x != *j);
                    }
                }
                load_c1[gene] =
                    population_loads[ordered_population[f1]][gene];

                //Genes del padre 2 a hijo dos
                for j in &population_vectors[ordered_population[f2]][gene]
                {
                    if cont_j_c2_remove.iter().filter(|&n| *n == *j).count() == 1{
                        child2[gene].push(*j);
                        //remove element form vector
                        cont_j_c2_remove.retain(|&x| x != *j);
                    }
                }
                load_c2[gene] =
                    population_loads[ordered_population[f2]][gene];
            } else {
                let mut load1 = 0;
                //Genes del padre 2 a padre uno
                for j in
                    &population_vectors[ordered_population[f2]][gene]
                {
                    if cont_j_c1_remove.iter().filter(|&n| *n == *j).count() == 1{
                        child1[gene].push(*j);
                        cont_j_c1_remove.retain(|&x| x != *j);
                        load1 = load1 + data[*j][gene];
                    }
                }
                load_c1[gene] = load1;
                let mut load2 = 0;
                //Genes del padre 1 a padre dos
                for j in
                    &population_vectors[ordered_population[f1]][gene]
                {
                    if cont_j_c2_remove.iter().filter(|&n| *n == *j).count() == 1{
                        child2[gene].push(*j);
                        //remove element form vector
                        cont_j_c2_remove.retain(|&x| x != *j);
                        load2 = load2 + data[*j][gene];
                    }
                }
                load_c2[gene] = load2;
            }
        }
        //Reasignación de trabajos liberados
        cont_j_c1_remove.shuffle(&mut thread_rng());
        cont_j_c2_remove.shuffle(&mut thread_rng());
        for j in cont_j_c1_remove {
            let mut m = load_c1[0] + data[j][0];
            let mut h = 0;
            for i in 1..machines {
                if load_c1[i] + data[j][i] < m {
                    h = i;
                    m = load_c1[i] + data[j][i]
                }
            }
            child1[h].push(j);
            load_c1[h] = load_c1[h] + data[j][h];
        }

        for j in cont_j_c2_remove {
            let mut m = load_c2[0] + data[j][0];
            let mut h = 0;
            for i in 1..machines {
                if load_c2[i] + data[j][i] < m {
                    h = i;
                    m = load_c2[i] + data[j][i]
                }
            }
            child2[h].push(j);
            load_c2[h] = load_c2[h] + data[j][h];
        }

        //Evaluación para ver si la encontré cruza
        let mut cmax1 = load_c1[0];
        let mut cmax2 = load_c2[0];
        for k in 1..machines {
            if load_c1[k] > cmax1 {
                cmax1 = load_c1[k];
            }
            if load_c2[k] > cmax2 {
                cmax2 = load_c2[k];
            }
        }
        let mut cont1 = 0;
        let mut cont2 = 0;
        for o in 1..machines {
            if load_c1[o] == cmax1 {
                cont1 += 1;
            }
            if load_c2[o] == cmax2 {
                cont2 += 1;
            }
        }
        load_c1.push(cmax1);
        load_c1.push(cont1);
        load_c2.push(cmax2);
        load_c2.push(cont2);
        //Aqui termina una cruza.
        //Comienza la mutación.

        let base = (jobs / machines) / 2;
        let mut released_jobs_c1 = Vec::new();
        let mut ind = 0;
        let mut gene_load = 0;

        for gene in &mut child1 {
            if gene.len() < base{
                ind = rng.gen_range(0, gene.len());
                released_jobs_c1.push(gene[ind]);
                load_c1[gene_load] = load_c1[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);
            } else {
                ind = rng.gen_range(0, gene.len());
                released_jobs_c1.push(gene[ind]);
                load_c1[gene_load] = load_c1[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);

                ind = rng.gen_range(0, gene.len());
                released_jobs_c1.push(gene[ind]);
                load_c1[gene_load] = load_c1[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);
            }
            gene_load += 1;
        }
        for j in &released_jobs_c1 {
            if rng.gen::<f64>() <= 0.005 {
                //Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
                let mut shuffle_machines: Vec<usize> = (0..machines).collect();
                shuffle_machines.shuffle(&mut thread_rng());
                let mut bn = true;
                for i in shuffle_machines {
                    if load_c1[i] + data[*j][i]
                        < load_c1[machines]
                    {
                        child1[i].push(*j);
                        load_c1[i] = load_c1[i] + data[*j][i];
                        bn = false;
                    }
                }
                if bn == true {
                    let random_i = rng.gen_range(0, machines);
                    child1[random_i].push(*j);
                    load_c1[random_i] =
                        load_c1[random_i] + data[*j][random_i];
                }
            } else {
                let mut m = load_c1[0] + data[*j][0];
                let mut h = 0;
                for i in 1..machines {
                    if load_c1[i] + data[*j][i] < m {
                        h = i;
                        m = load_c1[i] + data[*j][i]
                    }
                }
                child1[h].push(*j);
                load_c1[h] = load_c1[h] + data[*j][h];
            }
        }
        //Secound mutation
        let mut released_jobs_c2 = Vec::new();
        gene_load = 0;
        for gene in &mut child2 {
            if gene.len() < base{
                ind = rng.gen_range(0, gene.len());
                released_jobs_c2.push(gene[ind]);
                load_c2[gene_load] = load_c2[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);
            } else {
                ind = rng.gen_range(0, gene.len());
                released_jobs_c2.push(gene[ind]);
                load_c2[gene_load] = load_c2[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);

                ind = rng.gen_range(0, gene.len());
                released_jobs_c2.push(gene[ind]);
                load_c2[gene_load] = load_c2[gene_load]
                    - data[gene[ind]][gene_load];
                gene.remove(ind);
            }
            gene_load += 1;
        }

        for j in &released_jobs_c2 {
            if rng.gen::<f64>() <= 0.005 {
                //Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
                let mut shuffle_machines: Vec<usize> = (0..machines).collect();
                shuffle_machines.shuffle(&mut thread_rng());
                let mut bn = true;
                for i in shuffle_machines {
                    if load_c2[i] + data[*j][i]
                        < load_c2[machines]
                    {
                        child2[i].push(*j);
                        load_c2[i] = load_c2[i] + data[*j][i];
                        bn = false;
                    }
                }
                if bn == true {
                    let random_i = rng.gen_range(0, machines);
                    child2[random_i].push(*j);
                    load_c2[random_i] =
                        load_c2[random_i] + data[*j][random_i];
                }
            } else {
                let mut m = load_c2[0] + data[*j][0];
                let mut h = 0;
                for i in 1..machines {
                    if load_c2[i] + data[*j][i] < m {
                        h = i;
                        m = load_c2[i] + data[*j][i]
                    }
                }
                child2[h].push(*j);
                load_c2[h] = load_c2[h] + data[*j][h];
            }
        }

        //Evaluación para ver si la encontré cruza
        cmax1 = load_c1[0];
        cmax2 = load_c2[0];
        for k in 1..machines {
            if load_c1[k] > cmax1 {
                cmax1 = load_c1[k];
            }
            if load_c2[k] > cmax2 {
                cmax2 = load_c2[k];
            }
        }
        cont1 = 0;
        cont2 = 0;
        for o in 1..machines {
            if load_c1[o] == cmax1 {
                cont1 += 1;
            }
            if load_c2[o] == cmax2 {
                cont2 += 1;
            }
        }
        load_c1[machines] = cmax1;
        load_c1[machines+ 1] = cont1;
        load_c2[machines] = cmax2;
        load_c2[machines+ 1] = cont2;
        offspring_vectors.push(child1);
        offspring_vectors.push(child2);
        offspring_loads.push(load_c1);
        offspring_loads.push(load_c2);

        parents_new_order.push(f1);
        parents_new_order.push(f2);
    }
    (offspring_vectors, offspring_loads, parents_new_order)
}

fn is_feasible(vector: Vec<Vec<i32>>, jobs: usize) -> bool {
    let mut sum = 0;

    for gene in vector {
        sum = sum + gene.len();
    }

    sum == jobs
}
