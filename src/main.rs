extern crate rand; 
 

mod get_data;

use std::vec::Vec; 
use rand::thread_rng;
use rand::seq::SliceRandom; 
use rand::Rng;

 
//static machines: i32 = 0;
//static jobs: i32 = 0;
//static datos: Vec<Vec<i32>> = vec![];


fn main() {
	//Get Data
	let _tupla = get_data::run();  
	let _machines: i32 = _tupla.2;
    let _jobs: i32 = _tupla.1;
    let _data: Vec<Vec<i32>>=_tupla.0; 

    //Get initial population
    //Return vectores, cargas y bool
    let _population = get_initial_population(_jobs, _machines, &_data); 
    if _population.3 == false{
    	let mut _population_vectors: Vec<Vec<Vec<i32>>>= _population.0;
    	let mut _population_loads: Vec<Vec<i32>>= _population.1;
    	let mut _cmaxs: Vec<i32>= _population.2;
    	let _size_pop: i32 = 100;
    	let n_parents: i32 = 20;
    	//let mut _population_index: Vec<i32> = (0..100).collect(); 
    	let _max_generations: i32 = 2; 
    	let mut _g: i32 = 0;     	
    	//generación
    	while _g < _max_generations{
    		//Slección;
    		let _tupla2 = order_population_selection(&_population_loads, &_cmaxs, &_size_pop, &_machines, &n_parents, &_jobs); 
    		let mut _ordered_population: Vec<i32> = _tupla2.0;
    		let mut _parents: Vec<i32> = _tupla2.1; 
    		println!("G: {} Best {:?}",_g, _population_loads[_ordered_population[0] as usize]);
    		println!("G: {} Best {:?} \n",_g, _population_loads[_ordered_population[1] as usize]); 

    		//Generar Descendencia.
    		let _tuple3 = crossover_translocation(&mut _parents, &_machines,  &mut _population_loads, &mut _population_vectors, &_ordered_population, &_jobs, &_data, &mut _cmaxs);
    		let _offspring_vectors = _tuple3.0;
    		let _offspring_loads = _tuple3.1;
    		let _final_parents_order = _tuple3.2;
    		 //Remplazo
    		 let mut _p: i32 = 0; 
    		 for par in _final_parents_order{
    		 	println!("lililll, {} {} {} {}", par, _offspring_vectors.len(), _offspring_loads.len(), _cmaxs.len());
    		 	_population_vectors[par as usize] = _offspring_vectors [_p as usize].to_vec();
    		 	_cmaxs[par as usize] = _offspring_loads [_p as usize][_machines as usize];
    		 	_population_loads[par as usize] = _offspring_loads[_p as usize].to_vec();
    		 	_p += 1;

    		 } 

			//En vista del exito no obtenido, vamos atener qeu regresar una tupla con los padres ordenados
    		_g += 1;


    	}
    }else {
    	println!("****************************************** Solved");
    }
    
}

fn get_initial_population(_jobs: i32, _machines: i32, _datos: &Vec<Vec<i32>>) -> (Vec<Vec<Vec<i32>>>, Vec<Vec<i32>>, Vec<i32>, bool){	
	let mut _population_vectors: Vec<Vec<Vec<i32>>>= vec![];	
	let mut _population_loads: Vec<Vec<i32>>= vec![];	
	let mut _cmaxs: Vec<i32>= vec![];
	let mut base_vector: Vec<i32> = vec![]; 
	let mut cont: i32 = -1;
	loop{
		cont+= 1;
		base_vector.push(cont);
		if cont == (_jobs-1){
			break
		}		
	} 
	for _s in 0..100{ 
		base_vector.shuffle(&mut thread_rng()); 	
		let mut sol: Vec<Vec<i32>> = vec![];
		let mut load: Vec<i32> = vec![]; 
		//Inicialiar arreglos
		cont = 0;
		loop{
			cont+= 1;
			sol.push(vec![]);
			load.push(0);
			if cont==_machines{
				break
			}			
		}		
		//Dejamos min() aquí, para no andar pasando de un lado para otro la matriz de datos 
		let mut k = 0 as usize;
		loop{ 
			let _b: usize = base_vector[k] as usize; 
			let mut m: i32=load[0] + _datos[_b][0]; 
			let mut _h=0 as usize;
			let mut i = 0 as usize;
			loop{
				i+= 1;
				if i==_machines as usize{
					break
				}  
				if load[i] + _datos[_b][i] < m{
					_h = i;
					m = load[i] + _datos[_b][i] 
				}
			} 
			sol[_h].push(_b as i32); 
			load[_h]= load[_h] + _datos[_b][_h];  
			k+= 1;
			if k==_jobs as usize{
				break
			}	
		}  
		let mut k = 1;
		let mut cmax: i32= load[0];
		let mut cmin: i32= load[0];
		loop {
			if load[k] > cmax{
				cmax = load[k];
			}
			if load[k] < cmin{
				cmin = load[k];
			}
			k += 1;
			if k == _machines as usize{
				break
			}
		}
		let mut _o = 1;
		let mut cont=0;
		loop {
			if load[_o] == cmax{
				cont += 1;
			}
			_o += 1;
			if _o == _machines as usize{
				break
			}
		} 
		//Generamos nuestra lista a ordenar
		//Agregamos a loads max y número de repetidas con max, y agregamos min
		_cmaxs.push(cmax); 
		load.push(cmax);
		load.push(cont);
		load.push(cmin);
 		_population_vectors.push(sol);
		_population_loads.push(load);
	}
    //El booleano se utiliza para determinar si se encontró o no el óptimo, pero eso se dejará para el final. Leer el csv.
	(_population_vectors, _population_loads, _cmaxs, false) } 
fn order_population_selection(_population_loads: &Vec<Vec<i32>>, _cmaxs: &Vec<i32>, _size_pop: &i32, _machines: &i32, _n_parents: &i32, _jobs: &i32) -> (Vec<i32>, Vec<i32>){ 
	let mut _population_index: Vec<i32> = (0..100).collect(); 
	let mut _ordered_population: Vec<i32> = Vec::new();
	let _ordered_loads: Vec<i32> = quick_sort(_cmaxs.to_vec());

	for ele in 0..*_size_pop as usize{
		for aka in 0..*_size_pop as usize{
			if _ordered_loads[ele] == _cmaxs[aka] && _ordered_population.iter().filter(|&n| *n == aka as i32).count() == 0 {
				_ordered_population.push(aka as i32);
				break
			}
		}		
	}
	//Ordenado en una característica, inicia segunda caracteristica.
	let mut equals_index: Vec<i32> = Vec::new();
	let mut equals_load: Vec<i32> = Vec::new();
	for sol in 1.._ordered_population.len(){
		if  _population_loads[*&_ordered_population[&sol-1]  as usize][*_machines as usize] ==  _population_loads[*&_ordered_population[sol as usize] as usize][*_machines  as usize]{
			equals_index.push(_ordered_population[&sol-1]);
			equals_load.push(_population_loads[*&_ordered_population[&sol-1] as usize][*_machines as usize + 1]);
		}
		else{ 
			if equals_index.len() != 0{
				equals_index.push(_ordered_population[sol as usize-1]);
				equals_load.push(_population_loads[_ordered_population[sol as usize-1] as usize][*_machines as usize+1]); 
				let _ordered_equals_load: Vec<i32> = quick_sort(equals_load.to_vec()); 
				let mut final_equals_load: Vec<i32> = Vec::new();
				for ele in 0.._ordered_equals_load.len() as usize{
					for aka in 0.._ordered_equals_load.len() as usize{
						if _ordered_equals_load[ele] == _population_loads[*&equals_index[aka as usize] as usize][*_machines as usize+1] && final_equals_load.iter().filter(|&n| *n == *&equals_index[aka as usize]).count() == 0 {
							final_equals_load.push(*&equals_index[aka as usize]);
							break
						}
					}		
				} 
				let mut ind = 0; 
				for repe in (sol - &equals_index.len())..sol{ 
					_ordered_population[repe as usize] = final_equals_load[ind as usize]; 
					ind +=1 ;
				}  
				equals_index=vec![];
				equals_load=vec![]; 
			}
			else{
				equals_index=vec![];
				equals_load=vec![];
			}
		}
	}  
	//_ordered_population
	//Inicia selección
	let mut sum: f64 = 0.0;
	let mut probs: Vec<f64> = Vec::new();
	for sol in _population_loads{
		sum=sum+(_population_loads[_ordered_population[*_jobs as usize -1] as usize][*_machines as usize] - sol[*_machines as usize] ) as f64;
	}
	for sol in &_ordered_population{
		probs.push(( _population_loads[_ordered_population[*_jobs as usize -1] as usize][*_machines as usize] - _population_loads[*sol as usize][*_machines as usize]) as f64 / *&sum)  
	}
	let mut rng = rand::thread_rng();
	let mut parents: Vec<i32> = Vec::new(); 
	let mut sum_probs: Vec<f64> = Vec::new();
	for ind in 0..100 as usize{
		if ind ==0{
			sum_probs.push(probs[ind])
		}
		else{
			sum_probs.push(sum_probs[ind-1 as usize]+probs[ind as usize])
		}  
	} 
	for _p in 0..*_n_parents{
		let pa: f64 = rng.gen::<f64>();
		for ind in 0..100{ 
			if pa <= sum_probs[ind as usize]{ 
				parents.push(_ordered_population[ind]);
				break
			}
		}
	} 
	parents.shuffle(&mut thread_rng()); 
	(_ordered_population, parents)}
fn quick_sort(mut _vector: Vec<i32>) -> Vec<i32>{  
	if _vector.len() > 1{ 
		let pivote = _vector[0];
		let mut less: Vec<i32> = Vec::new();
		let mut equals: Vec<i32> = Vec::new();
		let mut greater: Vec<i32> = Vec::new();
		for value in _vector {
			if value > pivote{
				greater.push(value)
			}
			if value < pivote{
				less.push(value)
			}
			if value == pivote{
				equals.push(value)
			}
		} 	
 		let mut final_vector = Vec::new();
 		final_vector.extend( quick_sort(less));
 		final_vector.extend(equals);
 		final_vector.extend( quick_sort(greater));
 		return final_vector;
 	}else{ 
 		return _vector;
 	}}

fn crossover_translocation(_parents: &mut Vec<i32>, _machines:&i32,  _population_loads: &mut Vec<Vec<i32>>, _population_vectors: &mut Vec<Vec<Vec<i32>>>, _ordered_population:&Vec<i32>, _jobs:&i32, _data: &Vec<Vec<i32>>, _cmaxs: &mut Vec<i32>) -> (Vec<Vec<Vec<i32>>>, Vec<Vec<i32>>, Vec<i32>){
 	let mut _offspring_vectors: Vec<Vec<Vec<i32>>> = Vec::new();
 	let mut _offspring_loads: Vec<Vec<i32>> = Vec::new();
 	let mut rng = rand::thread_rng(); 
 	let mut _parents_new_order: Vec<i32> = Vec::new();


 	let mut f: usize = 0;
    let mut f1 = 0;
    let mut f2 = 0;
    while f < _parents.len() as usize {
    	f1 = _parents[f];
    	f += 1;
    	if _parents[f] != f1{
    		f2 = _parents[f];
    	}
    	else {
    		let mut fx: usize = &f+1;
    		let mut bn1: bool = false;
    		loop{
    			if _parents[fx] != f1{
    				let aux = _parents[f];
    		  		_parents[f] = _parents[fx];
    		  		_parents[fx] = aux;
    		  		f2 = _parents[f];
    		  		bn1 = true;
    		  		break
    		  	}
    		  	fx +=1;
    		}
    		if bn1 == false{
    			f2 = rng.gen_range(0, _jobs);
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

		let cp = rng.gen_range(1, _machines-1);
		for gene in 0..*_machines{
			child1.push(vec![]);
			child2.push(vec![]);
			load_c1.push(0);
			load_c2.push(0);
			if gene < cp{
				//Genes del padre 1 a hijo uno
				for j in &_population_vectors[_ordered_population[f1 as usize]as usize][gene as usize] {
					if cont_j_c1_remove.iter().filter(|&n| *n == **&j).count() == 1 as usize{ 
						child1[gene as usize].push(*j); 
						//remove element form vector
						cont_j_c1_remove.retain(|&x| x != *j); 
					}  
				} 
				load_c1[gene as usize]=_population_loads[_ordered_population[f1 as usize]as usize][gene as usize];			

				//Genes del padre 2 a hijo dos
				for j in &_population_vectors[_ordered_population[f2 as usize]as usize][gene as usize] {
					if cont_j_c2_remove.iter().filter(|&n| *n == **&j).count() == 1 as usize{ 
						child2[gene as usize].push(*j); 
						//remove element form vector
						cont_j_c2_remove.retain(|&x| x != *j); 
					}  
				} 
				load_c2[gene as usize]=_population_loads[_ordered_population[f2 as usize]as usize][gene as usize]; 

			}else {
				let mut _load1:i32 = 0;
				//Genes del padre 2 a padre uno
				for j in &_population_vectors[_ordered_population[f2 as usize]as usize][gene as usize] {
					if cont_j_c1_remove.iter().filter(|&n| *n == **&j).count() == 1 as usize{ 
						child1[gene as usize].push(*j);  
						cont_j_c1_remove.retain(|&x| x != *j); 
						_load1 = _load1+_data[*j as usize][gene as usize];
					}  
				}
				load_c1[gene as usize]=_load1;
				let mut _load2:i32 = 0;
				//Genes del padre 1 a padre dos
				for j in &_population_vectors[_ordered_population[f1 as usize]as usize][gene as usize] {
					if cont_j_c2_remove.iter().filter(|&n| *n == **&j).count() == 1 as usize{ 
						child2[gene as usize].push(*j); 
						//remove element form vector
						cont_j_c2_remove.retain(|&x| x != *j); 
						_load2 = _load2+_data[*j as usize][gene as usize];
					}  
				}  
				load_c2[gene as usize]=_load2;							
			}				
		}
		//Reasignación de trabajos liberados
		cont_j_c1_remove.shuffle(&mut thread_rng());
		cont_j_c2_remove.shuffle(&mut thread_rng());
		for j in cont_j_c1_remove{
			let mut m: i32=load_c1[0] + _data[j as usize][0]; 
			let mut _h=0 as usize; 
			for i in 1..*_machines{  
				if load_c1[i as usize] + _data[j as usize][i as usize] < m{
					_h = i as usize;
					m = load_c1[i as usize] + _data[j as usize][i as usize] 
				}
			} 
			child1[_h as usize].push(j as i32); 
			load_c1[_h as usize]= load_c1[_h as usize] + _data[j as usize][_h as usize];   		
		}

		for j in cont_j_c2_remove{
			let mut m: i32=load_c2[0] + _data[j as usize][0]; 
			let mut _h=0 as usize; 
			for i in 1..*_machines{  
				if load_c2[i as usize] + _data[j as usize][i as usize] < m{
					_h = i as usize;
					m = load_c2[i as usize] + _data[j as usize][i as usize] 
				}
			} 
			child2[_h as usize].push(j as i32); 
			load_c2[_h as usize]= load_c2[_h as usize] + _data[j as usize][_h as usize];  		
		} 

		//Evaluación para ver si la encontré cruza 
		let mut cmax1: i32= load_c1[0];
		let mut cmax2: i32= load_c2[0];
		for k in 1..*_machines{
			if load_c1[k as usize] > cmax1{
				cmax1 = load_c1[k as usize];
			}
			if load_c2[k as usize] > cmax2{
				cmax2 = load_c2[k as usize];
			} 
		} 
		let mut cont1=0;
		let mut cont2=0;
		for _o in 1..*_machines {
			if load_c1[_o as usize] == cmax1{
				cont1 += 1;
			}
			if load_c2[_o as usize] == cmax2{
				cont2 += 1;
			}
		} 
		load_c1.push(cmax1);
		load_c1.push(cont1);
		load_c2.push(cmax2);
		load_c2.push(cont2); 
		//Aqui termina una cruza.
		//Comienza la mutación.

		let base: i32 = (_jobs/_machines)/2;
		let mut released_jobs_c1: Vec<i32> = Vec::new(); 
		let mut ind: i32 = 0; 
		let mut gene_load: i32 =0; 

		for gene in &mut child1{
			if gene.len() < base as usize{ 
				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c1.push(gene[ind as usize]); 
				load_c1[gene_load as usize] = load_c1[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);
			}
			else {
				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c1.push(gene[ind as usize]);
				load_c1[gene_load as usize] = load_c1[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);

				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c1.push(gene[ind as usize]);
				load_c1[gene_load as usize] = load_c1[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);
				
			}
			gene_load += 1;
		}
		for j in &released_jobs_c1{
			if rng.gen::<f64>() <= 0.005{
				//Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
				let mut _shuffle_machines: Vec<i32> = (0..*_machines).collect(); 
				_shuffle_machines.shuffle(&mut thread_rng());
				let mut bn: bool = true;
				for i in _shuffle_machines{
					if load_c1[i as usize] + _data[*j as usize][i as usize] < load_c1[*_machines as usize]{
						child1[i as usize].push(*j as i32);
						load_c1[i as usize]= load_c1[i as usize] + _data[*j as usize][i as usize]; 
						bn = false;
					} 
				}
				if bn == true{
					let random_i = rng.gen_range(0, _machines); 
					child1[random_i as usize].push(*j as i32);
					load_c1[random_i as usize]= load_c1[random_i as usize] + _data[*j as usize][random_i as usize]; 
				}

			}else { 
				let mut m: i32=load_c1[0] + _data[*j as usize][0]; 
				let mut _h=0 as usize; 
				for i in 1..*_machines{  
					if load_c1[i as usize] + _data[*j as usize][i as usize] < m{
						_h = i as usize;
						m = load_c1[i as usize] + _data[*j as usize][i as usize] 
					}
				}
				child1[_h as usize].push(*j as i32);
				load_c1[_h as usize]= load_c1[_h as usize] + _data[*j as usize][_h as usize]; 
			}  		
		} 
		//Secound mutation
		let mut released_jobs_c2: Vec<i32> = Vec::new();  
		gene_load = 0; 
		for gene in &mut child2{
			if gene.len() < base as usize{ 
				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c2.push(gene[ind as usize]); 
				load_c2[gene_load as usize] = load_c2[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);
			}
			else {
				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c2.push(gene[ind as usize]);
				load_c2[gene_load as usize] = load_c2[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);

				ind = rng.gen_range(0, gene.len() as i32);
				released_jobs_c2.push(gene[ind as usize]);
				load_c2[gene_load as usize] = load_c2[gene_load as usize] -_data[gene[ind as usize] as usize][gene_load as usize];
				gene.remove(ind as usize);				
			}
			gene_load += 1;
		}

		for j in &released_jobs_c2{
			if rng.gen::<f64>() <= 0.005{
				//Crear arreglo con las máquinas aleatoreas, recorrerlas y asignarlo el trabajo a la primera maquina que genere un Si menor que Cmax
				let mut _shuffle_machines: Vec<i32> = (0..*_machines).collect(); 
				_shuffle_machines.shuffle(&mut thread_rng());
				let mut bn: bool = true;
				for i in _shuffle_machines{
					if load_c2[i as usize] + _data[*j as usize][i as usize] < load_c2[*_machines as usize]{
						child2[i as usize].push(*j as i32);
						load_c2[i as usize]= load_c2[i as usize] + _data[*j as usize][i as usize]; 
						bn = false;
					} 
				}
				if bn == true{
					let random_i = rng.gen_range(0, _machines); 
					child2[random_i as usize].push(*j as i32);
					load_c2[random_i as usize]= load_c2[random_i as usize] + _data[*j as usize][random_i as usize]; 
				}
			}else { 
				let mut m: i32=load_c2[0] + _data[*j as usize][0]; 
				let mut _h=0 as usize; 
				for i in 1..*_machines{  
					if load_c2[i as usize] + _data[*j as usize][i as usize] < m{
						_h = i as usize;
						m = load_c2[i as usize] + _data[*j as usize][i as usize] 
					}
				}
				child2[_h as usize].push(*j as i32);
				load_c2[_h as usize]= load_c2[_h as usize] + _data[*j as usize][_h as usize]; 
			}  		
		} 

		//Evaluación para ver si la encontré cruza 
		cmax1= load_c1[0];
		cmax2= load_c2[0];
		for k in 1..*_machines{
			if load_c1[k as usize] > cmax1{
				cmax1 = load_c1[k as usize];
			}
			if load_c2[k as usize] > cmax2{
				cmax2 = load_c2[k as usize];
			} 
		} 
		cont1=0;
		cont2=0;
		for _o in 1..*_machines {
			if load_c1[_o as usize] == cmax1{
				cont1 += 1;
			}
			if load_c2[_o as usize] == cmax2{
				cont2 += 1;
			}
		} 
		load_c1[*_machines as usize]=cmax1;
		load_c1[*_machines as usize+1]=cont1; 
		load_c2[*_machines as usize]=cmax2;
		load_c2[*_machines as usize + 1]=cont2; 
		_offspring_vectors.push(child1);
		_offspring_vectors.push(child2);
		_offspring_loads.push(load_c1);
		_offspring_loads.push(load_c2);


		_parents_new_order.push(f1);
		_parents_new_order.push(f2);

		
		


	}
	(_offspring_vectors, _offspring_loads, _parents_new_order)

	 
}

fn is_feasible(vector: Vec<Vec<i32>>, _jobs: i32) -> bool{
	let mut sum: i32= 0;
	for gene in vector{
		sum= sum + gene.len() as i32;
	}
	if sum ==_jobs{
		return true;
	}else { 
		return false;
	}
}

