import numpy
 
'''
 Un ejemplo de algoritmo genético iterativo
-----------------------------------------
The y=target is to maximize this equation:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
------------------------------------------
 Población original: se generan 8 soluciones aleatoriamente
 Calcular la aptitud: calcular el valor de la aptitud de 8 soluciones
 Elija padre: 4 de las soluciones con mayor aptitud
 Genere descendencia: el cruce y la mutación de los padres generan 4 nuevas soluciones
 Nueva población: 4 padres + 4 hijos
 Calcule la aptitud de la nueva población y observe si hay una mejora
------------------------------------------
'''
 
# Inputs of the equation.
equation_inputs = [4,-2,3.5,5,-11,-4.7]
 
# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)  # 6
 
sol_per_pop = 8
num_parents_mating = 4
 
# Defining the population size：shape(8,6).
pop_size = (sol_per_pop,num_weights)
# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
 
# Creating the initial population.
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
 
print("initial population:")
 imprimir (nueva_población) # salida matriz de población generada aleatoriamente (8,6)
 
 
 # Calcular fitness
def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
         fitness = numpy.sum (pop * equal_inputs, axis = 1) # Cada fila de 6 bits corresponde a multiplicar y sumar para obtener la suma, se calcula un total de 8 sumas
    return fitness
 
 
a = cal_pop_fitness(equation_inputs,new_population)
print('\ninitial fitness:\n', a)
 # Genere los valores de aptitud calculados de las 8 soluciones: (8,6) * (6,1) = (8,1)
 
 
 # Seleccionar padres
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
 
         padres = numpy.empty ((num_parents, pop.shape [1])) # padres se utilizan para almacenar el padre seleccionado, forma: (4,6)
 
    for parent_num in range(num_parents):
 
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
                 # print (max_fitness_idx) guarda la matriz ([3],), por lo que debe eliminar el valor específico 3 en el siguiente paso
        max_fitness_idx = max_fitness_idx[0][0]
        # print(max_fitness_idx)
 
                 padres [parent_num,:] = pop [max_fitness_idx,:] # ponga la solución con el valor de aptitud más grande en el padre
 
                 fitness [max_fitness_idx] = -99999999999 # Asigne el valor de aptitud de la solución seleccionada para que sea muy pequeño, para no volver a ser seleccionado
    return parents
 
 
b = select_mating_pool(new_population,a,num_parents_mating)
print('\nparents:\n',b)
 
 
 # Crossover de un solo punto
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1] / 2)  # 6/2=3
    print('\ncrossover point: ',crossover_point)
 
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
 
 
offspring_size = (pop_size[0]-b.shape[0],num_weights)  # 8-4=4
c = crossover(b,offspring_size)
print('crossover offspring:\n',c)
 
 
 # Mutación aleatoria
def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
                 # print (random_value) Genera un número de mutación aleatoriamente
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover
 
 
d = mutation(c)
print('\nmutation offspring:\n',d)
 
 
 # Nueva población: 4 padres + 4 crías
new_population[0:b.shape[0], :] = b
new_population[b.shape[0]:, :] = d
 
print("\nnew population:\n",new_population)
 
 # Calcule la aptitud de la nueva población
e = cal_pop_fitness(equation_inputs,new_population)
print("\nnew population fitness:\n",e)