import modules.mdtrain.pdef as mdtrain
import numpy as np
import pacril
import random
import time


class GASolver:

    parent_tournament_size = 4
    population_size = 100
    
    def __init__(self, trainset, influence_line, dmaxs, crossover_probability, mutation_probability):
        self.trainset = trainset
        self.influence_line = influence_line
        self.population = [trainset.get_random_chromosome() 
                           for n in range(self.population_size)]
        self.fitness_vector = [self.chromosome_damage(ch) for ch in self.population]
        self.dmaxs = dmaxs
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        
    def train_damage(self, T):
        zstatic = T.apply(self.influence_line.data)
        speed = T.locomotive.speed
        determinant_length = self.influence_line.length
        daf = pacril.find_daf_EC1(speed, determinant_length)
        zdyn = daf * zstatic
        d = mdtrain.find_damage(zdyn, k=64)
        return d
    
    def chromosome_damage(self, chromosome):
        T = self.trainset.decode(chromosome)
        return self.train_damage(T)
    
    def tournament_selection(self):
        """Return chromosome from population after a K-Way tournament selection"""
        ks = np.random.randint(0, len(self.population), self.parent_tournament_size)
        ds = [self.fitness_vector[k] for k in ks]
        n = np.argmax(ds)
        return self.population[ks[n]]
    
    def kick_worst_individuals(self, number):
        n = np.argsort(self.fitness_vector)[number:]
        self.population[:] = [self.population[j] for j in n]
        self.fitness_vector[:] = [self.fitness_vector[j] for j in n]

    def add_individuals(self):
        number = self.population_size - len(self.population)
        ps, ds = [], []
        for n in range(number):
            c1, c2 = self.tournament_selection(), self.tournament_selection()
            #copies one of the parents if neither crossover or mutation is performed
            c = random.choice([c1,c2])
            #crossover operation
            cr = [c, self.trainset.uniform_crossover_chromosome(c1,c2),
                self.trainset.bipoint_crossover_chromosome(c1,c2),
                self.trainset.onepoint_crossover_chromosome(c1,c2)]
            p1 = 1 - self.crossover_probability
            p2 = self.crossover_probability           
            num = np.random.choice(len(cr), 1, p = [p1,0,0,p2])[0]
            c = cr[num]
            #mutation operation
            mu = [c, self.trainset.flip_mutate_chromosome(c),
                self.trainset.scramble_mutate_chromosome(c),
                self.trainset.swap_mutate_chromosome(c), 
                self.trainset.inversion_mutate_chromosome(c)]
            p1 = 1 - self.mutation_probability
            p2 = self.mutation_probability
            num = np.random.choice(len(mu), 1, p = [p1,p2,0,0,0])[0]
            c = mu[num]

            d = self.chromosome_damage(c)
            ps.append(c)
            ds.append(d)
        for pi, di in zip(ps, ds):
            self.population.append(pi)
            self.fitness_vector.append(di)
                
    def terminate_search(self):
        return self.step >= 2000
    
    def get_best_chromosome(self):
        n = np.argmax(self.fitness_vector)
        return self.population[n]
    
    @property         
    def run(self):
        self.time_start = time.time()
        self.solution_history = []        
        il = self.influence_line
        ts = self.trainset
        self.dmax = self.dmaxs[il.id][il.length]
        self.step = 0
        F = max(self.fitness_vector)
        self.best_fitness = F
        self.best_chromosome = self.get_best_chromosome()
    
        while not self.terminate_search():

            r = self.best_fitness/self.dmax
            if abs(r-1) <= 0.0005 or r>1:
                break   

            kick_number = 1
            self.kick_worst_individuals(kick_number)
            self.add_individuals()
            F = max(self.fitness_vector)
            
            if F > self.best_fitness:
                self.best_fitness = F
                self.best_chromosome = self.get_best_chromosome()

            self.step += 1

        self.time = time.time() - self.time_start
        return self.best_chromosome, self.best_fitness, self.time, self.step



            
        
