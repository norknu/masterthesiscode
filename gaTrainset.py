import modules.mdtrain.pdef as mdtrain
import pickle
import numpy as np
import pacril

class GATrainSet(mdtrain.TrainSet):
    def __init__(self, trainset_number):
        super(GATrainSet, self).__init__(trainset_number)
        for n, L in enumerate(self.locomotives):
            L.allele = n
        for n, W in enumerate(self.wagons):
            W.allele = n

    def get_random_chromosome(self):
        L = [np.random.randint(0, len(self.locomotives))]
        Ws = np.zeros(self.Nmax, dtype=np.int) - 1
        while not (Ws[Ws>=0].size > self.Nmin):
            Ws = np.random.randint(-1, len(self.wagons), size=self.Nmax)
        return L + Ws.tolist()
    
    def decode(self, chromosome):
        L = self.locomotives[chromosome[0]]
        W = [self.wagons[n] for n in chromosome[1:] if n >= 0]
        T = pacril.Train(L, W)
        T.chromosome = chromosome
        return T
    
    def encode(self, train):
        return train.chromosome
    
    def get_train(self):
        ch = self.get_chromosome()
        return self.decode(ch)
    
    def flip_mutate_chromosome(self, chromosome):
        i = np.random.randint(self.Nmax+1, dtype=np.int)
        if i == 0:
            j = np.random.randint(0, len(self.locomotives))
        else:
            Nwag = len([n for n in chromosome[1:] if n >= 0])
            if Nwag > self.Nmin:
                jmin = -1
            else:
                jmin = 0
            j = np.random.randint(jmin, len(self.wagons))
        ch = chromosome[:]
        ch[i] = j
        return ch
    
    def scramble_mutate_chromosome(self, chromosome):
        while True:
            ch = chromosome[:]
            n = np.random.randint(0, len(ch), size=2)
            n1, n2 = min(n), max(n)
            if n1 == n2:
                pass
            else:
                if n1 == 0:
                    ch[0] = np.random.randint(0, len(self.locomotives))
                    n1 +=1
                if n2 - n1 > 0:
                    ci = ch[n1:n2+1]
                    np.random.shuffle(ci)
                    ch[n1:n2+1] = ci
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break
        return ch
    
    def swap_mutate_chromosome(self, chromosome):
        while True:
            ch = chromosome[:]
            n = np.random.randint(0, len(ch), size=2)
            n1, n2 = n.min(), n.max()
            if n1 == n2:
                pass
            else:
                if n1 == 0 or n2 == 0:
                    ch[0] = np.random.randint(0, len(self.locomotives))
                else:
                    c1 = ch[n1]
                    ch[n1] = ch[n2]
                    ch[n2] = c1
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break
        return ch
        
    def inversion_mutate_chromosome(self, chromosome):
        while True:
            ch = chromosome[:]
            n = np.random.randint(0, len(ch))
            if n == 0:
                ch[0] = np.random.randint(0, len(self.locomotives))
                pass
            else:
                ch[:] = ch[:n+1] + list(reversed(ch[n+1:]))
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break
        return ch
    
    def uniform_crossover_chromosome(self, chromosome1, chromosome2):
        while True:
            ch = []
            for allele1, allele2 in zip(chromosome1, chromosome2):
                if np.random.rand() < .5:
                    ch.append(allele1)
                else:
                    ch.append(allele2)
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break
        return ch
    
    def bipoint_crossover_chromosome(self, chromosome1, chromosome2):
        while True:
            n = np.random.randint(0, len(chromosome1), size=2)
            n1, n2 = min(n), max(n)
            ch = chromosome1[:n1] + chromosome2[n1:n2] + chromosome1[n2:]
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break
        return ch
    
    def onepoint_crossover_chromosome(self, chromosome1, chromosome2):
        while True:
            n = np.random.randint(0, len(chromosome1)+1)
            ch = chromosome1[:n] + chromosome2[n:]
            if len([j for j in ch if j >= 0]) >= self.Nmin + 1:
                break

        return ch 
    