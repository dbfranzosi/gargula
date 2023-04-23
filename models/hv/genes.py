from settings import *
import numpy as np
#from reality.biology import biology
from scipy.stats import poisson

#genetics = biology.genetics

class Genes:
    '''
    Class for dealing with genes.
    '''    

    def __init__(self, biology, haploid_father, haploid_mother):
        self.biology = biology
        self.sequence = [haploid_father, haploid_mother]        
        self.phenotype = Phenotype(biology.genetics, self.sequence)
        
    def meiosis(self):
        variation = poisson.rvs(self.biology.meiosis_variation, size=1)[0]        
        a = np.random.randint(0,2,size=GEN_SIZE)
        haploid = np.array([self.sequence[a[i]][i] for i in range(len(a))] )
        if variation > 0:
            variations = np.random.randint(0, GEN_SIZE, size=variation)
            for var in variations:
                haploid[var] = int(not (haploid[var] and 1)) # 1 -> 0, 0 -> 1
        return haploid


def fertilization(haploid_father, haploid_mother):
        return Genes(haploid_father, haploid_mother)
    
class Phenotype:
    '''
    Class of phenotype, characteristics determined by the gen. 
    In principle it could be simply a dictionary, 
    but let's keep it as a class in case we want to make it more general.
    '''
    def __init__(self, genetics, sequence):
        
        # energetics/metabolism 
        self.expression = [a*b for a,b in zip(sequence[0],sequence[1])]
        self.traits = {}
        for trait in genetics.w.keys():
            self.traits[trait] = np.dot(genetics.w[trait], self.expression)
    
def haploid2genes(haploid_father, haploid_mother):
    return [haploid_father, haploid_mother]