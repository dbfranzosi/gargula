from settings import *
import numpy as np
from reality.biology import biology
from scipy.stats import poisson

genetics = biology.genetics

class Genes:
    '''
    Class for dealing with genes.
    '''    

    def __init__(self, haploid_father, haploid_mother):
        self.sequence = [haploid_father, haploid_mother]        
        self.phenotype = Phenotype(self.sequence)
        
    def meiosis(self):
        variation = poisson.rvs(biology.meiosis_variation, size=1)[0]        
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
    def __init__(self, sequence):
        
        # energetics/metabolism 
        expression = [a*b for a,b in zip(sequence[0],sequence[1])]
        self.traits = {}
        for trait in genetics.w.keys():
            self.traits[trait] = np.dot(genetics.w[trait], expression)

        # self.energy_pool = np.dot(genetics.w['energy_pool'], expression) #total energy capacity        
        # self.food_consumption = np.dot(genetics.w['food_consumption'], expression) #total energy capacity        
        # #self.energy_egg = np.dot(genetics.w['energy_egg'], expression) #total energy capacity        
        
        # # fight and resistance power
        # self.power_attack = np.dot(genetics.w['power_attack'], expression) 
        # self.resistance_attack = np.dot(genetics.w['resistance_attack'], expression) 

        # # mind reward      
        # self.reward_eat = np.dot(genetics.w['reward_eat'], expression) 
        # self.reward_rest = np.dot(genetics.w['reward_rest'], expression) 
        # self.reward_sex = np.dot(genetics.w['reward_sex'], expression)
        # self.reward_violence = np.dot(genetics.w['reward_violence'], expression)

        # # appearence
        # self.feature1 = np.dot(genetics.w['feature1'], expression) #e.g. hair color
    
def haploid2genes(haploid_father, haploid_mother):
    return [haploid_father, haploid_mother]