import numpy as np
import math
from settings import *

def gene_expression(size):
    '''
    How genes express to phenotype.
    I choose here a bimodal distribution so each gen either contribute a lot (~1) or very few (~0) to each trait.
    '''
    sample = np.concatenate((np.random.normal(0., 0.1, int(math.ceil(size/2))),
                    np.random.normal(1., 0.1, int(math.floor(size/2)))))
    np.random.shuffle(sample)
    return sample

class Genetics:
    '''
    Class of genetics. Define the relation, once and for all, between genes and phenotype
    '''

    def __init__(self):
        
        self.w = {}
        for trait in TRAITS:
            self.w[trait] = gene_expression(GEN_SIZE)            

        # # metabolism                
        # self.w['energy_pool'] = gene_expression(GEN_SIZE)        
        # self.w['food_consumption'] = gene_expression(GEN_SIZE)
        
        # # action and resistance power
        # self.w['power_attack'] = gene_expression(GEN_SIZE)        
        # self.w['resistance_attack'] = gene_expression(GEN_SIZE)        

        # # mind reward
        # self.w['reward_eat'] = gene_expression(GEN_SIZE)        
        # self.w['reward_rest'] = gene_expression(GEN_SIZE)        
        # self.w['reward_sex'] = gene_expression(GEN_SIZE)        
        # self.w['reward_violence'] = gene_expression(GEN_SIZE)

        # # appearence
        # self.w['feature1'] = gene_expression(GEN_SIZE)        

class Biology:

    def __init__(self, genetics = Genetics(), gestationtime=90):            
        self.gestationtime = gestationtime        
        self.genetics = genetics

biology = Biology()

