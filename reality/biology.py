import numpy as np
import math
import pickle
from settings import *

def gene_expression(size):
    '''
    How genes express to phenotype.
    I choose here a bimodal distribution so each gen either contribute a lot (~1) or very few (~0) to each trait.
    '''
    # sample = np.concatenate((np.random.normal(0., 0.1, int(math.ceil(size/2))),
    #                 np.random.normal(1., 0.1, int(math.floor(size/2)))))
    sample = np.random.beta(0.1, 0.1, size)
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

class Biology:

    def __init__(self, genetics = Genetics(), gestationtime=90):            
        self.gestationtime = gestationtime        
        self.genetics = genetics
        self.meiosis_variation = 0.1 # mean value of Poisson distribution for nr of genes suffering mutation

    def get_info(self):
        return f'gestation time: {self.gestationtime} \n genetics: {self.genetics.w} \n meiosis_variation: {self.meiosis_variation}'  
            
    def load(self, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)              

biology = Biology()

