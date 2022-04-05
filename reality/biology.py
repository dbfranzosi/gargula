import numpy as np
from settings import *

class Genetics:
    '''
    Class of genetics. Define the relation, once and for all, between genes and phenotype
    '''

    def __init__(self):
        
        self.w = {}
        # metabolism
        self.w['energy_pool'] = np.random.random(GEN_SIZE)
        self.w['food_consumption'] = np.random.random(GEN_SIZE)
        
        # action and resistance power
        self.w['power_attack'] = np.random.random(GEN_SIZE)        
        self.w['resistance_attack'] = np.random.random(GEN_SIZE)        

        # mind reward
        self.w['reward_eat'] = np.random.random(GEN_SIZE)        
        self.w['reward_rest'] = np.random.random(GEN_SIZE)        
        self.w['reward_sex'] = np.random.random(GEN_SIZE)        
        self.w['reward_violence'] = np.random.random(GEN_SIZE)

        # appearence
        self.w['feature1'] = np.random.random(GEN_SIZE)        

class Biology:

    def __init__(self, genetics = Genetics(), gestationtime=90):            
        self.gestationtime = gestationtime        
        self.genetics = genetics

biology = Biology()