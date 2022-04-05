import numpy as np

RANDOM_SEED = 123456
MAX_DAYS = 100
MAX_HVS_IN_GROUP = 50
GEN_SIZE = 20
UNIT_ENERGY = 1./GEN_SIZE

# ages
AGE_BABY = 15
AGE_ADULT = 100
AGE_ELDER = 200

NR_FEATURES = 6 # len(features)

def visualize_settings():
    print(  f'=== SETTINGS === \n'
            f'RANDOM_SEED={RANDOM_SEED} \n' 
            f'MAX_DAYS={MAX_DAYS} \n'
            f'GEN_SIZE={GEN_SIZE} \n' 
            f'UNIT_ENERGY={UNIT_ENERGY}\n'
            f'===   ===')