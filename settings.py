import numpy as np

RANDOM_SEED = 123456
MAX_DAYS = 1000
MAX_HVS_IN_GROUP = 50
GEN_SIZE = 20
UNIT_ENERGY = 1./GEN_SIZE

# ages
AGE_BABY = 15
AGE_ADULT = 100
AGE_ELDER = 200

FEATURES = ['pregnant', 'age', 'energy', 'power_attack', 'resistance_attack', 'feature1']
NR_FEATURES = len(FEATURES) #visible

TRAITS = ['energy_pool','food_consumption','power_attack','resistance_attack',
            'reward_eat','reward_rest','reward_sex','reward_violence','feature1'] 

def visualize_settings():
    print(  f'=== SETTINGS === \n'
            f'RANDOM_SEED={RANDOM_SEED} \n' 
            f'MAX_DAYS={MAX_DAYS} \n'
            f'GEN_SIZE={GEN_SIZE} \n' 
            f'UNIT_ENERGY={UNIT_ENERGY}\n'
            f'===   ===')