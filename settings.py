import numpy as np

RANDOM_SEED = 123456
MAX_DAYS = 1000
MAX_HVS_IN_GROUP = 50
GEN_SIZE = 20
UNIT_ENERGY = 1./GEN_SIZE/100.

# ages
AGE_BABY = 15
AGE_ADULT = 100
AGE_ELDER = 200

FEATURES = ['pregnant', 'age', 'energy', 'energy_pool', 'power_attack', 'resistance_attack', 'feature1']
NR_FEATURES = len(FEATURES) #visible

TRAITS = ['energy_pool','food_consumption','power_attack','resistance_attack',
            'reward_eat','reward_rest','reward_sex','reward_violence','feature1'] 

ACTIONS = ['try_sex', 'attack', 'eat', 'rest']

# Training
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-5
#BATCH_SIZE = 128
#BATCH_SIZE = 20
BATCH_SIZE = 5
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

def visualize_settings():
    print(  f'=== SETTINGS === \n'
            f'RANDOM_SEED={RANDOM_SEED} \n' 
            f'MAX_DAYS={MAX_DAYS} \n'
            f'GEN_SIZE={GEN_SIZE} \n' 
            f'UNIT_ENERGY={UNIT_ENERGY}\n'
            f'===   ===')