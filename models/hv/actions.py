from settings import *

nr_targeted_actions = 2 # sex
nr_nontarg_actions = 2 # eat, rest
nr_class_actions = nr_targeted_actions + nr_nontarg_actions
nr_actions = MAX_HVS_IN_GROUP*nr_targeted_actions + nr_nontarg_actions
lst_actions = {}

for i in range(MAX_HVS_IN_GROUP):
    lst_actions[i] = ['sex', i]
for i in range(MAX_HVS_IN_GROUP, 2*MAX_HVS_IN_GROUP):
    lst_actions[i] = ['attack', i] 

lst_actions[nr_targeted_actions*MAX_HVS_IN_GROUP + nr_nontarg_actions -2] = ['eat', -1] 
lst_actions[nr_targeted_actions*MAX_HVS_IN_GROUP + nr_nontarg_actions -1] = ['rest', -1] # rest is always the last action

#ACTIONS = ['sex', 'attack', 'eat', 'rest'] #defined in settings

# Passive actions return a boolean
lst_passive_actions = {}
nr_passive_actions = 1 # accept_sex
lst_passive_actions['accept_sex'] = 0

def get_action(code, owner): 
    group = owner.group
    hvs_keys = list(group.hvs.keys())
    if (code[1] >= len(hvs_keys)): # target doesn't exist
        return Rest(owner)
    if code[0] == 'sex':   
        itarget = hvs_keys[code[1]]     
        return TrySex(owner, group.hvs[itarget])
    if code[0] == 'attack':   
        itarget = hvs_keys[code[1]]     
        return Attack(owner, group.hvs[itarget])    
    elif code[0] == 'rest':
        # doesn't matter the target
        return Rest(owner)
    elif code[0] == 'eat':
        return Eat(owner)


class Action:
    def __init__(self, owner, conditions = True, energy_cost = UNIT_ENERGY, 
            reward = 0.0, description = 'Generic action.'):
        self.energy_cost = energy_cost        
        self.owner = owner
        self.description = description
        self.conditions = conditions
        self.reward = reward
        self.name = 'Generic action'

    def effects(self):
        self.owner.energy -= self.energy_cost                   

class TargetedAction(Action):
    def __init__(self, owner, target, conditions = True, energy_cost = UNIT_ENERGY, description = 'Targeted action.'):
        super().__init__(owner, conditions=conditions, energy_cost= energy_cost, description = description)
        self.target = target

class ResistedAction(TargetedAction):
    def __init__(self, owner, target, power = 1., resistance = 0., conditions = True, energy_cost = UNIT_ENERGY, description = 'Resisted action.'):
        super().__init__(owner, target, conditions=conditions, energy_cost= energy_cost, description = description)
        self.target = target
        self.resistance = resistance
        self.power = power

    def achieve(self):
        if (self.owner != self.target):
            hit = (self.power - self.resistance)/(self.power + self.resistance)
            #if hit > 2*np.random.random()-1.0:
            if hit > 0.: #deterministic for test
                return True
            else:
                return False
        else:
            return False

class Rest(Action):
    def __init__(self, owner):
        super().__init__(owner)
        self.reward = self.owner.genes.phenotype.traits['reward_rest']
        self.description = f'{owner.name} is relaxing.'
        self.name = 'rest'

class Eat(Action):
    def __init__(self, owner):
        super().__init__(owner)        
        self.reward = 0.0
        self.description = f'{owner.name} is eating.'
        self.name = 'eat'
    def effects(self):
        super().effects()
        amount = self.owner.genes.phenotype.traits['food_consumption']*UNIT_ENERGY
        amount = min(amount, self.owner.group.home.food) # can't take more than available
        amount = min(amount, self.owner.genes.phenotype.traits['energy_pool'] - self.owner.energy) # can't take more than max

        self.owner.energy += amount
        self.owner.group.home.food -= amount
        #self.reward = self.owner.genes.phenotype.traits['reward_eat']*amount/UNIT_ENERGY
        self.reward = self.owner.genes.phenotype.traits['reward_eat']
        
class TrySex(ResistedAction):
    def __init__(self, owner, target):        
        super().__init__(owner, target)  
        self.power = owner.genes.phenotype.traits['power_attack']
        self.resistance = target.genes.phenotype.traits['resistance_attack']
        
        self.conditions = self.conditions and (owner != target)
        # mind based passive decision
        # conditions = conditions and target.mind.decide_passive("accept_sex", owner)  
        # passive resistance
        self.conditions = self.conditions and self.achieve()

        self.name = 'sex'
        self.reward = self.owner.genes.phenotype.traits['reward_sex']
                
        if owner == target:
            self.description = f'{owner.name} is touching hemself! (hem=him/her)'
            #self.reward = self.reward*0.1                 
            #self.reward = 0.0
        else:
            self.description = f'{owner.name} is naked with {target.name}!'
            self.energy_cost *= 2            

        if self.conditions:            
            self.description += f' They love each other!'            
            self.reward = self.reward*4
        else:
            # pass
            self.reward = self.reward*0.5
            
    def effects(self):
        super().effects()
        if self.conditions:
            if (self.target.pregnant == 0 and self.owner.pregnant == 0) and (self.owner.group.nr_hvs() < MAX_HVS_IN_GROUP):            
                baby = self.owner.make_baby(self.target)

class Attack(ResistedAction):
    def __init__(self, owner, target):              
        super().__init__(owner, target)        
        self.power = owner.genes.phenotype.traits['power_attack']
        self.resistance = target.genes.phenotype.traits['resistance_attack']

        self.conditions = self.achieve()

        self.reward = self.owner.genes.phenotype.traits['reward_violence']
        self.description = f'{owner.name} tried to hit {target.name}.'                
        self.name = 'attack'
        if self.conditions:
            self.reward = self.reward*4
            self.description += f' And did it!'
        else:            
            self.reward = self.reward*0.5
            #self.reward = 0.0

    def effects(self):
        super().effects()
        damage = max(1.0, self.power - self.resistance)
        if self.conditions:            
            #self.target.energy -= self.owner.genes.phenotype.traits['power_attack']*UNIT_ENERGY
            self.target.energy -= damage*UNIT_ENERGY
            self.reward *= damage

