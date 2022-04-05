from settings import *

nr_targeted_actions = 2 # try_sex
nr_nontarg_actions = 2 # eat, rest
nr_class_actions = nr_targeted_actions + nr_nontarg_actions
nr_actions = MAX_HVS_IN_GROUP*nr_targeted_actions + nr_nontarg_actions
lst_actions = {}

for i in range(MAX_HVS_IN_GROUP):
    lst_actions[i] = ['try_sex', i]
for i in range(MAX_HVS_IN_GROUP, 2*MAX_HVS_IN_GROUP):
    lst_actions[i] = ['attack', i] 

lst_actions[nr_targeted_actions*MAX_HVS_IN_GROUP + nr_nontarg_actions -2] = ['eat', -1] 
lst_actions[nr_targeted_actions*MAX_HVS_IN_GROUP + nr_nontarg_actions -1] = ['rest', -1] # rest is always the last action

lst_class_actions = ['try_sex', 'attack', 'eat', 'rest']

# Passive actions return a boolean
lst_passive_actions = {}
nr_passive_actions = 1 # accept_sex
lst_passive_actions['accept_sex'] = 0

def get_action(code, owner): 
    group = owner.group
    hvs_keys = list(group.hvs.keys())
    if (code[1] >= len(hvs_keys)): # target doesn't exist
        return Rest(owner)
    if code[0] == 'try_sex':   
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
    def __init__(self, owner, conditions = True, energy_cost_success = UNIT_ENERGY, 
                energy_cost_fail = UNIT_ENERGY, reward = 0.0, description = 'Generic action.'):
        self.energy_cost_success = energy_cost_success
        self.energy_cost_fail = energy_cost_fail
        self.owner = owner
        self.description = description
        self.conditions = conditions
        self.reward = reward

    def effects(self):
        if (self.conditions):
            self.owner.energy -= self.energy_cost_success            
        else:
            self.owner.energy -= self.energy_cost_fail

class TargetedAction(Action):
    def __init__(self, owner, target, conditions = True, energy_cost_success = UNIT_ENERGY, 
                energy_cost_fail = UNIT_ENERGY, description = 'Targeted action.'):
        super().__init__(owner, conditions=conditions, energy_cost_success= energy_cost_success, 
                energy_cost_fail=energy_cost_fail, description = description)
        self.target = target

class Rest(Action):
    def __init__(self, owner):
        super().__init__(owner)
        self.reward = self.owner.genes.phenotype.reward_rest
        self.description = f'{owner.name} is relaxing.'

class Eat(Action):
    def __init__(self, owner):
        super().__init__(owner)
        self.reward = self.owner.genes.phenotype.reward_eat
        self.description = f'{owner.name} is eating.'
    def effects(self):
        super().effects()
        amount = self.owner.genes.phenotype.food_consumption
        amount = min(amount, self.owner.group.home.food) # can't take more than available
        amount = min(amount, self.owner.genes.phenotype.energy_pool - self.owner.energy) # can't take more than max

        self.owner.energy += amount
        self.owner.group.home.food -= amount
        
class TrySex(TargetedAction):
    def __init__(self, owner, target):        

        conditions = (target.pregnant == 0 and owner.pregnant == 0)
        conditions = conditions and target.mind.decide_passive("accept_sex", owner)

        super().__init__(owner, target, conditions=conditions, energy_cost_success = 5.0*UNIT_ENERGY, 
                energy_cost_fail = 1.0*UNIT_ENERGY)  

        self.reward = self.owner.genes.phenotype.reward_sex      
                
        if owner == target:
            self.description = f'{owner.name} is touching hemself! (hem=him/her)'
        else:
            self.description = f'{owner.name} is naked with {target.name}!'
        if conditions:
            self.description += f' They love each other!'

    def effects(self):
        super().effects()
        if self.conditions:            
            baby = self.owner.make_baby(self.target)

class Attack(TargetedAction):
    def __init__(self, owner, target):        
        
        # probabilistic based on attack/resistance
        if owner != target:
            hit = (owner.genes.phenotype.power_attack - target.genes.phenotype.resistance_attack)/ \
                    (owner.genes.phenotype.power_attack + target.genes.phenotype.resistance_attack)
            if hit > 2*np.random.random()-1.0:
                conditions = True
            else:
                conditions = False
        else:
            conditions = False

        super().__init__(owner, target, conditions=conditions, energy_cost_success = 3.0*UNIT_ENERGY, 
                energy_cost_fail = 3.0*UNIT_ENERGY)        
        self.reward = self.owner.genes.phenotype.reward_violence
        self.description = f'{owner.name} tried to hit {target.name}.'                
        if conditions:
            self.description += f' And did it!'

    def effects(self):
        super().effects()
        if self.conditions:            
            self.target.energy -= self.owner.genes.phenotype.power_attack/UNIT_ENERGY





    

    
    
