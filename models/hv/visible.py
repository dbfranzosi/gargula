# defined in settings.py:
# FEATURES = ['pregnant', 'age', 'energy', 'power_attack', 'resistance_attack', 'feature1']

class Visible:
    def get_features(self, owner):
        traits = owner.genes.phenotype.traits
        # return [owner.pregnant, owner.age, owner.energy, traits['energy_pool'], traits['power_attack'], 
        #     traits['resistance_attack'], traits['feature1']]        
        return [owner.energy, traits['energy_pool'], traits['power_attack'], 
            traits['resistance_attack'], traits['feature1']]        

    def __init__(self, owner):
        traits = owner.genes.phenotype.traits
        self.features = self.get_features(owner)

    def update(self, owner):   
        traits = owner.genes.phenotype.traits     
        self.features = self.get_features(owner)


# Add visibles of area as weel, e.g. food.