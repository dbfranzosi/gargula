class Visible:
    def __init__(self, owner):
        traits = owner.genes.phenotype.traits
        self.features = [owner.pregnant, owner.age, owner.energy, traits['power_attack'], 
            traits['resistance_attack'], traits['feature1']]

    def update(self, owner):   
        traits = owner.genes.phenotype.traits     
        self.features = [owner.pregnant, owner.age, owner.energy, traits['power_attack'], 
            traits['resistance_attack'], traits['feature1']]

# Add visibles of area as weel, e.g. food.