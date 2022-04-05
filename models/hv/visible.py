class Visible:
    def __init__(self, owner):        
        self.features = [owner.pregnant, owner.age, owner.energy, owner.genes.phenotype.power_attack, 
            owner.genes.phenotype.resistance_attack, owner.genes.phenotype.feature1]

    def update(self, owner):        
        self.features = [owner.pregnant, owner.age, owner.energy, owner.genes.phenotype.power_attack, 
            owner.genes.phenotype.resistance_attack, owner.genes.phenotype.feature1]

        