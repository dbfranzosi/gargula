import numpy as np
import copy
from .genes import Genes
from .mind.mind import *
from .visible import Visible
from .actions import Rest
from settings import *
from .history import HvHistory
import time
import pandas as pd
import os
import python_avatars as pa

class Hv:
        
    def __init__(self, group, haploid_father=None, haploid_mother=None, energy = 0.0, generation=1, name=''):        

        self.group = group
        self.id = group.id_last
        self.group.id_last += 1
        group.hvs[self.id] = self          
        self.genes = Genes(group.biology, haploid_father, haploid_mother)         
        self.pregnant=0 
        self.age=0    
        self.generation = generation   
        self.birth = self.group.home.clock          
        if energy==0.0:
            self.energy = self.genes.phenotype.traits['energy_pool']
        else:
            self.energy = energy
        self.name = name
        self.parents = None
        self.visible = Visible(self)                
        self.action = Rest(self) 

        self.mind = GraphDQLMind(self, memory_capacity=1000)
        self.history = HvHistory(self, capacity=3000)

        self.avatar = self.get_avatar_rep()

    def act(self):
        self.action = self.mind.decide_action()        
        self.action.effects()        
        self.mind.update_memory()  
        self.mind.optimize_mind()           

    def aging(self):
        self.age += 1
        self.visible.update(self)
        if self.age > AGE_BABY:
            pass
            #self.mind = MemoryGraphMind(self, 1000)   

        if self.pregnant >= AGE_BABY:
            self.pregnant = 0
        if self.pregnant>0:
            self.pregnant += 1            
    
    def death(self):
        if (self.energy<0):
            # save importat info of history if it has a relevant history
            if self.age > 200:
                self.history.save()
                self.save_df()            
            del self.group.hvs[self.id]                                          
            # delete image
            # print(f"removing avatar hv{self.id}" )
            # os.remove(f"assets/avatars/hv{self.id}.svg")

    def make_baby(self, mother):
        haploid_father = self.genes.meiosis()
        haploid_mother = mother.genes.meiosis()
        #generation = max(self.generation, mother.generation) + 1
        generation = max(self.generation, mother.generation) + 1 
        baby_name = f'{mother.name.split()[0]} {generation}'
        baby = Hv(group=self.group, name=baby_name, haploid_father=haploid_father, 
                haploid_mother=haploid_mother, energy=UNIT_ENERGY, generation=generation) 
        #baby.energy = min(50*UNIT_ENERGY, baby.genes.phenotype.traits['energy_pool'])
        baby.energy = baby.genes.phenotype.traits['energy_pool']/5
        baby.parents = [self.id, mother.id]
        # baby.mind = MemoryGraphMind(baby, 1000)
        mother.pregnant += 1

        # Primitive version of teaching (copy father's mind)
        baby.mind = copy.copy(self.mind)
        baby.mind.owner = baby

        return baby

    def get_info(self, show_genes=False, show_action=False, show_visible=False):
        str = f'{self.name}({self.id}): age={self.age}, energy={self.energy}, pregnant={self.pregnant}. '
        if show_action:
            str += f'{self.action.description}'
        if show_genes:
            str += f'\n{self.genes.sequence[0]}\n'
            str += f'{self.genes.sequence[1]}'
        if show_visible:
            str += f'\n{self.visible.features}\n'    
        return str
        
    def get_genes(self):
        return self.genes.sequence[0]+self.genes.sequence[1]

    def to_df(self):        
        info = (self.genes.phenotype.traits).copy()
        info['birth'] = self.birth
        info['age'] = self.age         
        df = pd.DataFrame(data=info, index=[self.id])        
        
        return df

    def save_df(self, header=False): 
        df = self.to_df()

        # with open(f'./data/history/{self.owner.group.name}_hvs.csv', 'a') as f:
        with open(f'./data/history/tmp_info_hvs.csv', 'a') as f:
            df.to_csv(f, mode='a', header=header)

    def get_avatar_rep(self):

        # profile -> ['energy', 'energy_pool', 'power_attack', 'resistance_attack', 'feature1']
        profile = self.visible.features
        expression = self.genes.phenotype.expression
        
        avatar = pa.Avatar(
            style=pa.AvatarStyle.CIRCLE,            
            background_color=pa.BackgroundColor.WHITE,
            top=pa.HairType.SHORT_FLAT,
            eyebrows=pa.EyebrowType.DEFAULT_NATURAL,
            eyes=pa.EyeType.DEFAULT,
            nose=pa.NoseType.DEFAULT,
            mouth=pa.MouthType.DEFAULT,
            facial_hair=pa.FacialHairType.NONE,
            # You can use hex colors on any color attribute...            
            # skin_color="#{0:X}0000".format(int(profile[4]*20)),
            skin_color=pa.SkinColor.LIGHT,
            # Or you can use the colors provided by the library
            hair_color=pa.HairColor.BLACK,
            #hair_color="#{0:X}0000".format(int(profile[2]*20)),
            accessory=pa.AccessoryType.NONE,
            clothing=pa.ClothingType.HOODIE,
            clothing_color=pa.ClothingColor.HEATHER
        )

        # Change if gen is recessive
        if (expression[0]==1):
            avatar.top = pa.HairType.SHORT_WAVED 
        if (expression[1]==1):
            avatar.skin_color = pa.SkinColor.BROWN
        if (expression[2]==1):
            avatar.hair_color = pa.HairColor.RED
        if (expression[3]==1):
            avatar.nose = pa.NoseType.WIDE
        if (expression[4]==1):
            avatar.eyebrows = pa.EyebrowType.ANGRY
        if (expression[5]==1):
            avatar.eyes = pa.EyeType.SURPRISED
        if (expression[6]==1):
            avatar.mouth = pa.MouthType.GRIMACE        

        # # Save to a file        
        # avatar_file = f"assets/avatars/hv{self.id}.svg"
        # if os.path.exists(avatar_file):
        #     os.remove(avatar_file)        
        # avatar.render(avatar_file)

        return avatar    
