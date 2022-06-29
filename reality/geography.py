import time
import random
from settings import *

class Geography:
    ''' Areas and their relations.'''
    def __init__(self, areas={}):        
        self.areas = areas

class Area:
    '''
    Basic class for the area, where hv and food sits.
    '''

    def __init__(self, id=0, name='', groups={}, dimensions = [50,50], food=10.0):        
        self.id = id
        self.name = name
        self.timeunit = 2
        self.clock = 0 
        self.groups = groups        
        self.dimensions = dimensions # h x w 
        self.food = food
        self.food_generation = 10*UNIT_ENERGY # enough to keep 10 hv resting 
    
    def nr_groups(self):
        return len(self.groups) 

    def get_info(self, show_food=True):
        str = f'Area {self.name} at day {self.clock}.'
        if show_food:
            str += f' Food: {self.food:.2f}.'        
        return str

    def check_extinctions(self):
        group_keys = list(self.groups.keys()) # fix to avoid change in the loop
        for igroup in group_keys:
            group = self.groups[igroup]
            group.extinction()

    def pass_day(self):        
        self.clock += 1
        # time.sleep(self.timeunit) 
        # sleep given by dash
        self.food += self.food_generation

        self.check_extinctions()

        group_keys = list(self.groups.keys()) # fix to avoid change in the loop
        random.shuffle(group_keys) # random initiative         
        for igroup in group_keys:
            group = self.groups[igroup]
            #group.visualize()
            print(time.time())
            group.check_deaths()
            print(time.time())
            group.interact()
            print(time.time())
            group.history.update()
            print(time.time())
            # Save in csv file
            if (self.clock % group.history.capacity == 0):
                if (int(self.clock / group.history.capacity) == 1):
                    group.history.save(header=True)                    
                else:
                    group.history.save()      
            print(time.time())              

        return True

eden = Area(name='Eden')

 

        
