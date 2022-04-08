from collections import namedtuple, deque

Indicators = namedtuple('Indicators',
                        ('genes', 'traits'))

class GroupHistory:

    def __init__(self, capacity, group):
        self.memory = deque([],maxlen=capacity)
        self.group = group

    def push(self, *args):
        """Save a transition"""        
        self.memory.append(Indicators(*args))

    def __len__(self):
        return len(self.memory)
    