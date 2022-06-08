import matplotlib.pyplot as plt
from settings import *

# Graphical tools
def plot_genes_init(group, ax):
    #x = np.arange(GEN_SIZE)
    y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in group.hvs.values()])/(2*group.nr_hvs())
    steps = ax.stairs(y, fill=True)  
    ax.set_ylim(0.,1.2)
    return steps      

def plot_genes_update(group, ax, steps):        
    y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in group.hvs.values()])/(2*group.nr_hvs())
    steps.set_data(y)

def plot_init(y, ax):
    line, = ax.plot(y)
    return line

def plot_update(y, plot):    
    plot.set_ydata(y)

def plot_steps_init(y, ax):
    return ax.stairs(y, fill=True)
    
def plot_steps_update(y, steps):
    steps.set_data(y)

# def plot_traits_init(group, ax):
#     y = group.history.memory

def plots_init(group, ax):

    y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in group.hvs.values()])/(2*group.nr_hvs())
    steps = ax[0].stairs(y, fill=True)  
    ax[0].set_ylim(0.,1.2)
    
    y_genes, lines_genes = {}, {}
    for i in range(GEN_SIZE):
        y_genes[i] = np.zeros(100)
        lines_genes[i], = ax[1].plot(y_genes[i], label='f{i}')
    ax[1].set_ylim(0., 1.2)
    
    y_traits, lines_traits = {}, {}
    for trait in TRAITS:
        y_traits[trait] = np.zeros(100)
        lines_traits[trait], = ax[2].plot(y_traits[trait], label='f{trait}')
    ax[2].set_ylim(-1., 10.)
    
    return steps, lines_genes, lines_traits

def plots_update(group, steps, lines_genes, lines_traits):

    y = sum([hv.genes.sequence[0]+hv.genes.sequence[1] for hv in group.hvs.values()])/(2*group.nr_hvs())
    plot_steps_update(y, steps)

    y_genes, y_traits = group.get_indicators()
    
    for i in range(GEN_SIZE):  
        plot_update(y_genes[i], lines_genes[i])
    
    for trait in TRAITS:         
        plot_update(y_traits[trait], lines_traits[trait])

# Plotly

# def iplot_init():  
#     fig = make_subplots(rows=2, cols=2)  
#     fig.add_bar(y=[2, 1, 3],
#             marker=dict(color="MediumPurple"),
#             name="b", row=1, col=1)    
#     return fig

# def iplot_update(fig, data):
#     with fig.batch_update():
#         fig.data[0].y = data
    
 

