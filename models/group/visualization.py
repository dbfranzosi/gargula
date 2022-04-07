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