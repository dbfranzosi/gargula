from reality.biology import biology
from reality.geography import eden
from models.group.group import gargalo
from settings import *
import matplotlib.pyplot as plt
from models.group.visualization import *

visualize_settings()
eden.visualize()
gargalo.visualize()

plt.ion()
fig = plt.figure()
ax = plt.subplot(2,2,1)
steps = plot_genes_init(gargalo, ax)

while eden.clock < MAX_DAYS and eden.nr_groups() > 0:
    # Pass day is in area so when areas interact they must be set in the same day.
    eden.visualize()
    eden.pass_day()    
    plot_genes_update(gargalo, ax, steps)
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    
    