from reality.biology import biology
from reality.geography import eden
from models.group.group import gargalo
from settings import *
from models.group.visualization import *
import matplotlib.pyplot as plt

visualize_settings()
eden.visualize()
gargalo.visualize()

# Plots
plt.ion()
fig = plt.figure()
ax = {}
ax[0] = plt.subplot(2,2,1)
ax[1] = plt.subplot(2,2,2)
ax[2] = plt.subplot(2,2,3)
ax[3] = plt.subplot(2,2,4)

steps, lines_genes, lines_traits = plots_init(gargalo, ax)

while eden.clock < MAX_DAYS:
    # Pass day is in area so when areas interact they must be set in the same day.
    eden.visualize()
    eden.pass_day()    
    gargalo.visualize()
    
    #plots update
    plots_update(gargalo, steps, lines_genes, lines_traits)
    fig.canvas.draw()
    fig.canvas.flush_events()

    if eden.nr_groups() <= 0:
        print('All groups have been extinct.')
        break
else:
    print('Reached maximum of days, ', MAX_DAYS)
    
    
    