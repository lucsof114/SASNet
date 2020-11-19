import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os
import math
import sys
import random
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import json

models = ["test3", "test4", "test5", "transformer2", "transformer4", "waveunet++1"]
model_label = ["WaveUNet", "WaveUNet-F1", "WaveUNet-F2", "WaveUNet-T1", "WaveUNet-T2", "WaveUNet-C"]
model_data = []
colors = ["#e67e22", "#e67e22", "#e67e22", "#2980b9", "#2980b9", "#2ecc71"]

for model in models:
	with open('test/{}/metrics.json'.format(model)) as json_file:
	    model_data.append(json.load(json_file))
print(model_data)

songs = [key for key in model_data[0]]
print(songs)

fig, axs = plt.subplots(len(songs), 1)

for i in range(len(songs)):
	axs[i].title.set_text(songs[i])
	xdata = []
	ydata = []
	for j in range(len(models)):
		ydata.append(model_data[j][songs[i]])
		xdata.append(model_label[j])
	axs[i].bar(xdata, ydata, color=colors)
orange_patch = mpatches.Patch(color='#e67e22', label='Original WaveUNet Architecture')
blue_patch = mpatches.Patch(color='#2980b9', label='WaveUNet Architecture with Transformer')
green_patch = mpatches.Patch(color='#2ecc71', label='WaveUNet Architecture with Classifier')
fig.legend(handles=[orange_patch, blue_patch, green_patch],loc='upper right')
fig.tight_layout()
fig.set_size_inches(15,10)
fig.savefig('performance_visualization.png')