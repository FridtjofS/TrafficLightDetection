import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV files
#file_names =['/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/train_loss_iou.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/validiouloss.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/validmap50.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/validmap5095.csv']
file_names =['/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/RUN_300/csv.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/RUN_300/csv-2.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/RUN_300/csv-3.csv', '/Users/jakobsnel/Development/MLMaster/ComputerVisionPraktikum/viz/RUN_300/csv-4.csv']
dfs = [pd.read_csv(file) for file in file_names]
names = ['Train Loss IoU', 'Valid Loss IoU', 'mAP@0.5', 'mAP@0.5:0.95']

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

smooth_window = 5
smoothed_train_loss = np.convolve(dfs[0]['Value'], np.ones(smooth_window)/smooth_window, mode='valid')
smoothed_valid_loss = np.convolve(dfs[1]['Value'], np.ones(smooth_window)/smooth_window, mode='valid')
smoothed_map50 = np.convolve(dfs[2]['Value'], np.ones(smooth_window)/smooth_window, mode='valid')
smoothed_map5095 = np.convolve(dfs[3]['Value'], np.ones(smooth_window)/smooth_window, mode='valid')

color1 = '#FF9999'  # lighter red
color2 = 'tab:blue'  # typical blue
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y')
x_smoothed = dfs[0]['Step'][:-smooth_window+1]
ax1.plot(x_smoothed, smoothed_train_loss, color=color2, linestyle='--',label=names[0])

ax2 = ax1.twinx()  


ax2.set_ylabel('mAP', color='red')
ax2.tick_params(axis='y')
x_smoothed = dfs[2]['Step'][:-smooth_window+1]
ax2.plot(x_smoothed, smoothed_map50, color=color1, linestyle='--', label=names[2])

# Plot the Valid Loss IoU file
x_smoothed = dfs[1]['Step'][:-smooth_window+1]
ax1.plot(x_smoothed, smoothed_valid_loss, color='blue', label=names[1])

# Plot the mAP@0.5:0.95 file
x_smoothed = dfs[3]['Step'][:-smooth_window+1]
ax2.plot(x_smoothed, smoothed_map5095, color='red',  label=names[3])

# Set the axis label colors
ax1.yaxis.label.set_color('blue')
ax2.yaxis.label.set_color('red')

fig.tight_layout()  

plt.title('YOLO NAS Training metrics over epochs')
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax2.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))

# Show plot
plt.show()
