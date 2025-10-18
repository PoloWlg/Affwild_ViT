
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_folder = '/home/ens/AS84330/Context/ABAW3_EXPR4/weights_saved'
txt_path = os.path.join(root_folder, 'val_f1_scores.txt')

train_f1_list = []
val_f1_list = []
val_f1_list_max = []

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file == 'training_logs.csv':
            csv_path = os.path.join(root, file)
            
            df = pd.read_csv(csv_path, skiprows=4)
            val_f1 =  df['val_f1'].to_numpy()
            train_f1 = df['tr_f1'].to_numpy()
            
            # append the values a np array
            
            train_f1_list.append(train_f1)
            val_f1_list.append(val_f1)
            
            val_f1_list_max.append(np.max(val_f1))
            

# Display learning curves for train and val

# Save in a txt file
with open(txt_path, 'w') as f:
    f.write('Max val f1 scores: ' + str(np.max(val_f1_list_max)) + '\n')
    f.write('Min val f1 scores: ' + str(np.min(val_f1_list_max)) + '\n')
    f.write('Avg val f1 scores: ' + str(np.mean(val_f1_list_max)) + '\n')
    f.write('Std val f1 scores: ' + str(np.std(val_f1_list_max)) + '\n')

plt.figure(figsize=(10, 5))

for index, seed in enumerate(val_f1_list):
    plt.plot(seed, label=f'seed {index}')
    
    # Put a dot at the maximum of the curve with same color as the curve 
    max_idx = np.argmax(seed)
    max_val = seed[max_idx]
    plt.scatter(max_idx, max_val)
plt.title('Learning Curves Val')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder,'learning_curves_val.png'))


plt.figure(figsize=(10, 5))

for index, seed in enumerate(train_f1_list):
    plt.plot(seed, label=f'seed {index}')

    # Put a dot at the maximum of the curve with same color as the curve 
    
    max_idx = np.argmax(seed)
    max_val = seed[max_idx]
    plt.scatter(max_idx, max_val)
plt.title('Learning Curves Train')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder,'learning_curves_train.png'))