#!/bin/bash

python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 3 -batch_size 16 -learning_rate 0.00001 -unfreeze_all_clip 1
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 3 -batch_size 32 -learning_rate 0.00001 -unfreeze_all_clip 1
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 3 -batch_size 64 -learning_rate 0.00001 -unfreeze_all_clip 1
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 3 -batch_size 128 -learning_rate 0.00001 -unfreeze_all_clip 1
