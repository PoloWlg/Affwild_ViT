#!/bin/bash

python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 0 -batch_size 2 -learning_rate 0.0001 
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 0 -batch_size 4 -learning_rate 0.0001 
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 0 -batch_size 8 -learning_rate 0.0001 
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 0 -batch_size 16 -learning_rate 0.0001 
python /home/ens/AS84330/Stimuli/Affwild/ABAW3_EXPR4/main.py -gpu 0 -batch_size 32 -learning_rate 0.0001 





