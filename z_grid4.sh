#!/bin/sh


python main.py -gpu 0 -seed 3 -learning_rate 1e-5 -batch_size 4 -model_name Context &
python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 4 -model_name Audio 


