#!/bin/sh


python main.py -gpu 0 -seed 1 -learning_rate 1e-5 -batch_size 4 -model_name CAN &
python main.py -gpu 0 -seed 2 -learning_rate 1e-5 -batch_size 4 -model_name CAN &

python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 4 -model_name CAN &
python main.py -gpu 1 -seed 4 -learning_rate 1e-5 -batch_size 4 -model_name CAN &

python main.py -gpu 2 -seed 5 -learning_rate 1e-5 -batch_size 4 -model_name CAN &
python main.py -gpu 2 -seed 6 -learning_rate 1e-5 -batch_size 4 -model_name CAN &

python main.py -gpu 3 -seed 7 -learning_rate 1e-5 -batch_size 4 -model_name CAN &
python main.py -gpu 3 -seed 8 -learning_rate 1e-5 -batch_size 4 -model_name CAN 


