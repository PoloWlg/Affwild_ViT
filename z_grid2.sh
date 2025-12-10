#!/bin/sh

python main.py -gpu 0 -seed 1 -learning_rate 1e-5 -batch_size 4 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-5 -batch_size 4 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 4 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-5 -batch_size 4 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-5 -batch_size 4 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-5 -batch_size 4 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-5 -batch_size 4 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-5 -batch_size 4 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-5 -batch_size 8 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-5 -batch_size 8 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 8 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-5 -batch_size 8 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-5 -batch_size 8 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-5 -batch_size 8 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-5 -batch_size 8 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-5 -batch_size 8 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-5 -batch_size 16 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-5 -batch_size 16 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 16 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-5 -batch_size 16 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-5 -batch_size 16 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-5 -batch_size 16 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-5 -batch_size 16 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-5 -batch_size 16 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-5 -batch_size 32 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-5 -batch_size 32 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-5 -batch_size 32 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-5 -batch_size 32 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-5 -batch_size 32 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-5 -batch_size 32 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-5 -batch_size 32 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-5 -batch_size 32 -model_name Context 




###----------------------------------------------------------------------------------------


python main.py -gpu 0 -seed 1 -learning_rate 1e-4 -batch_size 4 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-4 -batch_size 4 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-4 -batch_size 4 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-4 -batch_size 4 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-4 -batch_size 4 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-4 -batch_size 4 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-4 -batch_size 4 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-4 -batch_size 4 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-4 -batch_size 8 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-4 -batch_size 8 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-4 -batch_size 8 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-4 -batch_size 8 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-4 -batch_size 8 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-4 -batch_size 8 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-4 -batch_size 8 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-4 -batch_size 8 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-4 -batch_size 16 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-4 -batch_size 16 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-4 -batch_size 16 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-4 -batch_size 16 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-4 -batch_size 16 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-4 -batch_size 16 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-4 -batch_size 16 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-4 -batch_size 16 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-4 -batch_size 32 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-4 -batch_size 32 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-4 -batch_size 32 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-4 -batch_size 32 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-4 -batch_size 32 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-4 -batch_size 32 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-4 -batch_size 32 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-4 -batch_size 32 -model_name Context 


###----------------------------------------------------------------------------------------


python main.py -gpu 0 -seed 1 -learning_rate 1e-3 -batch_size 4 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-3 -batch_size 4 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-3 -batch_size 4 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-3 -batch_size 4 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-3 -batch_size 4 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-3 -batch_size 4 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-3 -batch_size 4 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-3 -batch_size 4 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-3 -batch_size 8 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-3 -batch_size 8 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-3 -batch_size 8 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-3 -batch_size 8 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-3 -batch_size 8 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-3 -batch_size 8 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-3 -batch_size 8 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-3 -batch_size 8 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-3 -batch_size 16 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-3 -batch_size 16 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-3 -batch_size 16 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-3 -batch_size 16 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-3 -batch_size 16 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-3 -batch_size 16 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-3 -batch_size 16 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-3 -batch_size 16 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-3 -batch_size 32 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-3 -batch_size 32 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-3 -batch_size 32 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-3 -batch_size 32 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-3 -batch_size 32 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-3 -batch_size 32 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-3 -batch_size 32 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-3 -batch_size 32 -model_name Context 

###----------------------------------------------------------------------------------------


python main.py -gpu 0 -seed 1 -learning_rate 1e-6 -batch_size 4 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-6 -batch_size 4 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-6 -batch_size 4 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-6 -batch_size 4 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-6 -batch_size 4 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-6 -batch_size 4 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-6 -batch_size 4 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-6 -batch_size 4 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-6 -batch_size 8 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-6 -batch_size 8 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-6 -batch_size 8 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-6 -batch_size 8 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-6 -batch_size 8 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-6 -batch_size 8 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-6 -batch_size 8 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-6 -batch_size 8 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-6 -batch_size 16 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-6 -batch_size 16 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-6 -batch_size 16 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-6 -batch_size 16 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-6 -batch_size 16 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-6 -batch_size 16 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-6 -batch_size 16 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-6 -batch_size 16 -model_name Context 

python main.py -gpu 0 -seed 1 -learning_rate 1e-6 -batch_size 32 -model_name Video &
python main.py -gpu 0 -seed 2 -learning_rate 1e-6 -batch_size 32 -model_name Video &
 
python main.py -gpu 1 -seed 3 -learning_rate 1e-6 -batch_size 32 -model_name Video &
python main.py -gpu 1 -seed 4 -learning_rate 1e-6 -batch_size 32 -model_name Video &

python main.py -gpu 2 -seed 1 -learning_rate 1e-6 -batch_size 32 -model_name Context &
python main.py -gpu 2 -seed 2 -learning_rate 1e-6 -batch_size 32 -model_name Context &
 
python main.py -gpu 3 -seed 3 -learning_rate 1e-6 -batch_size 32 -model_name Context &
python main.py -gpu 3 -seed 4 -learning_rate 1e-6 -batch_size 32 -model_name Context 