#!/bin/sh

python main.py -gpu 0 -seed 11 &
python main.py -gpu 0 -seed 12 &
 
python main.py -gpu 1 -seed 13 &
python main.py -gpu 1 -seed 14 &

python main.py -gpu 2 -seed 15 &
python main.py -gpu 2 -seed 16 &

python main.py -gpu 3 -seed 17 &
python main.py -gpu 3 -seed 18 


