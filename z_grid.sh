#!/bin/sh

python main.py -gpu 0 -seed 1 &
python main.py -gpu 0 -seed 2 &
 
python main.py -gpu 1 -seed 3 &
python main.py -gpu 1 -seed 4 &

python main.py -gpu 2 -seed 5 &
python main.py -gpu 2 -seed 6 &

python main.py -gpu 3 -seed 7 &
python main.py -gpu 3 -seed 8

python main.py -gpu 0 -seed 9 &
python main.py -gpu 0 -seed 10 &
 
python main.py -gpu 1 -seed 11 &
python main.py -gpu 1 -seed 12 &

python main.py -gpu 2 -seed 13 &
python main.py -gpu 2 -seed 14 &

python main.py -gpu 3 -seed 15 &
python main.py -gpu 3 -seed 16 

python main.py -gpu 0 -seed 17 &
python main.py -gpu 0 -seed 18 &
 
python main.py -gpu 1 -seed 19 &
python main.py -gpu 1 -seed 20 
