#!/bin/sh
python main.py -gpu 0 -seed 1 &
python main.py -gpu 1 -seed 2 &

python main.py -gpu 2 -seed 3 &
python main.py -gpu 3 -seed 4  


