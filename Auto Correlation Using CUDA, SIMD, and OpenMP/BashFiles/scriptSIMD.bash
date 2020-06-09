#!/bin/bash
for try in 1 5 10 50
do
    g++ -DNUMTRIES=$try -lm -fopenmp mainSIMD.cpp -o out
    ./out
done