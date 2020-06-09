#!/bin/bash
for thread in 1 2 4 6 8
do
	# number of nodes
	for try in 1 5 10
	do
		g++ -DNUMT=$thread -DNUMTRIES=$try -lm -fopenmp mainOMP.cpp -o out
		./out
	done
done