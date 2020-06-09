###
### @file - Makefile. Require FopenMp
### @author - Christopher Mello
### @description - Makefile
###

#
# Parallel programming MakeFile
#

output: main.o
	g++-9 -lm -fopenmp main.o -o out

main.o: main.cpp
	g++-9 -lm -fopenmp -c main.cpp

clean:
	rm *.o out

