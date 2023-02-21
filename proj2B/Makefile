SHELL := /bin/bash

.PHONY: clean build run

all: clean build run

twosided: clean build-twosided run

clean:
	@echo "Cleaning up"
	rm -f proj2B
	rm -f a.out

build:
	@echo "Compiling"
	g++ -o proj2B proj2B.cxx -lglfw -lglew -framework OpenGL

build-twosided:
	@echo "Compiling with two sided lighting"
	g++ -o proj2B proj2B.cxx -lglfw -lglew -framework OpenGL -DTWOSIDED

run:
	@echo "Running"
	./proj2B

