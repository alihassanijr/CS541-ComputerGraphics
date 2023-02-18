SHELL := /bin/bash

.PHONY: clean build run

all: clean build run

clean:
	@echo "Cleaning up"
	rm -f proj2B
	rm -f a.out

build:
	@echo "Compiling -- Assuming you're on OSX; otherwise it's Barney Rubble"
	g++ -o proj2B proj2B.cxx -lglfw -lglew -framework OpenGL

run:
	@echo "Running"
	./proj2B

