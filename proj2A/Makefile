SHELL := /bin/bash

.PHONY: clean build run

all: clean build run

clean:
	@echo "Cleaning up"
	rm -f proj2A
	rm -f a.out

build:
	@echo "Compiling -- Assuming you're on OSX; otherwise it's Barney Rubble"
	g++ -o proj2A proj2A.cxx -lglfw -lglew -framework OpenGL

run:
	@echo "Running"
	./proj2A

