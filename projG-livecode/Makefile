SHELL := /bin/bash

.PHONY: clean build run

all: clean build run

cpu: clean build-cpu run

clean:
	@echo "Cleaning up"
	@rm -f a.out
	@rm -f helloworld 

build:
	@echo "Compiling"
	nvcc helloworld.cu -DUSE_CUDA -o helloworld 

build-cpu:
	@echo "Compiling"
	nvcc helloworld.cu -o helloworld 

run:
	@echo "Running"
	time ./helloworld
