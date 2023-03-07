SHELL := /bin/bash

.PHONY: clean build run convert

all: clean build run convert

clean:
	@echo "Cleaning up"
	@rm -f a.out
	@rm -f proj1A
	@rm -rf *.pnm
	@rm -rf *.png

build:
	@echo "Compiling"
	nvcc proj1A.cu -o proj1A

run:
	@echo "Running"
	time ./proj1A

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1A_cuda.pnm >> proj1A_cuda.png
