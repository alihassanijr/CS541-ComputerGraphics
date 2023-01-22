.PHONY: clean build run convert test diff

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1D
	rm -f proj1D_out.pnm
	rm -f proj1D_out.png
	rm -f a.out
	rm -f test/proj1D_out.pnm
	rm -f test/proj1D_out.png
	rm -f test/a.out
	rm -f tools/a.out
	rm -f differ

build:
	@echo "Compiling"
	g++ proj1D.cpp -o proj1D

run:
	@echo "Running"
	./proj1D

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1D_out.pnm >> proj1D_out.png

test:
	@echo "Testing against checker"
	cp proj1D.cpp test/
	cd test/ && ./proj1D_checker && rm a.out && rm proj1D.cpp

diff:
	@echo "Runing differencer"
	gcc tools/differencer.c -o differ
	./differ proj1D_out.pnm test/proj1D_soln.pnm
	rm differ

