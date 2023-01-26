.PHONY: clean build run convert test diff

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1E
	rm -f proj1E_out.pnm
	rm -f proj1E_out.png
	rm -f a.out
	rm -f test/proj1E_out.pnm
	rm -f test/proj1E_out.png
	rm -f test/a.out
	rm -f test/soln.png
	rm -f tools/a.out
	rm -f differ
	rm -f diff.png
	rm -f differences.pnm

build:
	@echo "Compiling"
	g++ proj1E.cpp -o proj1E

run:
	@echo "Running"
	./proj1E

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1E_out.pnm >> proj1E_out.png

test:
	@echo "Testing against reference"
	cp proj1E.cpp test/
	cd test/ && ./proj1E_checker && rm a.out && rm proj1E.cpp

diff:
	@echo "Runing differencer"
	gcc tools/differencer.c -o differ
	./differ proj1E_out.pnm test/proj1E_soln.pnm
	rm differ
