.PHONY: clean build run convert test diff

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1C
	rm -f proj1C_out.pnm
	rm -f proj1C_out.png
	rm -f a.out
	rm -f test/proj1C_out.pnm
	rm -f test/proj1C_out.png
	rm -f test/a.out
	rm -f tools/a.out
	rm -f differ

build:
	@echo "Compiling"
	g++ proj1C.cpp -o proj1C

run:
	@echo "Running"
	./proj1C

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1C_out.pnm >> proj1C_out.png

test:
	@echo "Testing against checker"
	cp proj1C.cpp test/
	cd test/ && ./proj1C_checker && rm a.out && rm proj1C.cpp

diff:
	@echo "Runing differencer"
	gcc tools/differencer.c -o differ
	./differ proj1C_out.pnm test/proj1C_soln.pnm
	rm differ

