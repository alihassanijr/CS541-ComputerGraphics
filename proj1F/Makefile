.PHONY: clean build run convert test

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1F
	rm -f proj1F_out.pnm
	rm -f proj1F_out.png
	rm -f a.out
	rm -f test/proj1F_out.pnm
	rm -f test/proj1F_out.png
	rm -f test/a.out
	rm -f test/soln.png
	rm -f tools/a.out
	rm -f differ
	rm -f diff.png
	rm -f differences.pnm
	rm -f proj1F_frame*.png
	rm -f proj1F_frame*.pnm
	rm -f test/proj1F_frame*.png
	rm -f test/proj1F_frame*.pnm
	rm -rf test/counter

build:
	@echo "Compiling"
	g++ proj1F.cpp -o proj1F

run:
	@echo "Running"
	./proj1F

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1F_frame0000.pnm >> proj1F_frame0000.png

test:
	@echo "Testing against reference"
	cp proj1F.cpp test/
	cd test/ && ./proj1F_checker && rm a.out && rm proj1F.cpp
