.PHONY: clean build run convert test

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1B
	rm -f proj1B_out.pnm
	rm -f proj1B_out.png
	rm -f a.out
	rm -f test/proj1B_out.pnm
	rm -f test/proj1B_out.png
	rm -f test/a.out

build:
	@echo "Compiling"
	g++ proj1B.cpp -o proj1B

run:
	@echo "Running"
	./proj1B

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1B_out.pnm >> proj1B_out.png

test:
	@echo "Testing against checker"
	cp proj1B.cpp test/
	cd test/ && ./proj1B_checker && rm a.out && rm proj1B.cpp
