.PHONY: clean build run convert test

all: clean build run convert test

clean:
	@echo "Cleaning up"
	rm -f proj1A
	rm -f proj1A_out.pnm
	rm -f proj1A_out.png
	rm -f a.out
	rm -f test/proj1A_out.pnm
	rm -f test/proj1A_out.png
	rm -f test/a.out

build:
	@echo "Compiling"
	g++ proj1A.cpp -o proj1A

run:
	@echo "Running"
	./proj1A

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1A_out.pnm >> proj1A_out.png

test:
	@echo "Testing against checker"
	cp proj1A.cpp test/
	cd test/ && ./proj1A_checker && rm a.out && rm proj1A.cpp
