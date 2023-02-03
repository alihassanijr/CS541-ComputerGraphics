SHELL := /bin/bash

.PHONY: clean build run convert test

all: clean build run convert test

video: clean build-video run convert-video generate-video

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
	rm -rf frames/

build:
	@echo "Compiling"
	g++ proj1F.cpp -o proj1F

build-video:
	@echo "Compiling"
	mkdir -p frames/
	g++ proj1F.cpp -DVIDEO -DVERBOSE -o proj1F

run:
	@echo "Running"
	./proj1F

convert-video:
	@echo "Converting frames to PNG"
	for f in frames/*.pnm ; do \
		echo "$$f >> $${f/pnm/png}"; \
		pnmtopng $$f >> $${f/pnm/png}; \
	done

generate-video:
	@echo "Converting to video using FFMPEG"
	ffmpeg -r 25 -f image2 -s 1000x1000 -i frames/proj1F_frame%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p proj1F.mp4

convert:
	@echo "Converting output to PNG"
	pnmtopng proj1F_frame0000.pnm >> proj1F_frame0000.png

test:
	@echo "Testing against reference"
	cp proj1F.cpp test/
	cd test/ && ./proj1F_checker && rm a.out && rm proj1F.cpp
