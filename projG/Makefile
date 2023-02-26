SHELL := /bin/bash

.PHONY: clean build run convert

all: clean build run convert

video: clean build-video run convert-video generate-video

benchmark: build-benchmark run

clean:
	@echo "Cleaning up"
	@rm -f projG
	@rm -f projG_out.pnm
	@rm -f projG_out.png
	@rm -f a.out
	@rm -f test/projG_out.pnm
	@rm -f test/projG_out.png
	@rm -f test/a.out
	@rm -f test/soln.png
	@rm -f tools/a.out
	@rm -f differ
	@rm -f diff.png
	@rm -f differences.pnm
	@rm -f projG_frame*.png
	@rm -f projG_frame*.pnm
	@rm -f test/projG_frame*.png
	@rm -f test/projG_frame*.pnm
	@rm -rf test/counter
	@rm -rf frames/
	@rm -rf projG.mp4

build:
	@echo "Compiling"
	nvcc projG.cu -o projG

build-video:
	@echo "Compiling"
	mkdir -p frames/
	nvcc projG.cu -DVIDEO -DVERBOSE -o projG

build-benchmark:
	nvcc projG.cu -DVIDEO -DVERBOSE -DNOWRITE -o projG

run:
	@echo "Running"
	./projG

convert-video:
	@echo "Converting frames to PNG"
	for f in frames/*.pnm ; do \
		echo "$$f >> $${f/pnm/png}"; \
		pnmtopng $$f >> $${f/pnm/png}; \
	done

generate-video:
	@echo "Converting to video using FFMPEG"
	ffmpeg -r 45 -f image2 -s 1000x1000 -i frames/projG_frame%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p projG.mp4

convert:
	@echo "Converting output to PNG"
	pnmtopng projG_frame0000.pnm >> projG_frame0000.png
