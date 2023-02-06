## Project 1-A

Implement a PNM writer and an Image class, and output an image divided into a 3x3 grid and
color the grid according to the project description.

## Description
Your job is to produce a C or C++ program that can create an image. 
Your code should be named proj1A.c (C code) or proj1A.cpp (C++ code).
It should produce an image of type PNM that is named proj1A_out.pnm.  You must follow exactly these conventions (proj1A.c/proj1A.cpp and proj1A_out.pnm) to work with our grader scripts.

Your program should produce a 300x300 pixel image as follows:

* The upper left 100x100 pixels should be black (0, 0, 0). 
* The upper middle 100x100 pixels should be gray (128, 128, 128).
* The upper right 100x100 pixels should be white (255, 255, 255).
* The middle left 100x100 pixels should be red (255, 0, 0). 
* The central 100x100 pixels should be green (0, 255, 0).
* The middle right 100x100 pixels should be blue (0, 0, 255).
* The bottom left 100x100 pixels should be purple (255, 0, 255). 
* The bottom middle 100x100 pixels should be cyan (0, 255, 255).
* The bottom right 100x100 pixels should be yellow (255, 255, 0).

## Solution
I'm going to do a Makefile for every project, just because it makes life easier.
`make` will compile, run, convert to PNG, and run the checker script in `test/`.
`make clean` will clean up compiled and output files.

For more info read `Makefile`.

To generate my solution, just do `make`.

## Output

<img src="../assets/outputs/proj1A.png" width="300" />

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
