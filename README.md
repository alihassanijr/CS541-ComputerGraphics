# CS541 - Intro to Computer Graphics
Took this class Winter 2023, and so far my favorite at UO.

## Overview
The class started off with how rasterizers work, and building our own rasterizer.
Project 1 was split into subprojects spread out over a few weeks, and in the end everyone wrote their own rasterizer in C++.
Project 2 was mostly working with OpenGL.
Final project was supposed to be either something proposed by the student, or doing 3 mini projects from a list of pre-defined
projects.
Grad students also had to contribute a mini project to the pre-defined project list, and do a 20 minute lecture on it.
It was also possible to request contributing both a project and doing an 80 minute lecture, and have that count as the final
project, which is what I did.

## Project 1

Project 1 is the basics: implementing the scanline algorithm, color interpolation, z-buffer, and 3D transformations.
It was previously set up with VTK, but that became a challenge over time so now we're writing things from scratch.

**Project 1-A:** PNM writer and a sample image

<img src="assets/outputs/proj1A.png" width="300" />

**Project 1-B:** Implement a limited rasterizer able to plot upward triangles 
(two vertices have an identical y value, and the third has a higher y value.)

<img src="assets/outputs/proj1B.png" width="300" />

**Project 1-C:** Extend 1-B to support arbitrary triangles.

<img src="assets/outputs/proj1C.png" width="300" />

**Project 1-D:** Implement color interpolation and z-buffer to support depth.

<img src="assets/outputs/proj1D.png" width="300" />

**Project 1-E:** Implement a Camera class and calculate transformations.

<img src="assets/outputs/proj1E_frame0000.png" width="300" />
<img src="assets/outputs/proj1E_frame0250.png" width="300" />
<img src="assets/outputs/proj1E_frame0500.png" width="300" />
<img src="assets/outputs/proj1E_frame0750.png" width="300" />

**Project 1-F:** Implement shading and generate a video.

<img src="assets/outputs/proj1F.png" width="300" />

[Link to video](https://ix.cs.uoregon.edu/~alih/proj1F.mp4)

## Project 2
Start using OpenGL.

**Project 2-A:** Re-implement 1-F using OpenGL.

<img src="assets/outputs/proj2A.png" width="300" />

**Project 2-B:** Build a dog with spheres and cylinders.

<img src="assets/outputs/proj2B.png" width="300" />

## Project G

Project G is one where graduate students contribute a project to the class that the undergraduates can pick and complete for
their final project. 
Of course, I picked CUDA, always fun.

[projG](projG/README.md) holds my contributed prompt, starter code I provided, and the solution (what I reduced to get the
starter code.) Obviously I didn't write even a good naive rasterizer, but it was the best I could do in a reasonable amount of
time, and what would also be easy enough for the students to get done within 8-10 hours.

[projG-livecode](projG-livecode/README.md) is the live code I did in class when presenting.

My presentation (80 minute lecture) is in [presentation/](presentation/README.md).

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
So far one of my favorite courses in grad school!
