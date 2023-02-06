## Project 1-F
There are two parts to this project: (1) extend your 1E code to do Phong shading as per class lecture and (2) generate a movie.
Use the same data file as 1E.

## Description
There is starter code which defines a data structure that contains the parameters for shading. 
I pasted the contents of this file into my code, and encourage you all to do the same. 
This file also contains a function called `GetLighting()`.
This function should be called for every render, since the light position updates with the camera.

You do not need new reader code or a new geometry file. 
That said, the code you got from 1E's starter code has `#ifdef NORMALS` in it.
You need those normals now, so add `#define NORMALS` early in your code.
Note you will need to add `double normals[3][3];` as a data member in your Triangle struct.  

Normals is indexed by the vertex first and the dimension second.

```
int vertexId = 0;
int x = 0, y = 1, z =2;
normals[vertexId][y] = …;
```

Note: I also added a `double shading[3];` data member to Triangle. 
I found this to be a helpful location to store per-vertex shading information.

### One-sided lighting

You will do one-sided lighting, so it is very important you get the conventions correct.

Conventions on vector directions:

1. The light source is coming from the triangles. Explicitly, if a triangle vertex is at (0,0,0) and if the light source is at (10,0,0), then the light direction is (1, 0, 0).

2. The view direction is coming from a triangle vertex. Explicitly, if a triangle vertex is at (0,0,0) and if the camera is at (0,10,0), then the view direction is (0, 1, 0).


## Solution

I pretty much anticipated the structure, and wrote most of it during the lecture, but I had to debug a bit afterwards,
and look back at the slides to make sure I wasn't missing anything.
Ended up spending more time stuck on a ~1000 pixel difference that was due to forgetting to swap the shadings when left
and right vertices are swapped in scanline ( :facepalm: ).

Anyway, it finally came together and I generated all the frames, set up ffmpeg locally, and set up new make targets
for video.

To generate my solution and run the checker script on it, just do `make`.

To generate my video (this might take a while), run `make video`.
Also look at my `Makefile` because I define macros to customize the video a bit and not mess up the image solution.

`make clean` will clean up compiled and output files.

For more info read `Makefile`.

## Output

<img src="../assets/outputs/proj1F.png" width="500" />

[Link to video](http://ix.cs.uroregon.edu/~alih/proj1F.mp4)

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
