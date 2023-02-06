## Project 1-E
Extend your code to support arbitrary camera positions using the approach discussed in lecture 
(camera transform, view transform, device space transform).

## Description

Your code must produce 4 images, called proj1E_frame0000.pnm, proj1E_frame0250.pnm, proj1E_frame0500.pnm, 
and proj1E_frame0750.pnm.  ("sprintf" with "%04d" formatting is useful for generating the names.)  
The exact positions for each camera are calculated from the starter code: 
GetCamera(0, 1000), GetCamera(250, 1000), GetCamera(500, 1000), and GetCamera(750, 1000). 

## Solution

I cleaned up a bit more, and because I'm obsessed, I wrote linalg classes like Vector and Matrix, wrote operators
like norm, sum, dot product, and matmul (and templated -- some of these were totally unnecessary).

Setting up the transforms was pretty much following the lectures, nothing special.
Didn't even take that long.

When the project was finally posted and I saw the input data, I just plugged it in and noticed the whole 
output looked warped. I ended up looking at 
[Ken Joy's graphics notes](http://www.idav.ucdavis.edu/education/GraphicsNotes/homepage.html)
and found I'm not normalizing the w and u vectors in the camera transform.
I ended up normalizing all three (w, u, and v), but I'm guessing the last one isn't even necessary,
since I imagine the cross product of two unit vectors is a unit vector (can't say that I recall my high school linalg.)

To generate my solution and run the checker script on it, just do `make`.

`make clean` will clean up compiled and output files.

For more info read `Makefile`.

## Output

<img src="../assets/outputs/proj1E_frame0000.png" width="300" />
<img src="../assets/outputs/proj1E_frame0250.png" width="300" />
<img src="../assets/outputs/proj1E_frame0500.png" width="300" />
<img src="../assets/outputs/proj1E_frame0750.png" width="300" />

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
