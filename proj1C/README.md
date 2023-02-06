## Project 1-C

Extend 1-B to support arbitrary triangles.

## Description
Extend your 1B code so that your scanline algorithm works on arbitrary triangles. 

### Phase 1

Create a single triangle and verify that your algorithms works on this triangle.
The details of the triangle are up to you, but I recommend you start by having unique values for every X and Y position.
After you have that working, I recommend you try some degenerate cases (going up, going down, going left, going right).

Only proceed to Phase 2 when you truly believe your scanline algorithm is working. 

Also, note that Phase 2 has a non-square image. 
So I recommend you change your image to non-square and make sure you do not have any lurking bugs that assumes the image is square.

Finally, when you are done with Phase 1, make a copy of your code for reference.  Phase 2 can get tricky and folks sometimes contort their code to understand an issue.  It is often useful to have something you can refer back to.

### Phase 2

Download my test infrastructure and confirm your code works with my test infrastructure.

## Solution

So I was kind of right in 1B: you can split triangles into a bottom half and a top half.
I was also right in that I don't have to create two new triangles by adding in a 4th vertex.

I was, however, wrong about a few things. I had to revise my Image class a bit because I made a mistake in my indexing,
and only noticed it here because it was a non-square image.
I also had to account for cases where the "left" and "right" vertices "change places" so there should be a check and swap
in the scanline mainloop. I was off by two pixels in puddles' nose because of that.

Anyway, it all works out, but there's still plenty of cleaning up to be done, which I will get done in future phases.

To generate my solution and run the checker script on it, just do `make`.

`make clean` will clean up compiled and output files.

For more info read `Makefile`.

## Output

<img src="../assets/outputs/proj1C.png" width="600" />

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
