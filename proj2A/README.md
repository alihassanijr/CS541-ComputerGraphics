## Project 2-A
Redo [Project 1-F](../proj1F/README.md) using OpenGL.

## Requirements
This project requires OpenGL, and `glew`, `glm`, and `glfw`.
On Mac, you can simply install all 3 via homebrew:
```
brew install glew glfw glm
```

I tried building them locally on Ubuntu, but I failed and eventually gave up because even
if I succeeded, I was on a server with no gui so tough luck.
And I tried local builds because I didn't have sudo on that server.

## Solution

To generate my solution on OSX (assuming `glew`, `glm`, and `glfw` are available), just do `make`.

If you're not on OSX, you probably want to try the `g++` build without the `-framework OpenGL` flag.

`make clean` will clean up compiled and output files.

For more info read `Makefile`.

## Output

<img src="../assets/outputs/proj2A.png" width="500" />

## Credits
CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
