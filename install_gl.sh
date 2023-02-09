#!/bin/bash

THISDIR=$(pwd)
BUILDDIR=$THISDIR/build
LOCALDIR=$THISDIR/local
TMPDIR=$THISDIR/tmp

# Ensure submodules are cloned
git submodule update --init --recursive

# Remove existing build and lib directory
rm -rf $BUILDDIR
rm -rf $LOCALDIR
rm -rf $TMPDIR

mkdir -p $LOCALDIR/include
mkdir -p $TMPDIR

# Install GLFW
mkdir -p $BUILDDIR/glfw
cd $THISDIR/third_party/opengl/glfw && \
       cmake -G "Unix Makefiles" -B $BUILDDIR/glfw/ -D CMAKE_INSTALL_PREFIX=$LOCALDIR
cd $BUILDDIR/glfw/src && make && make install
#cmake  -S $THISDIR/third_party/opengl/glfw -B $BUILDDIR/glfw/ -D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=$LOCALDIR
#cd $BUILDDIR/glfw/src && make && make install
#cp -r $THISDIR/third_party/opengl/glfw/include/GLFW $LOCALDIR/include/GLFW


# Install GLEW
#mkdir -p $BUILDDIR
#cd $BUILDDIR && \
#       wget https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz && \
#       tar -xzf glew-2.2.0.tgz &&
#       rm glew-2.2.0.tgz && \
#       mv glew-2.2.0/ glew
#cd $BUILDDIR/glew/build && \
#       cmake ./cmake -D CMAKE_INSTALL_PREFIX=$LOCALDIR && \
#       make -j4
## It's almost midnight -- zero patience
#mv $BUILDDIR/glew/build/bin $LOCALDIR/
#mv $BUILDDIR/glew/build/lib $LOCALDIR/
mkdir -p $TMPDIR/glew
cd $TMPDIR/glew && \
       wget https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz && \
       tar -xzf glew-2.2.0.tgz &&
       rm glew-2.2.0.tgz
cd $TMPDIR/glew/glew-2.2.0/ && \
       make && \
       make install GLEW_DEST=$THISDIR/local

# Install GLM
#mkdir -p $BUILDDIR/glm
mkdir -p $TMPDIR/glm
cd $TMPDIR/glm && \
       wget https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip && \
       unzip -q glm-0.9.9.8.zip && \
       rm glm-0.9.9.8.zip
mv $TMPDIR/glm/glm/glm $LOCALDIR/include/glm
#cmake -S $TMPDIR/glm/glm -B $BUILDDIR/glm -D CMAKE_INSTALL_PREFIX=$LOCALDIR
#cd $BUILDDIR/glm && make && make install
       
# Cleanup
#rm -rf $BUILDDIR
rm -rf $TMPDIR
