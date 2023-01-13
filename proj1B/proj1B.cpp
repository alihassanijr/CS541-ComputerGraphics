/*

Ali Hassani

Project 1-B

CS 441/541

*/
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <math.h>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

const double TOL = 0.00001;

double C441(double f) {
  return ceil(f-0.00001);
}

double F441(double f) {
  return floor(f+0.00001);
}

struct Triangle {
  double         X[3];
  double         Y[3];
  unsigned char color[3];
};

/// This returns vertex indices sorted by Y value in descending order
//// It's been a while and it took me a while to remember the best way to sort and get 
//// indices, so I just ended up hard coding it.
//// I'll fix it in future projects, I swear. I know how terrible this is.
vector<int> sorted_y_idx(Triangle &t) {
  if (t.Y[0] >= t.Y[1] && t.Y[1] >= t.Y[2])
    return {0, 1, 2};
  if (t.Y[0] >= t.Y[2] && t.Y[2] >= t.Y[1])
    return {0, 2, 1};
  if (t.Y[1] >= t.Y[0] && t.Y[0] >= t.Y[2])
    return {1, 0, 2};
  if (t.Y[1] >= t.Y[2] && t.Y[2] >= t.Y[0])
    return {1, 2, 0};
  if (t.Y[2] >= t.Y[1] && t.Y[1] >= t.Y[0])
    return {2, 1, 0};
  if (t.Y[2] >= t.Y[0] && t.Y[0] >= t.Y[1])
    return {2, 0, 1};
  cerr << "Sort failed! " << endl;
  terminate();
}

struct Pixel {
  unsigned char r, g, b;

  Pixel(): r(0), g(0), b(0) {}

  Pixel(unsigned char r, unsigned char g, unsigned char b): r(r), g(g), b(b) {}

  Pixel(unsigned char *color): r(color[0]), g(color[1]), b(color[2]) {}

  void set_value(unsigned char r_, unsigned char g_, unsigned char b_) {
    r = r_;
    g = g_;
    b = b_;
  }

  void set_value(Pixel p) {
    set_value(p.r, p.g, p.b);
  }

  void zfill() {
    set_value(0, 0, 0);
  }

  unsigned char get_r() {
    return r;
  }

  unsigned char get_g() {
    return g;
  }

  unsigned char get_b() {
    return b;
  }
};

struct Image {
  struct Params {
    int height, width;

    Params(int h, int w): height(h), width(w) { }

    int numel() {
      return height * width;
    }

    int stride(int dim) {
      if (dim == 0) {
        return width;
      } else if (dim == 1) {
        return 1;
      }
      cerr << "Valid dimensions for an image are 0 and 1, got " << dim << endl;
      terminate();
    }
  };

private:
  Pixel *_arr;

public:
  Params params;

  Image(): params(Params(0, 0)) {
    _arr = nullptr;
  }

  Image(int height, int width): params(Params(height, width)) {
    _arr = new Pixel[params.numel()];
    zfill();
  }

  void zfill() {
    for (int i=0; i < params.numel(); ++i) {
      _arr[i].zfill();
    }
  }

  int safe_coordinate(int index, int limit) {
    if (index >= 0 && index < limit)
      return index;
    return -1;
  }

  int safe_x_coordinate(int x) {
    int safe_x = safe_coordinate(x, params.height);
    if (safe_x < 0)
      return -1;
    // TODO: add a "layout" template to support both coordinate formats (even more) at the same time?
    return params.height - 1 - safe_x;
    //return safe_coordinate(x, params.height);
  }

  int safe_y_coordinate(int y) {
    return safe_coordinate(y, params.width);
  }

  int safe_limit(int index, int limit) {
    return std::min(std::max(index, limit), 0);
  }

  int safe_x_limit(int x) {
    return safe_limit(x, params.height);
  }

  int safe_y_limit(int y) {
    return safe_limit(y, params.width);
  }

  void set_pixel(int x, int y, Pixel v) {
    int x_ = safe_x_coordinate(x);
    int y_ = safe_y_coordinate(y);
    if (x_ < 0 || y_ < 0)
      return;
    _arr[x_ * params.stride(0) + y_].set_value(v);
  }

  void set_pixels(int x_start, int y_start, int x_end, int y_end, Pixel v) {
    int x_start_ = safe_x_coordinate(x_start);
    int y_start_ = safe_y_coordinate(y_start);
    int x_end_ = safe_x_limit(x_end);
    int y_end_ = safe_y_limit(y_end);
    for (int x = x_start_; x < x_end_; ++x) {
      for (int y = y_start_; y < y_end_; ++y) {
        _arr[x * params.stride(0) + y].set_value(v);
      }
    }
  }

  Pixel get_pixel(int x, int y) {
    return _arr[x * params.stride(0) + y];
  }

  Pixel* get_data() {
    return _arr;
  }

};

void Image2PNM(Image img, string fn) {
  const char* format = "P6";
  const char* maxval = "255";
  int height = img.params.height;
  int width = img.params.width;
  int numel = img.params.numel();
  FILE *f = fopen(fn.c_str(), "wb");
  assert(f != NULL);
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n", height, width);
  fprintf(f, "%d\n", 255);
  fwrite(img.get_data(), height * width, sizeof(Pixel), f);
  fclose(f);
}

struct TriangleList {
  int numTriangles;
  Triangle *triangles;
};

/// Squared difference will be used when checking for edge cases in the next function.
//// Comparing squared difference to a threshold should be --safer-- than just comparing if two
//// doubles are the same?
template <typename T>
T squared_difference(T a, T b) {
  T diff = a - b;
  return diff * diff;
}

/// The magic function.
//// Surprisingly, most of my time went out the window, not because of this,
//// but because I was stupid enough to check if X coords are within bounds, but did Y wrong.
//// and as a result, I had a 3 pixel (out of 1000) difference with the solution that wasted an hour.
double intersectionEndpoint(double x0, double y0, double x1, double y1, double y) {
  if (squared_difference(x0, x1) < TOL) {
    return std::min(x0, x1);
  }
  if (squared_difference(y0, y1) < TOL) {
    cerr << "This shouldn't happen. Did you compute the top vertex correctly?";
    cerr << "Coordinates: <" << x0 << ", " << y0 << "> <" << x1 << ", " << y1 << ">, Y= " << y << endl;
    terminate();
  }
  double m = (y1 - y0) / (x1 - x0);
  double b = y1 - (m * x1);
  return (y - b) / m;
}

/// RasterizeGoingUpTriangle
//// Unless I'm totally wrong, this should cover more than just this project where the
//// two bottom-most vertices are aligned. At least my dummy example was such a triangle and
//// worked like a charm.
void RasterizeGoingUpTriangle(Triangle &t, Image &x) {
  vector<int> sorted_idx = sorted_y_idx(t);
  // We're going to need these sorted vertices later, but we could just use them
  // here instead of recalculating rowMin and rowMax.
  int rowMin = C441(t.Y[sorted_idx[2]]);
  int rowMax = F441(t.Y[sorted_idx[0]]);

  // Bottom half (non-existent in 1-B triangles)
  for (int r=rowMin; r < C441(t.Y[sorted_idx[1]]); ++r) {
    int anchor = sorted_idx[2]; // In the bottom half our anchor is the bottom most vertex
    int leftv, rightv;
    // Again, seemed more trouble than it was worth compared to a simple if-else
    if (t.X[sorted_idx[1]] < t.X[sorted_idx[0]]) {
      leftv = sorted_idx[1];
      rightv = sorted_idx[0];
    } else {
      leftv = sorted_idx[0];
      rightv = sorted_idx[1];
    }

    int leftEnd  = C441(intersectionEndpoint(t.X[leftv], t.Y[leftv], t.X[anchor], t.Y[anchor], r));
    int rightEnd = F441(intersectionEndpoint(t.X[rightv], t.Y[rightv], t.X[anchor], t.Y[anchor], r));
    for (int c = leftEnd; c <= rightEnd; ++c) {
      x.set_pixel(r, c, Pixel(t.color));
    }
  }

  // Top half
  for (int r=F441(t.Y[sorted_idx[1]]); r <= rowMax; ++r) {
    int anchor = sorted_idx[0]; // In the top half our anchor is the top most vertex
    int leftv, rightv;
    // Again, seemed more trouble than it was worth compared to a simple if-else
    if (t.X[sorted_idx[1]] < t.X[sorted_idx[2]]) {
      leftv = sorted_idx[1];
      rightv = sorted_idx[2];
    } else {
      leftv = sorted_idx[2];
      rightv = sorted_idx[1];
    }

    int leftEnd  = C441(intersectionEndpoint(t.X[leftv], t.Y[leftv], t.X[anchor], t.Y[anchor], r));
    int rightEnd = F441(intersectionEndpoint(t.X[rightv], t.Y[rightv], t.X[anchor], t.Y[anchor], r));
    for (int c = leftEnd; c <= rightEnd; ++c) {
      x.set_pixel(r, c, Pixel(t.color));
    }
  }
}

TriangleList GetTriangles() {
  TriangleList tl;
  tl.numTriangles = 100;
  tl.triangles = new Triangle[100];
 
  unsigned char colors[6][3] = { {255,128,0}, {255, 0, 127}, {0,204,204}, 
                                 {76,153,0}, {255, 204, 204}, {204, 204, 0}};
  for (int i = 0 ; i < 100 ; i++) {
      int idxI = i % 10;
      int posI = idxI*100;
      int idxJ = i/10;
      int posJ = idxJ*100;
      int firstPt = (i % 3);
      tl.triangles[i].X[firstPt] = posI;
      if (i == 50) {
          tl.triangles[i].X[firstPt] = -10;
      }
      tl.triangles[i].Y[firstPt] = posJ+10*(idxJ+1);
      tl.triangles[i].X[(firstPt+1) % 3] = posI+105;
      tl.triangles[i].Y[(firstPt+1) % 3] = posJ;
      tl.triangles[i].X[(firstPt+2) % 3] = posI+i;
      tl.triangles[i].Y[(firstPt+2) % 3] = posJ;
      if (i == 95) {
         tl.triangles[i].Y[firstPt] = 1050;
      }
      tl.triangles[i].color[0] = colors[i % 6][0];
      tl.triangles[i].color[1] = colors[i % 6][1];
      tl.triangles[i].color[2] = colors[i % 6][2];
  }
 
  return tl;
}


int main() {
    cout << "Generating image" << endl;

    Image x = Image(1000, 1000);

    TriangleList list = GetTriangles();

    for (int i=0; i < list.numTriangles; ++i)
      RasterizeGoingUpTriangle(list.triangles[i], x);

    cout << "Saving image" << endl;

    Image2PNM(x, "proj1B_out.pnm");

    return 0;
}
