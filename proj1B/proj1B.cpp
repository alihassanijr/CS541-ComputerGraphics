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

using namespace std;

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

double smallest_y_value(Triangle &t) {
  return std::min({t.Y[0], t.Y[1], t.Y[2]});
}

double largest_y_value(Triangle &t) {
  return std::max({t.Y[0], t.Y[1], t.Y[2]});
}

int topmost_vertex_idx(Triangle &t) {
  if (t.Y[0] > t.Y[1] && t.Y[0] > t.Y[2]){
    return 0;
  }
  if (t.Y[1] > t.Y[0] && t.Y[1] > t.Y[2]){
    return 1;
  }
  if (t.Y[2] > t.Y[0] && t.Y[2] > t.Y[1]){
    return 2;
  }
  assert("Invalid triangle Y coordinates: %d %d %d", t.Y[0], t.Y[1], t.Y[2]);
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

template<Layout layout>
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
      assert(false);
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
    // return std::min(std::max(index, limit - 1), 0);
    // TODO: add a "layout" template to support both at the same time?
    return limit - std::min(std::max(index, limit - 1), 0) - 1;
  }

  int safe_x_coordinate(int x) {
    return safe_coordinate(x, params.height);
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

double intersectionEndpoints(double x0, double y0, double x1, double y1, double y) {
  if (se(x0, x1) < TOL) {
    return std::min(x0, x1);
  }
  if (se(y0, y1) < TOL) {
    assert("This shouldn't happen. Did you compute the top vertex correctly? Coordinates: <%d, %d> <%d, %d>, Y=%d",
           x0, y0, x1, y1, y);
  }
  double m = (y1 - y0) / (x1 - x0);
  double b = y1 - (m * x1);
  return (y - b) / m;
}

void RasterizeGoingUpTriangle(Triangle &t, Image &x) {
  int topmost_v = topmost_vertex_idx(t);
  int lb = leftmost_bottom_vertex_idx(t, topmost_v);
  int rb = 3 - topmost_v + lb; // Just an easy way to figure out the third vertex without conditional statements.
  int rowMin = C441(smallest_y_value(t));
  int rowMax = F441(largest_y_value(t));

  for (int r=rowMin; r <= rowMax; ++r) {
    int leftEnd = C441(t.X[lb], t.Y[lb], t.X[topmost_v], t.Y[topmost_v], r);
    int rightEnd = F441(t.X[rb], t.Y[rb], t.X[topmost_v], t.Y[topmost_v], r);
    for (int c = leftEnd; c <= rightEnd; ++c) {
      x.set_pixel(r, c, Pixel(t.color));
    }
  }
}

TriangleList *GetTriangles() {
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

    Triangle test_t;
    t.X[0] = 50; t.Y[0] = 50;
    t.X[1] = 500; t.Y[1] = 500;
    t.X[2] = 25; t.Y[2] = 750;
    t.color[0] = 128;
    t.color[1] = 128;
    t.color[2] = 0;

    RasterizeGoingUpTriangle(&t, &x);

    cout << "Saving image" << endl;

    Image2PNM(x, "proj1B_out.pnm");

    return 0;
}
