/*

Ali Hassani

Project 1-D

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

namespace project {

////////////////////////////////////////////////////////////////////////////////////////

namespace math {
// namespace math contains all math operations.

/// Value swap
template <typename T>
void swap(T* a, T* b) {
  T tmp = a[0];
  a[0] = b[0];
  b[0] = tmp;
}

/// Conditional swap
template <typename T>
void swap(T* a, T* b, bool condition) {
  if (condition) {
    swap<T>(a, b);
  }
}

/// Linear interpolation (lerp.)
template <typename T, typename value_t>
value_t lerp(T a, T b, value_t f_a, value_t f_b, T c) {
  T t = (c - a) / (b - a);
  return f_a + value_t(t * (f_b - f_a));
}

/// CS441 Ceil function.
double C441(double f) {
  return ceil(f-0.00001);
}

/// CS441 Floor function.
double F441(double f) {
  return floor(f+0.00001);
}

/// argsort returns the sorted indices of an array in ascending order.
template <typename data_t, typename index_t>
index_t* argsort(data_t *array, const index_t length) {
  index_t* index = new index_t[length];
  for (index_t i=0; i < length; ++i) {
    index[i] = i;
  }
  sort(index, index + length,
      [&](const index_t &i, const index_t &j) {
        return (array[i] < array[j]);
      }
  );
  return index;
}

/// abs_difference: returns the absolute difference of two values.
template <typename T>
T abs_difference(T a, T b) {
  T diff = a - b;
  if (diff < 0)
    return -1 * diff;
  return diff;
}

} // namespace math

////////////////////////////////////////////////////////////////////////////////////////

namespace ops {
// namespace ops

/// clip_sort takes in an array of sorted indices (sorted vertex indices) and removes a specific index.
//// This is useful because we can pre-sort vertices once w.r.t. X and once w.r.t. Y.
//// The Y sort tells us which vertex is the top-most, middle, and bottom-most.
//// Then we remove the top-most from the X sort and figure out the left and right vertices for the top half
//// triangle, and we remove the bottom-most one and figure out the left and right vertices for the bottom half
//// triangle.
template <typename index_t>
index_t* clip_sort(index_t* index, const index_t length, index_t exclude) {
  index_t* index_out = new index_t[length - 1];
  index_t j = 0;
  for (index_t i=0; i < length; ++i) {
    if (index[i] != exclude && j < length - 1) {
      index_out[j] = index[i];
      ++j;
    } else if (j >= length - 1) {
      break;
    }
  }
  return index_out;
}

}

////////////////////////////////////////////////////////////////////////////////////////

namespace triangles {
// namespace triangles

/// Vertex
//// I like to define a Vertex struct because it makes it so much easier to pass an array of pointers
//// when we're pre-sorting all vertices.
struct Vertex {
  double* X;
  double* Y;

  Vertex(double &x, double &y) {
    X = &x;
    Y = &y;
  }

  double x() {
    return X[0];
  }
  double y() {
    return Y[0];
  }
};

/// Triangle
//// As defined in the starter code, with the pre-sort logic added.
//// The pre-sorter is just a method that gets called once before rasterization.
struct Triangle {
  double         X[3];
  double         Y[3];
  unsigned char color[3];

  bool sorted = false; /* Checks if it's already been sorted (assume Triangle coordinates do not change while rasterizing) */
  int* sorted_X_top; /* Left and right vertices in the top half of the triangle */
  int* sorted_X_bottom; /* Left and right vertices in the bottom half of the triangle */
  int* sorted_Y; /* Top, middle, bottom vertices. */

  void precompute_sorts() {
    if (!sorted) {
      sorted = true;
      sorted_Y        = math::argsort<double, int>(Y, 3);
      int* sorted_X   = math::argsort<double, int>(X, 3);
      sorted_X_top    = ops::clip_sort<int>(sorted_X, 3, sorted_Y[2]);
      sorted_X_bottom = ops::clip_sort<int>(sorted_X, 3, sorted_Y[0]);
    }
  }

  Vertex top() {
    return Vertex(X[sorted_Y[2]], Y[sorted_Y[2]]);
  }
  Vertex middle() {
    return Vertex(X[sorted_Y[1]], Y[sorted_Y[1]]);
  }
  Vertex bottom() {
    return Vertex(X[sorted_Y[0]], Y[sorted_Y[0]]);
  }

  Vertex top_left() {
    return Vertex(X[sorted_X_top[0]], Y[sorted_X_top[0]]);
  }
  Vertex top_right() {
    return Vertex(X[sorted_X_top[1]], Y[sorted_X_top[1]]);
  }

  Vertex bottom_left() {
    return Vertex(X[sorted_X_bottom[0]], Y[sorted_X_bottom[0]]);
  }
  Vertex bottom_right() {
    return Vertex(X[sorted_X_bottom[1]], Y[sorted_X_bottom[1]]);
  }
};

struct TriangleList {
  int numTriangles;
  Triangle *triangles;
};

} // namespace triangles

////////////////////////////////////////////////////////////////////////////////////////

namespace geometry {

/// Line
//// Initialized with a slope and bias, and optionally a "left" and "right" coordinate to support cases
//// where slope is 0 or infinity.
struct Line {
  double m, b, x_left, x_right;

  Line(): m(0), b(0), x_left(-1), x_right(-1) {}
  Line (double m, double b, double x_left, double x_right): m(m), b(b), x_left(x_left), x_right(x_right) {}

  double intersect(double y) {
    if (m == 0){
      assert(x_left >= 0 && x_right >= 0 && x_left == x_right);
      return x_left;
    }
    return (y - b) / m;
  }

  double leftIntersection(double y) {
    return (m == 0) ? (x_left) : ((y - b) / m);
  }

  double rightIntersection(double y) {
    return (m == 0) ? (x_right) : ((y - b) / m);
  }

  bool valid() {
    if (m == 0 && (x_left < 0 || x_right < 0))
      return false;
    return true;
  }
};

Line intercept(triangles::Vertex a, triangles::Vertex b) {
  if (math::abs_difference(a.x(), b.x()) == 0) {
      // Horizontal line -- to prevent zero division, we just return the leftmost and rightmost X coordinates.
      return Line(0, 0, std::min(a.x(), b.x()), std::max(a.x(), b.x()));
  }
  if (math::abs_difference(a.y(), b.y()) == 0) {
      return Line(); // Vertical lines are considered invalid.
  }
  double m_ = (b.y() - a.y()) / (b.x() - a.x());
  double b_ = b.y() - (m_ * b.x());
  return Line(m_, b_, -1, -1);
}

} // namespace geometry

////////////////////////////////////////////////////////////////////////////////////////

namespace image {

/// Pixel
//// Holds a single RGB value, initialized as black (0, 0, 0).
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

/// Image
//// Image struct containing an array of Pixels and resolution.
struct Image {
  struct Params {
    int x_length, y_length;

    Params(int h, int w): x_length(h), y_length(w) { }

    int numel() {
      return x_length * y_length;
    }

    int stride(int dim) {
      if (dim == 0) {
        return x_length;
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

  Image(int x_length, int y_length): params(Params(x_length, y_length)) {
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
    return safe_coordinate(x, params.x_length);
  }

  int safe_y_coordinate(int y) {
    int safe_y = safe_coordinate(y, params.y_length);
    if (safe_y < 0)
      return -1;
    // TODO: add a "layout" template to support both coordinate formats (even more) at the same time?
    return params.y_length - 1 - safe_y;
    //return safe_coordinate(y, params.y_length);
  }

  int safe_limit(int index, int limit) {
    return std::min(std::max(index, limit), 0);
  }

  int safe_x_limit(int x) {
    return safe_limit(x, params.x_length);
  }

  int safe_y_limit(int y) {
    return safe_limit(y, params.y_length);
  }

  void set_pixel(int y, int x, Pixel v) {
    int x_ = safe_x_coordinate(x);
    int y_ = safe_y_coordinate(y);
    if (x_ < 0 || y_ < 0)
      return;
    _arr[y_ * params.stride(0) + x_].set_value(v);
  }

  Pixel get_pixel(int y, int x) {
    int x_ = safe_x_coordinate(x);
    int y_ = safe_y_coordinate(y);
    if (x_ < 0 || y_ < 0)
      return Pixel();
    return _arr[y_ * params.stride(0) + x_];
  }

  Pixel* get_data() {
    return _arr;
  }

};

} // namespace image

////////////////////////////////////////////////////////////////////////////////////////

namespace algorithms {

void fillTriangle(
    triangles::Triangle &t, 
    image::Image &x, 
    int rowMin, 
    int rowMax, 
    triangles::Vertex anchor, 
    triangles::Vertex left, 
    triangles::Vertex right) {
  geometry::Line leftEdge  = geometry::intercept( left, anchor);
  geometry::Line rightEdge = geometry::intercept(right, anchor);
  if (leftEdge.valid() && rightEdge.valid()) {
    for (int r=rowMin; r <= rowMax; ++r) {
      double leftEndD  =   leftEdge.leftIntersection(r);
      double rightEndD = rightEdge.rightIntersection(r);
      math::swap<double>(&leftEndD, &rightEndD, leftEndD > rightEndD);
      int leftEnd  = math::C441( leftEndD);
      int rightEnd = math::F441(rightEndD);
      for (int c = leftEnd; c <= rightEnd; ++c) {
        x.set_pixel(r, c, image::Pixel(t.color));
      }
    }
  }
}

void fillBottomTriangle(triangles::Triangle &t, image::Image &x) {
  int rowMin = math::C441(t.bottom().y());
  int rowMax = math::F441(t.middle().y());
  triangles::Vertex anchor = t.bottom();
  triangles::Vertex   left = t.bottom_left();
  triangles::Vertex  right = t.bottom_right();
  fillTriangle(t, x, rowMin, rowMax, anchor, left, right);
}

void fillTopTriangle(triangles::Triangle &t, image::Image &x) {
  int rowMin = math::C441(t.middle().y());
  int rowMax = math::F441(t.top().y());
  triangles::Vertex anchor = t.top();
  triangles::Vertex   left = t.top_left();
  triangles::Vertex  right = t.top_right();
  fillTriangle(t, x, rowMin, rowMax, anchor, left, right);
}

void RasterizeGoingUpTriangle(triangles::Triangle &t, image::Image &x) {
  t.precompute_sorts();

  fillBottomTriangle(t, x);
  fillTopTriangle(   t, x);
}

} // namespace algorithms

////////////////////////////////////////////////////////////////////////////////////////

namespace io {
// namespace io handles reading/writing to file.

/// Image2PNM: Dumps an Image instance into a PNM file.
void Image2PNM(image::Image img, string fn) {
  const char* format = "P6";
  const char* maxval = "255";
  int x_length = img.params.x_length;
  int y_length = img.params.y_length;
  int numel = img.params.numel();
  FILE *f = fopen(fn.c_str(), "wb");
  assert(f != NULL);
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n", x_length, y_length);
  fprintf(f, "%d\n", 255);
  fwrite(img.get_data(), x_length * y_length, sizeof(image::Pixel), f);
  fclose(f);
}

} // namespace io

////////////////////////////////////////////////////////////////////////////////////////

namespace skel {
// namespace skel holds all the code I didn't write (usually skeleton/starter code.)

triangles::TriangleList GetTriangles(int small_read) {
  using TriangleList = typename triangles::TriangleList;
  using Triangle = typename triangles::Triangle;
  FILE *f = fopen("tris.txt", "r");
  if (f == NULL)
  {
      fprintf(stderr, "You must place the tris.txt file in the current directory.\n");
      exit(EXIT_FAILURE);
  }
  fseek(f, 0, SEEK_END);
  int numBytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (numBytes != 241511792)
  {
      fprintf(stderr, "Your tris.txt file is corrupted.  It should be 241511792 bytes, but you only have %d.\n", numBytes);
      exit(EXIT_FAILURE);
  }

  if (small_read == 1)
  {
      numBytes = 10000;
  }

  char *buffer = (char *) malloc(numBytes);
  if (buffer == NULL)
  {
      fprintf(stderr, "Unable to allocate enough memory to load file.\n");
      exit(EXIT_FAILURE);
  }
  
  fread(buffer, sizeof(char), numBytes, f);

  char *tmp = buffer;
  int numTriangles = atoi(tmp);
  while (*tmp != '\n')
      tmp++;
  tmp++;
 
  if (numTriangles != 2566541)
  {
      fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
      exit(EXIT_FAILURE);
  }

  if (small_read == 1)
      numTriangles = 100;

  TriangleList tl;
  tl.numTriangles = numTriangles;
  tl.triangles = (Triangle *) malloc(sizeof(Triangle)*tl.numTriangles);

  for (int i = 0 ; i < tl.numTriangles ; i++)
  {
    double x1, y1, x2, y2, x3, y3;
    int    r, g, b;
    /*
     * Weird: sscanf has a terrible implementation for large strings.
     * When I did the code below, it did not finish after 45 minutes.
     * Reading up on the topic, it sounds like it is a known issue that
     * sscanf fails here.  Stunningly, fscanf would have been faster.
     *     sscanf(tmp, "(%lf, %lf), (%lf, %lf), (%lf, %lf) = (%d, %d, %d)\n%n",
     *              &x1, &y1, &x2, &y2, &x3, &y3, &r, &g, &b, &numRead);
     *
     *  So, instead, do it all with atof/atoi and advancing through the buffer manually...
     */
     tmp++,
     x1 = atof(tmp);
     while (*tmp != ',')
        tmp++;
     tmp += 2; // comma+space
     y1 = atof(tmp);
     while (*tmp != ')')
        tmp++;
     tmp += 4; // right-paren+comma+space+left-paren
     x2 = atof(tmp);
     while (*tmp != ',')
        tmp++;
     tmp += 2; // comma+space
     y2 = atof(tmp);
     while (*tmp != ')')
        tmp++;
     tmp += 4; // right-paren+comma+space+left-paren
     x3 = atof(tmp);
     while (*tmp != ',')
        tmp++;
     tmp += 2; // comma+space
     y3 = atof(tmp);
     while (*tmp != ')')
        tmp++;
     tmp += 5; // right-paren+space+equal+space+left-paren
     r = atoi(tmp);
     while (*tmp != ',')
        tmp++;
     tmp += 2; // comma+space
     g = atoi(tmp);
     while (*tmp != ',')
        tmp++;
     tmp += 2; // comma+space
     b = atoi(tmp);
     while (*tmp != '\n')
        tmp++;
     tmp++; // onto next line
     
     tl.triangles[i].X[0] = x1;
     tl.triangles[i].X[1] = x2;
     tl.triangles[i].X[2] = x3;
     tl.triangles[i].Y[0] = y1;
     tl.triangles[i].Y[1] = y2;
     tl.triangles[i].Y[2] = y3;
     tl.triangles[i].color[0] = r;
     tl.triangles[i].color[1] = g;
     tl.triangles[i].color[2] = b;
     //printf("Read triangle %f, %f, %f, %f, %f, %f, %d, %d, %d\n", x1, y1, x2, y2, x3, y3, r, g, b);
  }

  free(buffer);
  return tl;
}

} // namespace skel

////////////////////////////////////////////////////////////////////////////////////////

} // namespace project

using namespace project;
using Image = typename image::Image;
using TriangleList = typename triangles::TriangleList;

int main() {
    cout << "Generating image" << endl;

    Image x = Image(1786, 1344);
    TriangleList list = skel::GetTriangles(0);

    for (int i=0; i < list.numTriangles; ++i)
      algorithms::RasterizeGoingUpTriangle(list.triangles[i], x);

    cout << "Saving image" << endl;

    io::Image2PNM(x, "proj1D_out.pnm");

    return 0;
}
