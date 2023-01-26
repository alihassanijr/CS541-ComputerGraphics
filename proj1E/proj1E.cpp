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
#include <limits>

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
  value_t t = value_t(c - a) / value_t(b - a);
  return f_a + (t * (f_b - f_a));
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

struct Color {
  double colors[3];

  Color() { }

  Color(double r, double g, double b) {
    //colors = new double[3];
    colors[0] = r;
    colors[1] = g;
    colors[2] = b;
  }

  Color(double c) {
    //colors = new double[3];
    for (int i=0; i < 3; ++i)
      colors[i] = c;
  }

  Color(double * rgb) {
    for (int i=0; i < 3; ++i)
      colors[i] = rgb[i];
  }

  bool valid() const {
    return colors != nullptr;
  }

  Color operator*(const double &b) {
    assert(this->valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] * b;
    return c;
  }

  Color operator/(const double &b) {
    assert(this->valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] / b;
    return c;
  }

  Color operator+(const double &b) {
    assert(this->valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] + b;
    return c;
  }

  Color operator-(const double &b) {
    assert(this->valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] - b;
    return c;
  }

  Color operator*(const Color &b) {
    assert(this->valid());
    assert(b.valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] * b.colors[i];
    return c;
  }

  Color operator/(const Color &b) {
    assert(this->valid());
    assert(b.valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] / b.colors[i];
    return c;
  }

  Color operator+(const Color &b) {
    assert(this->valid());
    assert(b.valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] + b.colors[i];
    return c;
  }

  Color operator-(const Color &b) {
    assert(this->valid());
    assert(b.valid());
    Color c/*(new double[3])*/;
    for (int i=0; i < 3; ++i)
      c.colors[i] = this->colors[i] - b.colors[i];
    return c;
  }
};

/// Vertex
//// I like to define a Vertex struct because it makes it so much easier to pass an array of pointers
//// when we're pre-sorting all vertices.
struct Vertex {
  double* X;
  double* Y;
  double* Z;

  Vertex(double &x, double &y, double &z) {
    X = &x;
    Y = &y;
    Z = &z;
  }

  double x() const {
    return X[0];
  }
  double y() const {
    return Y[0];
  }
  double z() const {
    return Z[0];
  }
};

/// Triangle
//// As defined in the starter code, with the pre-sort logic added.
//// The pre-sorter is just a method that gets called once before rasterization.
struct Triangle {
  double         X[3];
  double         Y[3];
  double         Z[3];
  double         color[3][3];

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
    return Vertex(X[sorted_Y[2]], Y[sorted_Y[2]], Z[sorted_Y[2]]);
  }                                             
  Vertex middle() {                             
    return Vertex(X[sorted_Y[1]], Y[sorted_Y[1]], Z[sorted_Y[1]]);
  }                                             
  Vertex bottom() {                             
    return Vertex(X[sorted_Y[0]], Y[sorted_Y[0]], Z[sorted_Y[0]]);
  }

  Vertex top_left() {
    return Vertex(X[sorted_X_top[0]], Y[sorted_X_top[0]], Z[sorted_X_top[0]]);
  }                                                     
  Vertex top_right() {                                  
    return Vertex(X[sorted_X_top[1]], Y[sorted_X_top[1]], Z[sorted_X_top[1]]);
  }

  Vertex bottom_left() {
    return Vertex(X[sorted_X_bottom[0]], Y[sorted_X_bottom[0]], Z[sorted_X_bottom[0]]);
  }                                                           
  Vertex bottom_right() {                                     
    return Vertex(X[sorted_X_bottom[1]], Y[sorted_X_bottom[1]], Z[sorted_X_bottom[1]]);
  }

  Color top_color() {
    return Color(color[sorted_Y[2]]);
  }                            
  Color middle_color() {             
    return Color(color[sorted_Y[1]]);
  }                            
  Color bottom_color() {             
    return Color(color[sorted_Y[0]]);
  }

  Color top_left_color() {
    return Color(color[sorted_X_top[0]]);
  }                                
  Color top_right_color() {              
    return Color(color[sorted_X_top[1]]);
  }

  Color bottom_left_color() {
    return Color(color[sorted_X_bottom[0]]);
  }                                   
  Color bottom_right_color() {              
    return Color(color[sorted_X_bottom[1]]);
  }
};

struct TriangleList {
  int numTriangles;
  Triangle *triangles;
};

} // namespace triangles

////////////////////////////////////////////////////////////////////////////////////////

namespace geometry {

struct Coord {
  double* X;
  double* Y;

  Coord(double x, double y) {
    X = new double[1];
    Y = new double[1];
    X[0] = x;
    Y[0] = y;
  }

  Coord(double &x, double &y) {
    X = &x;
    Y = &y;
  }

  Coord(double *x, double *y) {
    X = x;
    Y = y;
  }

  Coord(triangles::Vertex v) {
    X = v.X;
    Y = v.Y;
  }

  double x() const {
    return X[0];
  }
  double y() const {
    return Y[0];
  }

  Coord operator*(const Coord &b) {
    Coord c(new double[1], new double[1]);
    c.X[0] = this->x() * b.x();
    c.Y[0] = this->y() * b.y();
    return c;
  }

  Coord operator/(const Coord &b) {
    Coord c(new double[1], new double[1]);
    c.X[0] = this->x() / b.x();
    c.Y[0] = this->y() / b.y();
    return c;
  }

  Coord operator+(const Coord &b) {
    Coord c(new double[1], new double[1]);
    c.X[0] = this->x() + b.x();
    c.Y[0] = this->y() + b.y();
    return c;
  }

  Coord operator-(const Coord &b) {
    Coord c(new double[1], new double[1]);
    c.X[0] = this->x() - b.x();
    c.Y[0] = this->y() - b.y();
    return c;
  }

  operator double() const {
    return std::sqrt(std::pow(x(), 2) + std::pow(y(), 2));
  }

  operator triangles::Color() const {
    double c = std::sqrt(std::pow(x(), 2) + std::pow(y(), 2));
    return triangles::Color(c, c, c);
  }
};

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

unsigned char pixeldouble2char(double c) {
  return math::C441(c * 255);
}

/// Pixel
//// Holds a single RGB value, initialized as black (0, 0, 0).
struct Pixel {
  unsigned char r, g, b;

  Pixel(): r(0), g(0), b(0) {}

  Pixel(unsigned char r, unsigned char g, unsigned char b): r(r), g(g), b(b) {}

  Pixel(double r, double g, double b): r(pixeldouble2char(r)), g(pixeldouble2char(g)), b(pixeldouble2char(b)) {}

  Pixel(unsigned char *color): r(color[0]), g(color[1]), b(color[2]) {}

  Pixel(double *color): r(pixeldouble2char(color[0])), g(pixeldouble2char(color[1])), b(pixeldouble2char(color[2])) {}

  void set_value(unsigned char r_, unsigned char g_, unsigned char b_) {
    r = r_;
    g = g_;
    b = b_;
  }

  void set_value(double r_, double g_, double b_) {
    r = pixeldouble2char(r_);
    g = pixeldouble2char(g_);
    b = pixeldouble2char(b_);
  }

  void set_value(Pixel p) {
    set_value(p.r, p.g, p.b);
  }

  void zfill() {
    set_value(0.0d, 0.0d, 0.0d);
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
  double *z_buffer;

public:
  Params params;

  Image(): params(Params(0, 0)) {
    _arr = nullptr;
    z_buffer = nullptr;
  }

  Image(int x_length, int y_length): params(Params(x_length, y_length)) {
    _arr = new Pixel[params.numel()];
    z_buffer = new double[params.numel()];
    zfill();
  }

  void zfill() {
    for (int i=0; i < params.numel(); ++i) {
      _arr[i].zfill();
      z_buffer[i] = std::numeric_limits<double>::lowest();
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
    set_pixel(y, x, 0, v);
  }

  void set_pixel(int y, int x, double z, Pixel v) {
    int x_ = safe_x_coordinate(x);
    int y_ = safe_y_coordinate(y);
    if (x_ < 0 || y_ < 0)
      return;
    int linearIndex = y_ * params.stride(0) + x_;
    if (z < z_buffer[linearIndex])
      return;
    z_buffer[linearIndex] = z;
    _arr[linearIndex].set_value(v);
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
    double rowMin, 
    double rowMax, 
    triangles::Vertex anchor, 
    triangles::Vertex left, 
    triangles::Vertex right, 
    triangles::Color anchorColor, 
    triangles::Color leftColor, 
    triangles::Color rightColor) {
  geometry::Line leftEdge  = geometry::intercept( left, anchor);
  geometry::Line rightEdge = geometry::intercept(right, anchor);
  if (leftEdge.valid() && rightEdge.valid()) {
    for (int r=math::C441(rowMin); r <= math::F441(rowMax); ++r) {
      double leftEnd  =   leftEdge.leftIntersection(r);
      double rightEnd = rightEdge.rightIntersection(r);
      double leftZ = math::lerp<geometry::Coord, double>(
          geometry::Coord(left), 
          geometry::Coord(anchor), 
          left.z(), 
          anchor.z(), 
          geometry::Coord(leftEnd, r));
      double rightZ = math::lerp<geometry::Coord, double>(
          geometry::Coord(right), 
          geometry::Coord(anchor), 
          right.z(), 
          anchor.z(), 
          geometry::Coord(rightEnd, r));
      triangles::Color leftColorX = math::lerp<geometry::Coord, triangles::Color>(
          geometry::Coord(left), 
          geometry::Coord(anchor), 
          leftColor, 
          anchorColor, 
          geometry::Coord(leftEnd, r));
      triangles::Color rightColorX = math::lerp<geometry::Coord, triangles::Color>(
          geometry::Coord(right), 
          geometry::Coord(anchor), 
          rightColor, 
          anchorColor, 
          geometry::Coord(rightEnd, r));
      if (leftEnd >= rightEnd) {
        math::swap<double>(&leftZ, &rightZ);
        math::swap<triangles::Color>(&leftColorX, &rightColorX);
        math::swap<double>(&leftEnd, &rightEnd);
      }
      for (int c = math::C441(leftEnd); c <= math::F441(rightEnd); ++c) {
        double z = math::lerp<double, double>(
             leftEnd, 
            rightEnd, 
            leftZ, 
            rightZ, 
            c);
        triangles::Color color = math::lerp<double, triangles::Color>(
             leftEnd, 
            rightEnd, 
            leftColorX, 
            rightColorX, 
            c);
        x.set_pixel(r, c, z, image::Pixel(color.colors[0], color.colors[1], color.colors[2]));
        //x.set_pixel(r, c, z, image::Pixel(t.color[0]));
      }
    }
  }
}

void fillBottomTriangle(triangles::Triangle &t, image::Image &x) {
  double rowMinD = t.bottom().y();
  double rowMaxD = t.middle().y();
  triangles::Vertex anchor = t.bottom();
  triangles::Vertex   left = t.bottom_left();
  triangles::Vertex  right = t.bottom_right();
  triangles::Color anchorColor =       t.bottom_color();
  triangles::Color   leftColor =  t.bottom_left_color();
  triangles::Color  rightColor = t.bottom_right_color();
  fillTriangle(t, x, rowMinD, rowMaxD, anchor, left, right, anchorColor, leftColor, rightColor);
}

void fillTopTriangle(triangles::Triangle &t, image::Image &x) {
  double rowMinD = t.middle().y();
  double rowMaxD = t.top().y();
  triangles::Vertex anchor = t.top();
  triangles::Vertex   left = t.top_left();
  triangles::Vertex  right = t.top_right();
  triangles::Color anchorColor =       t.top_color();
  triangles::Color   leftColor =  t.top_left_color();
  triangles::Color  rightColor = t.top_right_color();
  fillTriangle(t, x, rowMinD, rowMaxD, anchor, left, right, anchorColor, leftColor, rightColor);
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

char* ReadTuple3(char *tmp, double *v1, double *v2, double *v3) {
    tmp++; /* left paren */
    *v1 = atof(tmp);
    while (*tmp != ',')
       tmp++;
    tmp += 2; // comma+space
    *v2 = atof(tmp);
    while (*tmp != ',')
       tmp++;
    tmp += 2; // comma+space
    *v3 = atof(tmp);
    while (*tmp != ')')
       tmp++;
    tmp++; /* right paren */
    return tmp;
}

triangles::TriangleList Get3DTriangles()
{
   FILE *f = fopen("tris_w_r_rgb.txt", "r");
   if (f == NULL)
   {
       fprintf(stderr, "You must place the tris_w_r_rgb.txt file in the current directory.\n");
       exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   int numBytes = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (numBytes != 13488634)
   {
       fprintf(stderr, "Your tris_w_r_rgb.txt file is corrupted.  It should be 13488634 bytes, but you have %d.\n", numBytes);
       exit(EXIT_FAILURE);
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
 
   if (numTriangles != 42281)
   {
       fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
       exit(EXIT_FAILURE);
   }

   triangles::TriangleList tl;
   tl.numTriangles = numTriangles;
   tl.triangles = (triangles::Triangle *) malloc(sizeof(triangles::Triangle)*tl.numTriangles);

   for (int i = 0 ; i < tl.numTriangles ; i++)
   {
       double x1, y1, z1, x2, y2, z2, x3, y3, z3;
       double r[3], g[3], b[3];
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
       tmp = ReadTuple3(tmp, &x1, &y1, &z1);
       tmp += 3; /* space+equal+space */
       tmp = ReadTuple3(tmp, r+0, g+0, b+0);
       tmp += 2; /* comma+space */
       tmp = ReadTuple3(tmp, &x2, &y2, &z2);
       tmp += 3; /* space+equal+space */
       tmp = ReadTuple3(tmp, r+1, g+1, b+1);
       tmp += 2; /* comma+space */
       tmp = ReadTuple3(tmp, &x3, &y3, &z3);
       tmp += 3; /* space+equal+space */
       tmp = ReadTuple3(tmp, r+2, g+2, b+2);
       tmp++;    /* newline */

       tl.triangles[i].X[0] = x1;
       tl.triangles[i].X[1] = x2;
       tl.triangles[i].X[2] = x3;
       tl.triangles[i].Y[0] = y1;
       tl.triangles[i].Y[1] = y2;
       tl.triangles[i].Y[2] = y3;
       tl.triangles[i].Z[0] = z1;
       tl.triangles[i].Z[1] = z2;
       tl.triangles[i].Z[2] = z3;
       tl.triangles[i].color[0][0] = r[0];
       tl.triangles[i].color[0][1] = g[0];
       tl.triangles[i].color[0][2] = b[0];
       tl.triangles[i].color[1][0] = r[1];
       tl.triangles[i].color[1][1] = g[1];
       tl.triangles[i].color[1][2] = b[1];
       tl.triangles[i].color[2][0] = r[2];
       tl.triangles[i].color[2][1] = g[2];
       tl.triangles[i].color[2][2] = b[2];
       //printf("Read triangle (%f, %f, %f) / (%f, %f, %f), (%f, %f, %f) / (%f, %f, %f), (%f, %f, %f) / (%f, %f, %f)\n", x1, y1, z1, r[0], g[0], b[0], x2, y2, z2, r[1], g[1], b[1], x3, y3, z3, r[2], g[2], b[2]);
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

    Image x = Image(1000, 1000);
    TriangleList list = skel::Get3DTriangles();

    for (int i=0; i < list.numTriangles; ++i)
      algorithms::RasterizeGoingUpTriangle(list.triangles[i], x);

    cout << "Saving image" << endl;

    io::Image2PNM(x, "proj1D_out.pnm");

    return 0;
}
