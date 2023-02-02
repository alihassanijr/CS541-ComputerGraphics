/*

Ali Hassani

Project 1-F

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

// Cotangent
template <typename T>
T cot(T v) {
  return T(1.0) / tan(v);
}

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

namespace linalg {
// namespace linalg holds all linear algebra routines and sturcts.

template <typename T, int M_>
struct Vector {
  T data[M_];
  static const int Size = M_;
  static const int M = M_;

  Vector() {}

  Vector(T* _data) {
    for (int i=0; i < Size; ++i)
      data[i] = _data[i];
  }

  Vector operator*(const T b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] * b;
    return c;
  }
  Vector operator/(const T b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] / b;
    return c;
  }
  Vector operator+(const T b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] + b;
    return c;
  }
  Vector operator-(const T b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] - b;
    return c;
  }

  Vector operator*(const Vector &b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] * b.data[i];
    return c;
  }
  Vector operator/(const Vector &b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] / b.data[i];
    return c;
  }
  Vector operator+(const Vector &b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] + b.data[i];
    return c;
  }
  Vector operator-(const Vector &b) {
    Vector c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] - b.data[i];
    return c;
  }

  T get(int index) {
    assert(index >= 0 && index < Size);
    return data[index];
  }
  void set(int index, T value) {
    assert(index >= 0 && index < Size);
    data[index] = value;
  }

  T norm() {
    T norm = T(0.0);
    for (int i=0; i < Size; ++i)
      norm += pow(data[i], 2);
    return sqrt(norm);
  }

  T sum() {
    T acc = T(0.0);
    for (int i=0; i < Size; ++i) {
      acc += data[i];
    }
    return acc;
  }
};

template <typename T, int M_, int N_>
struct Matrix {
  T data[M_*N_];
  static const int Size = M_ * N_;
  static const int M = M_;
  static const int N = N_;

  Matrix() {}

  Matrix(T* _data) {
    for (int i=0; i < Size; ++i)
      data[i] = _data[i];
  }
  Matrix(T fill_val) {
    for (int i=0; i < Size; ++i)
      data[i] = fill_val;
  }

  Matrix operator*(const T b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] * b;
    return c;
  }
  Matrix operator/(const T b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] / b;
    return c;
  }
  Matrix operator+(const T b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] + b;
    return c;
  }
  Matrix operator-(const T b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] - b;
    return c;
  }

  Matrix operator*(const Matrix &b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] * b.data[i];
    return c;
  }
  Matrix operator/(const Matrix &b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] / b.data[i];
    return c;
  }
  Matrix operator+(const Matrix &b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] + b.data[i];
    return c;
  }
  Matrix operator-(const Matrix &b) {
    Matrix c;
    for (int i=0; i < Size; ++i)
      c.data[i] = this->data[i] - b.data[i];
    return c;
  }

  T get(int index_i, int index_j) {
    assert(index_i >= 0 && index_i < M && index_j >= 0 && index_j < N);
    return data[index_i * N + index_j];
  }
  void set(int index_i, int index_j, T value) {
    assert(index_i >= 0 && index_i < M && index_j >= 0 && index_j < N);
    data[index_i * N + index_j] = value;
  }

  operator Vector<T, N>() const {
    static_assert(M == 1);
    Vector<T, N> c(this->data);
  }
};

//template <typename T, int M_, int N_>
//Vector<T, M_> matmul(Vector<T, M_> a, Matrix<T, M_, N_> b) {
//  Vector<T, N_> c;
//  for (int n=0; n < N_; ++n) {
//    T val = T(0.0);
//    for (int k=0; k < M_; ++k) {
//      val += a.get(k) * b.get(k, n);
//    }
//    c.set(n, val);
//  }
//  return c;
//}

template <typename T, int M_, int N_, int K_>
Matrix<T, M_, N_> matmul(Matrix<T, M_, K_> a, Matrix<T, K_, N_> b) {
  Matrix<T, M_, N_> c(T(0.0));
  for (int k=0; k < K_; ++k) {
    for (int m=0; m < M_; ++m) {
      for (int n=0; n < N_; ++n) {
        T val = c.get(m, n);
        val += a.get(m, k) * b.get(k, n);
        c.set(m, n, val);
      }
    }
  }
  return c;
}

template <typename T>
Vector<T, 3> cross_prod(Vector<T, 3> a, Vector<T, 3> b) {
  Vector<T, 3> c;
  c.set(0, a.get(1) * b.get(2) - a.get(2) * b.get(1));
  c.set(1, a.get(2) * b.get(0) - a.get(0) * b.get(2)); // -1 already applied
  c.set(2, a.get(0) * b.get(1) - a.get(1) * b.get(0));
  return c;
}

} // namespace linalg

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

  operator linalg::Vector<double, 3>() const {
    linalg::Vector<double, 3> v;
    v.set(0, X[0]);
    v.set(1, Y[0]);
    v.set(2, Z[0]);
    return v;
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
  double         normals[3][3];

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

struct Coord2D {
  double* X;
  double* Y;

  Coord2D(double x, double y) {
    X = new double[1];
    Y = new double[1];
    X[0] = x;
    Y[0] = y;
  }

  Coord2D(double &x, double &y) {
    X = &x;
    Y = &y;
  }

  Coord2D(double *x, double *y) {
    X = x;
    Y = y;
  }

  Coord2D(triangles::Vertex v) {
    X = v.X;
    Y = v.Y;
  }

  double x() const {
    return X[0];
  }
  double y() const {
    return Y[0];
  }

  Coord2D operator*(const Coord2D &b) {
    Coord2D c(new double[1], new double[1]);
    c.X[0] = this->x() * b.x();
    c.Y[0] = this->y() * b.y();
    return c;
  }

  Coord2D operator/(const Coord2D &b) {
    Coord2D c(new double[1], new double[1]);
    c.X[0] = this->x() / b.x();
    c.Y[0] = this->y() / b.y();
    return c;
  }

  Coord2D operator+(const Coord2D &b) {
    Coord2D c(new double[1], new double[1]);
    c.X[0] = this->x() + b.x();
    c.Y[0] = this->y() + b.y();
    return c;
  }

  Coord2D operator-(const Coord2D &b) {
    Coord2D c(new double[1], new double[1]);
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

  operator linalg::Vector<double, 2>() const {
    linalg::Vector<double, 2> v;
    v.set(0, X[0]);
    v.set(1, Y[0]);
    return v;
  }
};

struct Coord3D {
  linalg::Vector<double, 3> vec;

  Coord3D(double x, double y, double z) {
    vec.data[0] = x;
    vec.data[1] = y;
    vec.data[2] = z;
  }

  Coord3D(double *x, double *y, double *z) {
    vec.data[0] = x[0];
    vec.data[1] = y[0];
    vec.data[2] = z[0];
  }

  Coord3D(triangles::Vertex v) {
    vec.data[0] = v.X[0];
    vec.data[1] = v.Y[0];
    vec.data[2] = v.Z[0];
  }

  Coord3D(linalg::Vector<double, 3> v) {
    vec.data[0] = v.data[0];
    vec.data[1] = v.data[1];
    vec.data[2] = v.data[2];
  }

  double x() const {
    return vec.data[0];
  }
  double y() const {
    return vec.data[1];
  }
  double z() const {
    return vec.data[2];
  }

  Coord3D operator*(const Coord3D &b) {
    Coord3D c(new double[1], new double[1], new double[1]);
    c.vec.data[0] = this->x() * b.x();
    c.vec.data[1] = this->y() * b.y();
    c.vec.data[2] = this->z() * b.z();
    return c;
  }

  Coord3D operator/(const Coord3D &b) {
    Coord3D c(new double[1], new double[1], new double[1]);
    c.vec.data[0] = this->x() / b.x();
    c.vec.data[1] = this->y() / b.y();
    c.vec.data[2] = this->z() / b.z();
    return c;
  }

  Coord3D operator+(const Coord3D &b) {
    Coord3D c(new double[1], new double[1], new double[1]);
    c.vec.data[0] = this->x() + b.x();
    c.vec.data[1] = this->y() + b.y();
    c.vec.data[2] = this->z() + b.z();
    return c;
  }

  Coord3D operator-(const Coord3D &b) {
    Coord3D c(new double[1], new double[1], new double[1]);
    c.vec.data[0] = this->x() - b.x();
    c.vec.data[1] = this->y() - b.y();
    c.vec.data[2] = this->z() - b.z();
    return c;
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

namespace views {
// namespace views

struct Camera {
  using Coord = typename geometry::Coord3D;

  double near, far;
  double angle;
  Coord position;
  Coord focus;
  Coord up;

  Camera (Coord position, Coord focus, Coord up, double angle, double near, double far) :
    position(position),
    focus(focus),
    up(up),
    angle(angle),
    near(near),
    far(far) {}
};

} // namespace views

////////////////////////////////////////////////////////////////////////////////////////

namespace transforms {

using Transform = typename linalg::Matrix<double, 4, 4>;

Transform zeros() {
  return Transform(0.0d);
}

Transform identity() {
  Transform transform = zeros();
  transform.set(0, 0, 1.0d);
  transform.set(1, 1, 1.0d);
  transform.set(2, 2, 1.0d);
  transform.set(3, 3, 1.0d);
  return transform;
}

Transform compose(vector<Transform> transforms) {
  Transform composed = identity();
  
  for (Transform t : transforms) {
    composed = matmul(composed, t);
  }

  return composed;
}

// image_to_device: device transform
Transform image_to_device(double height, double width) {
  Transform transform = zeros();

  //double scale = min(height, width) / 2;
  double scale_h = height / 2;
  double scale_w = width / 2;
  transform.set(0, 0, scale_w);
  transform.set(1, 1, scale_h);
  transform.set(2, 2, 1.0d);
  transform.set(3, 0, scale_w);
  transform.set(3, 1, scale_h);
  transform.set(3, 3, 1.0d);

  return transform;
}

// world_to_camera: camera transform
Transform world_to_camera(views::Camera camera) {
  Transform transform = zeros();

  linalg::Vector w_ = (camera.position - camera.focus).vec;
  w_ = (w_) / (w_.norm());
  linalg::Vector u_ = linalg::cross_prod<double>(camera.up.vec, w_);
  u_ = (u_) / (u_.norm());
  linalg::Vector v_ = linalg::cross_prod<double>(w_, u_);
  v_ = (v_) / (v_.norm());
  linalg::Vector t_ = (geometry::Coord3D(0.0d, 0.0d, 0.0d) - camera.position).vec;

  geometry::Coord3D w = geometry::Coord3D(w_);
  geometry::Coord3D u = geometry::Coord3D(u_);
  geometry::Coord3D v = geometry::Coord3D(v_);
  geometry::Coord3D t = geometry::Coord3D(t_);

  transform.set(0, 0, u.x());
  transform.set(0, 1, v.x());
  transform.set(0, 2, w.x());
  transform.set(1, 0, u.y());
  transform.set(1, 1, v.y());
  transform.set(1, 2, w.y());
  transform.set(2, 0, u.z());
  transform.set(2, 1, v.z());
  transform.set(2, 2, w.z());

  transform.set(3, 0, (u.vec * t.vec).sum());
  transform.set(3, 1, (v.vec * t.vec).sum());
  transform.set(3, 2, (w.vec * t.vec).sum());
  transform.set(3, 3, 1.0d);

  return transform;
}

// camera_to_image: view transform
Transform camera_to_image(views::Camera camera) {
  Transform transform = zeros();

  double cot_alpha_div_2 = math::cot(camera.angle / 2);
  double n = camera.near;
  double f = camera.far;

  transform.set(0, 0, cot_alpha_div_2);
  transform.set(1, 1, cot_alpha_div_2);
  transform.set(2, 2, (f + n) / (f - n));
  transform.set(3, 2, (2 * f * n) / (f - n));

  transform.set(2, 3, -1.0d);

  return transform;
}

Transform world_to_device(views::Camera camera, double height, double width) {
  Transform w2c = world_to_camera(camera);
  Transform c2i = camera_to_image(camera);
  Transform i2d = image_to_device(height, width);

  return compose({w2c, c2i, i2d});
}

triangles::Triangle transform_triangle(Transform transform, triangles::Triangle triangle) {
  linalg::Matrix<double, 3, 4> trig(0.0d);
  for (int i=0; i < 3; ++i) {
    trig.set(i, 0, triangle.X[i]);
    trig.set(i, 1, triangle.Y[i]);
    trig.set(i, 2, triangle.Z[i]);
    trig.set(i, 3, 1.0d);
  }
  trig = linalg::matmul(trig, transform);
  for (int i=0; i < 3; ++i) {
    double w = trig.get(i, 3);
    triangle.X[i] = trig.get(i, 0) / w;
    triangle.Y[i] = trig.get(i, 1) / w;
    triangle.Z[i] = trig.get(i, 2) / w;
  }
  return triangle;
}

} // namespace transforms

////////////////////////////////////////////////////////////////////////////////////////

namespace lighting {

struct LightingParameters {
  linalg::Vector<double, 3> lightDir; // The direction of the light source
  double Ka;           // The coefficient for ambient lighting.
  double Kd;           // The coefficient for diffuse lighting.
  double Ks;           // The coefficient for specular lighting.
  double alpha;        // The exponent term for specular lighting.
};

LightingParameters GetLighting(views::Camera c) {
  LightingParameters lp;
  lp.Ka = 0.3;
  lp.Kd = 0.7;
  lp.Ks = 2.8;
  lp.alpha = 50.5;
  lp.lightDir = (c.position - c.focus).vec;
  double mag = lp.lightDir.norm();
  if (mag > 0) {
      lp.lightDir = lp.lightDir / mag;
  }
  return lp;
}


} // namespace lighting

////////////////////////////////////////////////////////////////////////////////////////

namespace algorithms {

void fillTriangle(
    triangles::Triangle t, 
    image::Image image, 
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
      double leftZ = math::lerp<geometry::Coord2D, double>(
          geometry::Coord2D(left), 
          geometry::Coord2D(anchor), 
          left.z(), 
          anchor.z(), 
          geometry::Coord2D(leftEnd, r));
      double rightZ = math::lerp<geometry::Coord2D, double>(
          geometry::Coord2D(right), 
          geometry::Coord2D(anchor), 
          right.z(), 
          anchor.z(), 
          geometry::Coord2D(rightEnd, r));
      triangles::Color leftColorX = math::lerp<geometry::Coord2D, triangles::Color>(
          geometry::Coord2D(left), 
          geometry::Coord2D(anchor), 
          leftColor, 
          anchorColor, 
          geometry::Coord2D(leftEnd, r));
      triangles::Color rightColorX = math::lerp<geometry::Coord2D, triangles::Color>(
          geometry::Coord2D(right), 
          geometry::Coord2D(anchor), 
          rightColor, 
          anchorColor, 
          geometry::Coord2D(rightEnd, r));
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
        image.set_pixel(r, c, z, image::Pixel(color.colors[0], color.colors[1], color.colors[2]));
      }
    }
  }
}

void fillBottomTriangle(triangles::Triangle t, image::Image image) {
  double rowMinD = t.bottom().y();
  double rowMaxD = t.middle().y();
  triangles::Vertex anchor = t.bottom();
  triangles::Vertex   left = t.bottom_left();
  triangles::Vertex  right = t.bottom_right();
  triangles::Color anchorColor =       t.bottom_color();
  triangles::Color   leftColor =  t.bottom_left_color();
  triangles::Color  rightColor = t.bottom_right_color();
  fillTriangle(t, image, rowMinD, rowMaxD, anchor, left, right, anchorColor, leftColor, rightColor);
}

void fillTopTriangle(triangles::Triangle &t, image::Image image) {
  double rowMinD = t.middle().y();
  double rowMaxD = t.top().y();
  triangles::Vertex anchor = t.top();
  triangles::Vertex   left = t.top_left();
  triangles::Vertex  right = t.top_right();
  triangles::Color anchorColor =       t.top_color();
  triangles::Color   leftColor =  t.top_left_color();
  triangles::Color  rightColor = t.top_right_color();
  fillTriangle(t, image, rowMinD, rowMaxD, anchor, left, right, anchorColor, leftColor, rightColor);
}

void RasterizeGoingUpTriangle(triangles::Triangle t, image::Image image) {
  t.precompute_sorts();

  fillBottomTriangle(t, image);
  fillTopTriangle(   t, image);
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

char *
Read3Numbers(char *tmp, double *v1, double *v2, double *v3)
{
    *v1 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v2 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v3 = atof(tmp);
    while (*tmp != ' ' && *tmp != '\n')
       tmp++;
    return tmp;
}

triangles::TriangleList Get3DTriangles() {
   FILE *f = fopen("ws_tris.txt", "r");
   if (f == NULL)
   {
       fprintf(stderr, "You must place the ws_tris.txt file in the current directory.\n");
       exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   int numBytes = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (numBytes != 3892295)
   {
       fprintf(stderr, "Your ws_tris.txt file is corrupted.  It should be 3892295 bytes, but you have %d.\n", numBytes);
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
 
   if (numTriangles != 14702)
   {
       fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
       exit(EXIT_FAILURE);
   }

   triangles::TriangleList tl;
   tl.numTriangles = numTriangles;
   tl.triangles = (triangles::Triangle *) malloc(sizeof(triangles::Triangle)*tl.numTriangles);

   for (int i = 0 ; i < tl.numTriangles ; i++)
   {
       for (int j = 0 ; j < 3 ; j++)
       {
           double x, y, z;
           double r, g, b;
           double normals[3];
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
           tmp = Read3Numbers(tmp, &x, &y, &z);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, &r, &g, &b);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, normals+0, normals+1, normals+2);
           tmp++;    /* newline */

           tl.triangles[i].X[j] = x;
           tl.triangles[i].Y[j] = y;
           tl.triangles[i].Z[j] = z;
           tl.triangles[i].color[j][0] = r;
           tl.triangles[i].color[j][1] = g;
           tl.triangles[i].color[j][2] = b;
           tl.triangles[i].normals[j][0] = normals[0];
           tl.triangles[i].normals[j][1] = normals[1];
           tl.triangles[i].normals[j][2] = normals[2];
       }
   }

   free(buffer);
   return tl;
}

double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

views::Camera GetCamera(int frame, int nframes) {
    using Coord = typename geometry::Coord3D;
    using Camera = typename views::Camera;
    double t = SineParameterize(frame, nframes, nframes/10);
    Coord position(40.0d*sin(2*M_PI*t), 40.0d*cos(2*M_PI*t), 40.0d);
    Coord focus(0.0d, 0.0d, 0.0d);
    Coord up(0.0d, 1.0d, 0.0d);
    double near = 5.0d;
    double far = 200.0d;
    double angle = M_PI/6;
    return Camera(position, focus, up, angle, near, far);
}

} // namespace skel

////////////////////////////////////////////////////////////////////////////////////////

} // namespace project

string gen_filename(int f) {
  char str[256];
  sprintf(str, "proj1F_frame%04d.pnm", f);
  return str;
}


using namespace project;
using Image = typename image::Image;
using Camera = typename views::Camera;
using TriangleList = typename triangles::TriangleList;
using Transform = typename transforms::Transform;

int main() {
    int height = 1000;
    int width  = 1000;
    Image image = Image(width, height);
    TriangleList list = skel::Get3DTriangles();

    int f = 0;
    //for (int f=0; f < 1000; ++f) {
    //  if (f % 250 != 0)
    //    continue;

    //  image.zfill();
    Camera camera = skel::GetCamera(f, 1000);
    Transform transform = transforms::world_to_device(camera, double(height), double(width));
    for (int i=0; i < list.numTriangles; ++i)
      algorithms::RasterizeGoingUpTriangle(transforms::transform_triangle(transform, list.triangles[i]), image);

    io::Image2PNM(image, gen_filename(f));
    //}

    return 0;
}
