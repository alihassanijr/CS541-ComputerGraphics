/*

Ali Hassani

Project 1-C

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

double TOL = 0.0;

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

void Image2PNM(Image img, string fn) {
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
  fwrite(img.get_data(), x_length * y_length, sizeof(Pixel), f);
  fclose(f);
}

struct TriangleList {
  int numTriangles;
  Triangle *triangles;
};

template <typename T>
T abs_difference(T a, T b) {
  T diff = a - b;
  if (diff < 0)
    return -1 * diff;
  return diff;
}

struct Line {
  double m, b, x_left, x_right;

  Line(): m(0), b(0), x_left(0), x_right(0) {}
  Line (double m, double b, double x_left, double x_right): m(m), b(b) {}

  double intersect(double y) {
    return (y - b) / m;
  }

  double intersectL(double y) {
    return (m == 0) ? (x_left) : ((y - b) / m);
  }

  double intersectR(double y) {
    return (m == 0) ? (x_right) : ((y - b) / m);
  }
};

Line intercept(double x0, double y0, double x1, double y1) {
  if (abs_difference(x0, x1) == 0) {
      //cerr << "Uh-oh, invalid coordinates, got <" << x0 << ", " << y0 << "> and <" << x1 << ", " << y1 << ">" << endl;
      //terminate();
      return Line(0, 0, std::min(x0, x1), std::max(x0, x1));
  }
  assert(abs_difference(y0, y1) > 0);
  double m = (y1 - y0) / (x1 - x0);
  double b = y1 - (m * x1);
  return Line(m, b, -1, -1);
}

void fillBottomTriangle(Triangle &t, Image &x, vector<int> sorted_idx) {
  int rowMin = C441(t.Y[sorted_idx[2]]);
  int rowMax = /*F441(t.Y[sorted_idx[0]])*/ F441(t.Y[sorted_idx[1]]);
  // In the bottom half our anchor is the bottom most vertex
  double anchor[2] = {t.X[sorted_idx[2]], t.Y[sorted_idx[2]]};
  double *leftv, *rightv;
  if (t.X[sorted_idx[1]] < t.X[sorted_idx[0]]) {
    double _leftv[2]  = {t.X[sorted_idx[1]], t.Y[sorted_idx[1]]};
    double _rightv[2] = {t.X[sorted_idx[0]], t.Y[sorted_idx[0]]};
    leftv = _leftv;
    rightv = _rightv;
  } else {
    double _leftv[2]  = {t.X[sorted_idx[0]], t.Y[sorted_idx[0]]};
    double _rightv[2] = {t.X[sorted_idx[1]], t.Y[sorted_idx[1]]};
    leftv = _leftv;
    rightv = _rightv;
  }
  Line leftInt  = intercept( leftv[0],  leftv[1], anchor[0], anchor[1]);
  Line rightInt = intercept(rightv[0], rightv[1], anchor[0], anchor[1]);
  // Fill bottom half
  for (int r=rowMin; r <= rowMax; ++r) {
    double leftEndD  =  leftInt.intersectL(r); // This put me off by ~ 60 pixels for a while
    double rightEndD = rightInt.intersectR(r); // This put me off by ~ 60 pixels for a while
    if (leftEndD > rightEndD) {
      double tmp = rightEndD;
      rightEndD = leftEndD;
      leftEndD  = tmp;
    }
    int leftEnd  = C441( leftEndD);
    int rightEnd = F441(rightEndD);
    for (int c = leftEnd; c <= rightEnd; ++c) {
      x.set_pixel(r, c, Pixel(t.color));
    }
  }
}

void fillTopTriangle(Triangle &t, Image &x, vector<int> sorted_idx) {
  int rowMin = /*C441(t.Y[sorted_idx[2]])*/ C441(t.Y[sorted_idx[1]]);
  int rowMax = F441(t.Y[sorted_idx[0]]);
  // In the top half our anchor is the top most vertex
  double anchor[2] = {t.X[sorted_idx[0]], t.Y[sorted_idx[0]]};
  double *leftv, *rightv;
  if (t.X[sorted_idx[1]] < t.X[sorted_idx[2]]) {
    double _leftv[2]  = {t.X[sorted_idx[1]], t.Y[sorted_idx[1]]};
    double _rightv[2] = {t.X[sorted_idx[2]], t.Y[sorted_idx[2]]};
    leftv = _leftv;
    rightv = _rightv;
  } else {
    double _leftv[2]  = {t.X[sorted_idx[2]], t.Y[sorted_idx[2]]};
    double _rightv[2] = {t.X[sorted_idx[1]], t.Y[sorted_idx[1]]};
    leftv = _leftv;
    rightv = _rightv;
  }
  Line leftInt  = intercept( leftv[0],  leftv[1], anchor[0], anchor[1]);
  Line rightInt = intercept(rightv[0], rightv[1], anchor[0], anchor[1]);
  // Fill top half
  for (int r=rowMin; r <= rowMax; ++r) {
    double leftEndD  =  leftInt.intersectL(r); // This put me off by ~ 60 pixels for a while
    double rightEndD = rightInt.intersectR(r); // This put me off by ~ 60 pixels for a while
    if (leftEndD > rightEndD) {
      double tmp = rightEndD;
      rightEndD = leftEndD;
      leftEndD  = tmp;
    }
    int leftEnd  = C441( leftEndD);
    int rightEnd = F441(rightEndD);
    for (int c = leftEnd; c <= rightEnd; ++c) {
      x.set_pixel(r, c, Pixel(t.color));
    }
  }
}

/// RasterizeGoingUpTriangle
void RasterizeGoingUpTriangle(Triangle &t, Image &x) {
  vector<int> sorted_idx = sorted_y_idx(t);

  fillBottomTriangle(t, x, sorted_idx);
  fillTopTriangle(   t, x, sorted_idx);
}

TriangleList GetTriangles(int small_read) {
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


int main() {
    cout << "Generating image" << endl;

    Image x = Image(1786, 1344);
    TriangleList list = GetTriangles(0);

    for (int i=0; i < list.numTriangles; ++i)
      RasterizeGoingUpTriangle(list.triangles[i], x);

    cout << "Saving image" << endl;

    Image2PNM(x, "proj1C_out.pnm");

    return 0;
}
