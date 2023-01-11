/*

Ali Hassani

*/
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

using namespace std;


struct Pixel {
  unsigned char r, g, b;

  Pixel(): r(0), g(0), b(0) {}

  Pixel(unsigned char r, unsigned char g, unsigned char b): r(r), g(g), b(b) {}

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

const Pixel BLACK(0, 0, 0);
const Pixel GRAY(128, 128, 128);
const Pixel WHITE(255, 255, 255);
const Pixel RED(255, 0, 0);
const Pixel GREEN(0, 255, 0);
const Pixel BLUE(0, 0, 255);
const Pixel PURPLE(255, 0, 255);
const Pixel CYAN(0, 255, 255);
const Pixel YELLOW(255, 255, 0);

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

  void set_pixel(int x, int y, Pixel v) {
    _arr[x * params.stride(0) + y].set_value(v);
  }

  void set_pixels(int x_start, int y_start, int x_end, int y_end, Pixel v) {
    for (int x = x_start; x < x_end; ++x) {
      for (int y = y_start; y < y_end; ++y) {
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


int main() {
    cout << "Generating image" << endl;

    Image x = Image(300, 300);

    /* Row 1 */
    x.set_pixels(  0,   0, 100, 100, BLACK);
    x.set_pixels(  0, 100, 100, 200, GRAY);
    x.set_pixels(  0, 200, 100, 300, WHITE);

    /* Row 2 */
    x.set_pixels(100,   0, 200, 100, RED);
    x.set_pixels(100, 100, 200, 200, GREEN);
    x.set_pixels(100, 200, 200, 300, BLUE);

    /* Row 3 */
    x.set_pixels(200,   0, 300, 100, PURPLE);
    x.set_pixels(200, 100, 300, 200, CYAN);
    x.set_pixels(200, 200, 300, 300, YELLOW);

    cout << "Saving image" << endl;

    Image2PNM(x, "proj1A_out.pnm");

    return 0;
}
