/*

Ali Hassani

Project 2-B

CS 441/541

*/

#define WIN_HEIGHT 700
#define WIN_WIDTH  700

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using std::endl;
using std::cerr;

#include <GL/glew.h>
#include <GLFW/glfw3.h> // GLFW helper library

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>   // glm::vec3
#include <glm/vec4.hpp>   // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale

class RenderManager;

void        SetUpDog(RenderManager &, float);
const char *GetVertexShader();
const char *GetFragmentShader();

// This file is split into four parts:
// - Part 1: code to set up spheres and cylinders
// - Part 2: a "RenderManager" module
// - Part 3: main function
// - Part 4: SetUpDog and the shader programs -- things you modify
//
// It is intended that you will only need to modify code in Part 4.
// That said, you will need functions in Part 2 and should review
// those functions.
// Further, you are encouraged to look through the entire code base.
//


//
//
// PART 1: code to set up spheres and cylinders
//
//

class Triangle
{
  public:
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

std::vector<Triangle> SplitTriangle(std::vector<Triangle> &list)
{
    std::vector<Triangle> output(4*list.size());
    output.resize(4*list.size());
    for (unsigned int i = 0 ; i < list.size() ; i++)
    {
        Triangle t = list[i];
        glm::vec3 vmid1, vmid2, vmid3;
        vmid1 = (t.v0 + t.v1) / 2.0f;
        vmid2 = (t.v1 + t.v2) / 2.0f;
        vmid3 = (t.v0 + t.v2) / 2.0f;
        output[4*i+0].v0 = t.v0;
        output[4*i+0].v1 = vmid1;
        output[4*i+0].v2 = vmid3;
        output[4*i+1].v0 = t.v1;
        output[4*i+1].v1 = vmid2;
        output[4*i+1].v2 = vmid1;
        output[4*i+2].v0 = t.v2;
        output[4*i+2].v1 = vmid3;
        output[4*i+2].v2 = vmid2;
        output[4*i+3].v0 = vmid1;
        output[4*i+3].v1 = vmid2;
        output[4*i+3].v2 = vmid3;
    }
    return output;
}

void PushVertex(std::vector<float>& coords,
                const glm::vec3& v)
{
  coords.push_back(v.x);
  coords.push_back(v.y);
  coords.push_back(v.z);
}

//
// Sets up a cone
// Z=0 to Z=1.
//
void GetConeData(std::vector<float>& coords, std::vector<float>& normals)
{
  int nfacets = 30;
  for (int i = 0 ; i < nfacets ; i++)
  {
    double angle = M_PI*2.0*i/nfacets;
    double nextAngle = (i == nfacets-1 ? 0 : M_PI*2.0*(i+1)/nfacets);
    glm::vec3 fnormal(0.0f, 0.0f, 1.0f);
    glm::vec3 bnormal(0.0f, 0.0f, -1.0f);
    glm::vec3 fv0(0.0f, 0.0f, 1.0f);
    glm::vec3 fv1(cos(angle),     sin(angle), 1);
    glm::vec3 fv2(cos(nextAngle), sin(nextAngle), 1);
    glm::vec3 bv0(0.0f, 0.0f, 0.0f);
    glm::vec3 bv1(0.5*cos(angle),     0.5*sin(angle), 0);
    glm::vec3 bv2(0.5*cos(nextAngle), 0.5*sin(nextAngle), 0);
    // top and bottom circle vertices
    PushVertex(coords, fv0);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv1);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv2);
    PushVertex(normals, fnormal);
    PushVertex(coords, bv0);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv1);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv2);
    PushVertex(normals, bnormal);
    // curves surface vertices
    glm::vec3 v1normal(cos(angle),     sin(angle), 0);
    glm::vec3 v2normal(cos(nextAngle), sin(nextAngle), 0);
    //fv1 fv2 bv1
    PushVertex(coords, fv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    //fv2 bv1 bv2
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, bv2);
    PushVertex(normals, v2normal);
  }
}

//
// Sets up a cylinder that is the circle x^2+y^2=1 extruded from
// Z=0 to Z=1.
//
void GetCylinderData(std::vector<float>& coords, std::vector<float>& normals)
{
  int nfacets = 30;
  for (int i = 0 ; i < nfacets ; i++)
  {
    double angle = M_PI*2.0*i/nfacets;
    double nextAngle = (i == nfacets-1 ? 0 : M_PI*2.0*(i+1)/nfacets);
    glm::vec3 fnormal(0.0f, 0.0f, 1.0f);
    glm::vec3 bnormal(0.0f, 0.0f, -1.0f);
    glm::vec3 fv0(0.0f, 0.0f, 1.0f);
    glm::vec3 fv1(cos(angle), sin(angle), 1);
    glm::vec3 fv2(cos(nextAngle), sin(nextAngle), 1);
    glm::vec3 bv0(0.0f, 0.0f, 0.0f);
    glm::vec3 bv1(cos(angle), sin(angle), 0);
    glm::vec3 bv2(cos(nextAngle), sin(nextAngle), 0);
    // top and bottom circle vertices
    PushVertex(coords, fv0);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv1);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv2);
    PushVertex(normals, fnormal);
    PushVertex(coords, bv0);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv1);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv2);
    PushVertex(normals, bnormal);
    // curves surface vertices
    glm::vec3 v1normal(cos(angle), sin(angle), 0);
    glm::vec3 v2normal(cos(nextAngle), sin(nextAngle), 0);
    //fv1 fv2 bv1
    PushVertex(coords, fv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    //fv2 bv1 bv2
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, bv2);
    PushVertex(normals, v2normal);
  }
}

//
// Sets up a hemisphere
//
void
GetHemisphereData(std::vector<float>& coords, std::vector<float>& normals)
{
  int recursionLevel = 3;
  std::vector<Triangle> list;
  {
    Triangle t;
    t.v0 = glm::vec3(1.0f,0.0f,0.0f);
    t.v1 = glm::vec3(0.0f,1.0f,0.0f);
    t.v2 = glm::vec3(0.0f,0.0f,1.0f);
    list.push_back(t);
  }
  for (int r = 0 ; r < recursionLevel ; r++)
  {
      list = SplitTriangle(list);
  }

  for (int octant = 0 ; octant < 4 ; octant++)
  {
    glm::mat4 view(1.0f);
    float angle = 90.0f*octant;
    if(angle != 0.0f)
      view = glm::rotate(view, glm::radians(angle), glm::vec3(1, 0, 0));
    if (octant >= 4)
      view = glm::rotate(view, glm::radians(180.0f), glm::vec3(0, 0, 1));
    for(int i = 0; i < list.size(); i++)
    {
      Triangle t = list[i];
      float mag_reci;
      glm::vec3 v0 = view*glm::vec4(t.v0, 1.0f);
      glm::vec3 v1 = view*glm::vec4(t.v1, 1.0f);
      glm::vec3 v2 = view*glm::vec4(t.v2, 1.0f);
      mag_reci = 1.0f / glm::length(v0);
      v0 = glm::vec3(v0.x * mag_reci, v0.y * mag_reci, v0.z * mag_reci);
      mag_reci = 1.0f / glm::length(v1);
      v1 = glm::vec3(v1.x * mag_reci, v1.y * mag_reci, v1.z * mag_reci);
      mag_reci = 1.0f / glm::length(v2);
      v2 = glm::vec3(v2.x * mag_reci, v2.y * mag_reci, v2.z * mag_reci);
      PushVertex(coords, v0);
      PushVertex(coords, v1);
      PushVertex(coords, v2);
      PushVertex(normals, v0);
      PushVertex(normals, v1);
      PushVertex(normals, v2);
    }
  }
}

//
// Sets up a sphere with equation x^2+y^2+z^2=1
//
void
GetSphereData(std::vector<float>& coords, std::vector<float>& normals)
{
  int recursionLevel = 3;
  std::vector<Triangle> list;
  {
    Triangle t;
    t.v0 = glm::vec3(1.0f,0.0f,0.0f);
    t.v1 = glm::vec3(0.0f,1.0f,0.0f);
    t.v2 = glm::vec3(0.0f,0.0f,1.0f);
    list.push_back(t);
  }
  for (int r = 0 ; r < recursionLevel ; r++)
  {
      list = SplitTriangle(list);
  }

  for (int octant = 0 ; octant < 8 ; octant++)
  {
    glm::mat4 view(1.0f);
    float angle = 90.0f*(octant%4);
    if(angle != 0.0f)
      view = glm::rotate(view, glm::radians(angle), glm::vec3(1, 0, 0));
    if (octant >= 4)
      view = glm::rotate(view, glm::radians(180.0f), glm::vec3(0, 0, 1));
    for(int i = 0; i < list.size(); i++)
    {
      Triangle t = list[i];
      float mag_reci;
      glm::vec3 v0 = view*glm::vec4(t.v0, 1.0f);
      glm::vec3 v1 = view*glm::vec4(t.v1, 1.0f);
      glm::vec3 v2 = view*glm::vec4(t.v2, 1.0f);
      mag_reci = 1.0f / glm::length(v0);
      v0 = glm::vec3(v0.x * mag_reci, v0.y * mag_reci, v0.z * mag_reci);
      mag_reci = 1.0f / glm::length(v1);
      v1 = glm::vec3(v1.x * mag_reci, v1.y * mag_reci, v1.z * mag_reci);
      mag_reci = 1.0f / glm::length(v2);
      v2 = glm::vec3(v2.x * mag_reci, v2.y * mag_reci, v2.z * mag_reci);
      PushVertex(coords, v0);
      PushVertex(coords, v1);
      PushVertex(coords, v2);
      PushVertex(normals, v0);
      PushVertex(normals, v1);
      PushVertex(normals, v2);
    }
  }
}


//
//
// PART 2: RenderManager module
//
//

void _print_shader_info_log(GLuint shader_index) {
  int max_length = 2048;
  int actual_length = 0;
  char shader_log[2048];
  glGetShaderInfoLog(shader_index, max_length, &actual_length, shader_log);
  printf("shader info log for GL index %u:\n%s\n", shader_index, shader_log);
}

class RenderManager
{
  public:
   enum ShapeType
   {
      SPHERE,
      HEMISPHERE,
      CYLINDER,
      CONE
   };

                 RenderManager();
   void          SetView(glm::vec3 &c, glm::vec3 &, glm::vec3 &);
   void          SetUpGeometry();
   void          SetColor(double r, double g, double b);
   void          Render(ShapeType, glm::mat4 model);
   GLFWwindow   *GetWindow() { return window; };

  private:
   glm::vec3 color;
   GLuint sphereVAO;
   GLuint sphereNumPrimitives;
   GLuint hemisphereVAO;
   GLuint hemisphereNumPrimitives;
   GLuint cylinderVAO;
   GLuint cylinderNumPrimitives;
   GLuint coneVAO;
   GLuint coneNumPrimitives;
   GLuint mvploc;
   GLuint modelloc;
   GLuint colorloc;
   GLuint camloc;
   GLuint ldirloc;
   glm::mat4 projection;
   glm::mat4 view;
   GLuint shaderProgram;
   GLFWwindow *window;

   void SetUpWindowAndShaders();
   void MakeModelView(glm::mat4 &);
};

RenderManager::RenderManager()
{
  SetUpWindowAndShaders();
  SetUpGeometry();
  projection = glm::perspective(
        glm::radians(45.0f), (float)1000 / (float)1000,  5.0f, 100.0f);

  // Get a handle for our MVP and color uniforms
  mvploc = glGetUniformLocation(shaderProgram, "MVP");
  modelloc = glGetUniformLocation(shaderProgram, "model");
  colorloc = glGetUniformLocation(shaderProgram, "color");
  camloc = glGetUniformLocation(shaderProgram, "cameraloc");
  ldirloc = glGetUniformLocation(shaderProgram, "lightdir");

  glm::vec4 lightcoeff(0.3, 0.7, 2.0, 50.5); // Lighting coeff, Ka, Kd, Ks, alpha
  GLuint lcoeloc = glGetUniformLocation(shaderProgram, "lightcoeff");
  glUniform4fv(lcoeloc, 1, &lightcoeff[0]);
}

void
RenderManager::SetView(glm::vec3 &camera, glm::vec3 &origin, glm::vec3 &up)
{ 
   glm::mat4 v = glm::lookAt(
                       camera, // Camera in world space
                       origin, // looks at the origin
                       up      // and the head is up
                 );
   view = v; 
   glUniform3fv(camloc, 1, &camera[0]);
   // Direction of light
   glm::vec3 lightdir = glm::normalize(camera - origin);   
   glUniform3fv(ldirloc, 1, &lightdir[0]);
};

void
RenderManager::SetUpWindowAndShaders()
{
  // start GL context and O/S window using the GLFW helper library
  if (!glfwInit()) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "CS 541", NULL, NULL);
  if (!window) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);
  // start GLEW extension handler
  glewExperimental = GL_TRUE;
  glewInit();

  // get version info
  const GLubyte *renderer = glGetString(GL_RENDERER); // get renderer string
  const GLubyte *version = glGetString(GL_VERSION);   // version as a string
  printf("Renderer: %s\n", renderer);
  printf("OpenGL version supported %s\n", version);

  // tell GL to only draw onto a pixel if the shape is closer to the viewer
  glEnable(GL_DEPTH_TEST); // enable depth-testing
  //glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"
  glDepthFunc(GL_LEQUAL); 

  const char* vertex_shader = GetVertexShader();
  const char* fragment_shader = GetFragmentShader();

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vertex_shader, NULL);
  glCompileShader(vs);
  int params = -1;
  glGetShaderiv(vs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: GL shader index %i did not compile\n", vs);
    _print_shader_info_log(vs);
    exit(EXIT_FAILURE);
  }

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fragment_shader, NULL);
  glCompileShader(fs);
  glGetShaderiv(fs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: GL shader index %i did not compile\n", fs);
    _print_shader_info_log(fs);
    exit(EXIT_FAILURE);
  }

  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, fs);
  glAttachShader(shaderProgram, vs);
  glLinkProgram(shaderProgram);
  glUseProgram(shaderProgram);
}

void RenderManager::SetColor(double r, double g, double b)
{
   color[0] = r;
   color[1] = g;
   color[2] = b;
}

void RenderManager::MakeModelView(glm::mat4 &model)
{
   glm::mat4 modelview = projection * view * model;
   glUniformMatrix4fv(mvploc, 1, GL_FALSE, &modelview[0][0]);
   glUniformMatrix4fv(modelloc, 1, GL_FALSE, &model[0][0]);
}

void RenderManager::Render(ShapeType st, glm::mat4 model)
{
   int numPrimitives = 0;
   if (st == SPHERE)
   {
      glBindVertexArray(sphereVAO);
      numPrimitives = sphereNumPrimitives;
   }
   else if (st == HEMISPHERE)
   {
      glBindVertexArray(hemisphereVAO);
      numPrimitives = hemisphereNumPrimitives;
   }
   else if (st == CYLINDER)
   {
      glBindVertexArray(cylinderVAO);
      numPrimitives = cylinderNumPrimitives;
   }
   else if (st == CONE)
   {
      glBindVertexArray(coneVAO);
      numPrimitives = coneNumPrimitives;
   }
   MakeModelView(model);
   glUniform3fv(colorloc, 1, &color[0]);
   glDrawElements(GL_TRIANGLES, numPrimitives, GL_UNSIGNED_INT, NULL);
}

void SetUpVBOs(std::vector<float> &coords, std::vector<float> &normals,
               GLuint &points_vbo, GLuint &normals_vbo, GLuint &index_vbo)
{
  int numIndices = coords.size()/3;
  std::vector<GLuint> indices(numIndices);
  for(int i = 0; i < numIndices; i++)
    indices[i] = i;

  points_vbo = 0;
  glGenBuffers(1, &points_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
  glBufferData(GL_ARRAY_BUFFER, coords.size() * sizeof(float), coords.data(), GL_STATIC_DRAW);

  normals_vbo = 0;
  glGenBuffers(1, &normals_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
  glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);

  index_vbo = 0;    // Index buffer object
  glGenBuffers(1, &index_vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
}

void RenderManager::SetUpGeometry()
{
  std::vector<float> sphereCoords;
  std::vector<float> sphereNormals;
  GetSphereData(sphereCoords, sphereNormals);
  sphereNumPrimitives = sphereCoords.size() / 3;
  GLuint sphere_points_vbo, sphere_normals_vbo, sphere_indices_vbo;
  SetUpVBOs(sphereCoords, sphereNormals, 
            sphere_points_vbo, sphere_normals_vbo, sphere_indices_vbo);

  std::vector<float> hemisphereCoords;
  std::vector<float> hemisphereNormals;
  GetHemisphereData(hemisphereCoords, hemisphereNormals);
  hemisphereNumPrimitives = hemisphereCoords.size() / 3;
  GLuint hemisphere_points_vbo, hemisphere_normals_vbo, hemisphere_indices_vbo;
  SetUpVBOs(hemisphereCoords, hemisphereNormals, 
            hemisphere_points_vbo, hemisphere_normals_vbo, hemisphere_indices_vbo);

  std::vector<float> cylCoords;
  std::vector<float> cylNormals;
  GetCylinderData(cylCoords, cylNormals);
  cylinderNumPrimitives = cylCoords.size() / 3;
  GLuint cyl_points_vbo, cyl_normals_vbo, cyl_indices_vbo;
  SetUpVBOs(cylCoords, cylNormals, 
            cyl_points_vbo, cyl_normals_vbo, cyl_indices_vbo);

  std::vector<float> coneCoords;
  std::vector<float> coneNormals;
  GetConeData(coneCoords, coneNormals);
  coneNumPrimitives = coneCoords.size() / 3;
  GLuint cone_points_vbo, cone_normals_vbo, cone_indices_vbo;
  SetUpVBOs(coneCoords, coneNormals, 
            cone_points_vbo, cone_normals_vbo, cone_indices_vbo);

  GLuint vao[6];
  glGenVertexArrays(6, vao);

  glBindVertexArray(vao[SPHERE]);
  sphereVAO = vao[SPHERE];
  glBindBuffer(GL_ARRAY_BUFFER, sphere_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, sphere_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(vao[HEMISPHERE]);
  hemisphereVAO = vao[HEMISPHERE];
  glBindBuffer(GL_ARRAY_BUFFER, hemisphere_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, hemisphere_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, hemisphere_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(vao[CYLINDER]);
  cylinderVAO = vao[CYLINDER];
  glBindBuffer(GL_ARRAY_BUFFER, cyl_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, cyl_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cyl_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(vao[CONE]);
  coneVAO = vao[CONE];
  glBindBuffer(GL_ARRAY_BUFFER, cone_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, cone_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cone_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
}

//
// PART3: main function
//
int main() 
{
  RenderManager rm;
  GLFWwindow *window = rm.GetWindow();

  glm::vec3 origin(0, 0, 0);
  glm::vec3 up(0, 1, 0);

  int counter=0;
  while (!glfwWindowShouldClose(window)) 
  {
    double angle=counter/300.0*2*M_PI;
    double distance = sin(float(counter)/20.0) * 3 + 10;
    counter++;

    glm::vec3 camera(distance*sin(angle), -0.5, distance*cos(angle));
    rm.SetView(camera, origin, up);

    // wipe the drawing surface clear
    glClearColor(0.3, 0.3, 0.8, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SetUpDog(rm, counter);

    // update other events like input handling
    glfwPollEvents();
    // put the stuff we've been drawing onto the display
    glfwSwapBuffers(window);
  }

  // close GL context and any other GLFW resources
  glfwTerminate();
  return 0;
}

glm::mat4 RotateMatrix(float degrees, float x, float y, float z)
{
   glm::mat4 identity(1.0f);
   glm::mat4 rotation = glm::rotate(identity, 
                                    glm::radians(degrees), 
                                    glm::vec3(x, y, z));
   return rotation;
}

glm::mat4 ScaleMatrix(double x, double y, double z)
{
   glm::mat4 identity(1.0f);
   glm::vec3 scale(x, y, z);
   return glm::scale(identity, scale);
}

glm::mat4 TranslateMatrix(double x, double y, double z)
{
   glm::mat4 identity(1.0f);
   glm::vec3 translate(x, y, z);
   return glm::translate(identity, translate);
}

glm::mat4 gen_rot(int axis, float deg) {
  if (axis == 0)
    return RotateMatrix(deg, 1, 0, 0);
  else if (axis == 1)
    return RotateMatrix(deg, 0, 1, 0);
  return RotateMatrix(deg, 0, 0, 1);
}

glm::mat4 gen_translate(int axis, float v) {
  if (axis == 0)
    return TranslateMatrix(v, 0, 0);
  else if (axis == 1)
    return TranslateMatrix(0, v, 0);
  return TranslateMatrix(0, 0, v);
}

glm::mat4 gen_scale(int axis, float scale, float axis_scale) {
  if (axis == 0)
    return ScaleMatrix(axis_scale, scale, scale);
  else if (axis == 1)
    return ScaleMatrix(scale, axis_scale, scale);
  return ScaleMatrix(scale, scale, axis_scale);
}

glm::mat4 gen_axis_rot(int axis) {
  if (axis == 0)
    return RotateMatrix(-90, 0, 1, 0);
  else if (axis == 1)
    return RotateMatrix(-90, 1, 0, 0);
  return glm::mat4(1.0);
}

glm::mat4 Shape1(glm::mat4 modelSoFar, RenderManager &rm, 
            float scale_start, 
            float scale_end,
            float steps, 
            float length,
            float rotate_start,
            float rotate_end,
            int rotate_axis,
            int sign,
            float power = 1.0,
            int axis = 1,
            glm::mat4 post_transform = glm::mat4(1.0f)
            ) {
  float l = length / steps;
  for (int i=0; i < int(steps); ++i) {
    float a = float(i) * l;
    float m = float(i) / steps;
    float s = pow(m, power);
    float sc = scale_start*(1-s) + s*scale_end;
    float d = rotate_start*(1-m) + m*rotate_end;
    glm::mat4 scale = gen_scale(axis, sc, l);
    glm::mat4 shift = gen_translate(axis, sign * a);
    glm::mat4 rotate = gen_rot(rotate_axis, d);
    glm::mat4 axis_rot = gen_axis_rot(axis);
    rm.Render(RenderManager::CONE, modelSoFar*rotate*shift*scale*post_transform*axis_rot);
  }
  return modelSoFar * gen_rot(rotate_axis, rotate_end) * gen_translate(axis, sign * length);
}

glm::mat4 Shape2(glm::mat4 modelSoFar, RenderManager &rm, 
            float scale_start, 
            float scale_end,
            float steps, 
            float length,
            float rotate_start,
            float rotate_end,
            int rotate_axis,
            int sign,
            float power = 1.0,
            int axis = 1,
            glm::mat4 post_transform = glm::mat4(1.0f)
            ) {
  float l = length / steps;
  for (int i=0; i < int(steps); ++i) {
    float a = float(i) * l;
    float m = float(i) / steps;
    float s = pow(m, power);
    float sc = scale_start*(1-s) + s*scale_end;
    float d = rotate_start*(1-m) + m*rotate_end;
    glm::mat4 scale = gen_scale(axis, sc, l);
    glm::mat4 shift = gen_translate(axis, sign * a);
    glm::mat4 rotate = gen_rot(rotate_axis, d);
    glm::mat4 axis_rot = gen_axis_rot(axis);
    rm.Render(RenderManager::CYLINDER, modelSoFar*rotate*shift*scale*post_transform*axis_rot);
  }
  return modelSoFar * gen_rot(rotate_axis, rotate_end) * gen_translate(axis, sign * length);
}

void SetUpBone(glm::mat4 modelSoFar, RenderManager &rm, float t) {
  glm::mat4 offset = TranslateMatrix(0.0, 0.9, 0.0);
  modelSoFar = modelSoFar * offset;
  
  glm::mat4 scale = ScaleMatrix(0.8, 0.1, 0.1);
  rm.SetColor(1, 1, 1);
  glm::mat4 boneOff = TranslateMatrix(0.8, 0.0, 0.0);
  rm.Render(RenderManager::CYLINDER, modelSoFar*boneOff*scale*gen_axis_rot(0)/* cylinder through X axis instead of Z */);

  scale = ScaleMatrix(0.1, 0.1, 0.1);
  glm::mat4 shift = TranslateMatrix(0.0, -0.5, 0.0);
  rm.Render(RenderManager::SPHERE, modelSoFar*scale*shift);
  shift = TranslateMatrix(0.0, 0.5, 0.0);
  rm.Render(RenderManager::SPHERE, modelSoFar*scale*shift);

  shift = TranslateMatrix(7.8, -0.5, 0.0);
  rm.Render(RenderManager::SPHERE, modelSoFar*scale*shift);
  shift = TranslateMatrix(7.8, 0.5, 0.0);
  rm.Render(RenderManager::SPHERE, modelSoFar*scale*shift);
}

void SetUpHeadBase(glm::mat4 modelSoFar, RenderManager &rm, float t)
{
  // A lot of trickery got this to the point where it doesn't show those black lines anymore
  // (or if we do two-sided lighting, just overly bright points)
  //
  // The reason why we get those in the first place is because we're using disks. And disks are terrible,
  // because their top and bottom sides are sharp.
  //
  // Managed to avoid that in head by setting up half of those cylinders to be "half-cones".
  // Also had to make the first half cone just a touch bigger to avoid the dark side from the cylinder directly above it.
  rm.SetColor(1, 232.0f/255.0f, 205.0f/255.0f);
  for (int i=0; i < 500; ++i) {
    float s = 0.5 - (pow(float(i) / 500.0, 4) / 2);
    glm::mat4 scale = ScaleMatrix(s, 0.1, s);
    glm::mat4 shift = TranslateMatrix(0, 0.0 + (float(i)/1000.0), -0.45);
    rm.Render(RenderManager::CYLINDER, modelSoFar*shift*scale*gen_axis_rot(1)/* cylinder through Y axis instead of Z */);
  }
  for (int i=0; i < 500; ++i) {
    float s = 0.501 - (pow(float(i) / 500.0, 4) / 2);
    glm::mat4 scale = ScaleMatrix(s, 0.1, s);
    glm::mat4 shift = TranslateMatrix(0, 0.0 - (float(i)/1000.0), -0.45);
    rm.Render(RenderManager::CONE, modelSoFar*shift*scale*gen_axis_rot(1)/* cylinder through Y axis instead of Z */);
  }
}

void SetUpNose(glm::mat4 modelSoFar, RenderManager &rm, float t)
{
  float breathing = sin(t/10) * 0.0025;
  modelSoFar = modelSoFar * TranslateMatrix(0, breathing, 0);
  glm::mat4 shift = TranslateMatrix(0.025, 0.0, -0.025);
  glm::mat4 scale = ScaleMatrix(0.2, 0.1667, -0.1);
  rm.SetColor(219.0f/255.0f, 169.0f/255.0f, 139.0f/255.0f);
  rm.Render(RenderManager::SPHERE, modelSoFar*shift*scale);

  shift = TranslateMatrix(0.025, 0.0, 0.025);
  scale = ScaleMatrix(0.065, 0.05, -0.1);
  rm.SetColor(130.0f/255.0f, 103.0f/255.0f, 94.0f/255.0f);
  rm.Render(RenderManager::SPHERE, modelSoFar*shift*scale);
}

void SetUpEyeball(glm::mat4 modelSoFar, glm::mat4 eye_rotation, RenderManager &rm, float t)
{
  float blinkfactor = pow(sin(pow(sin(t / 15.0), 14)), 4);
  float lidjump = blinkfactor * 0.01;
  glm::mat4 scale = ScaleMatrix(0.15, 0.185, 0.115);
  glm::mat4 translate = TranslateMatrix(0, 0.35, -0.4);
  glm::mat4 rotate = RotateMatrix(-45, 1, 0, 0);
  rm.SetColor(254.0f/255.0f, 188.0f/255.0f, 156.0f/255.0f);
  rm.Render(RenderManager::SPHERE, modelSoFar*scale*rotate*translate);

  //// Set up upper lid
  glm::mat4 ulidscale = ScaleMatrix(0.125, 0.125, 0.125);
  glm::mat4 ulidtranslate = TranslateMatrix(0, lidjump, -0.025);
  glm::mat4 ulidrotateZ = RotateMatrix(90, 0, 0, 1);
  glm::mat4 ulidrotateY = RotateMatrix(40, 0, 1, 0);
  rm.SetColor(242.0f/255.0f, 157.0f/255.0f, 129.0f/255.0f);
  rm.Render(RenderManager::HEMISPHERE, modelSoFar*ulidtranslate*ulidscale*ulidrotateZ*ulidrotateY);

  for (int i=0; i < 20; ++i){
    ulidtranslate = TranslateMatrix(0.0, lidjump + 0.1 * float(i)/40, 0.0);
    float darkness = float(i) / 100;
    rm.SetColor(242.0f/255.0f - darkness, 157.0f/255.0f - darkness, 129.0f/255.0f - darkness);
    rm.Render(RenderManager::HEMISPHERE, modelSoFar*ulidtranslate*ulidscale*ulidrotateZ*ulidrotateY);
  }

  //// Set up eye lid
  glm::mat4 lidscale = ScaleMatrix(0.1025, 0.1025, 0.1025);
  glm::mat4 lidtranslate = TranslateMatrix(0, 0.0, 0.02);
  glm::mat4 lidrotateZ = RotateMatrix(90, 0, 0, 1);
  glm::mat4 lidrotateY = RotateMatrix(10 - 120 * blinkfactor, 0, 1, 0);
  rm.SetColor(144.0f/255.0f, 100.0f/255.0f, 77.0f/255.0f);
  rm.Render(RenderManager::HEMISPHERE, modelSoFar*lidtranslate*lidscale*lidrotateZ*lidrotateY);

  // Set up eye ball
  modelSoFar = modelSoFar * eye_rotation;
  glm::mat4 scaled10 = ScaleMatrix(0.1, 0.1, 0.1);
  rm.SetColor(1,1,1);
  rm.Render(RenderManager::SPHERE, modelSoFar*scaled10);

  // Pupil
  glm::mat4 scaleB = ScaleMatrix(0.085, 0.085, 0.085);
  glm::mat4 translateB = TranslateMatrix(0, -0.05, 0.3);
  rm.SetColor(181.0f/255.0f, 59.0f/255.0f, 40.0f/255.0f);
  rm.Render(RenderManager::SPHERE, modelSoFar*scaleB*translateB);

  scaleB = ScaleMatrix(0.06, 0.06, 0.06);
  translateB = TranslateMatrix(0, 0, 0.9);
  rm.SetColor(21.0f/255.0f, 22.0f/255.0f, 21.0f/256.0f);
  rm.Render(RenderManager::SPHERE, modelSoFar*scaleB*translateB);
}

void SetUpStache(glm::mat4 modelSoFar, RenderManager &rm, int sign) {
  rm.SetColor(99.0f/255.0f, 65.0f/255.0f, 59.0f/255.0f);
  modelSoFar = modelSoFar * RotateMatrix(sign*45, 0, 0, 1);
  Shape1(modelSoFar, rm, 0.075, 0.001, 100.0, 0.2,
        0, 0, 2,
        1, 5.0,
        1);
  Shape1(modelSoFar, rm, 0.075, 0.001, 100.0, 0.1,
        0, 0, 2,
        -1, 4.0,
        1);
}

void SetUpJaw(glm::mat4 modelSoFar, RenderManager &rm) {
  rm.SetColor(96.0f/255.0f, 66.0f/255.0f, 53.0f/255.0f);
  Shape1(modelSoFar, rm, 0.075, 0.001, 100.0, 0.2,
        0, 0, 2,
        1, 4.0,
        0);
  Shape1(modelSoFar, rm, 0.075, 0.001, 100.0, 0.2,
        0, 0, 2,
        -1, 4.0,
        0);
}

void SetUpEar(glm::mat4 modelSoFar, RenderManager &rm, float t) {
  float ext = sin(t/5) * 0.05;
  rm.SetColor(118.0f/255.0f, 94.0f/255.0f, 79.0f/256.0f);
  modelSoFar = Shape1(modelSoFar, rm, 0.15, 0.15, 100.0, 0.175,
        60, 45, 2,
        -1, 4.0,
        1,
        ScaleMatrix(0.1, 1, 1));
  modelSoFar = Shape1(modelSoFar, rm, 0.15, 0.2, 100.0, 0.225,
        0, -15, 2,
        -1, 4.0,
        1,
        ScaleMatrix(0.1, 1, 1));
  modelSoFar = Shape1(modelSoFar, rm, 0.2, 0.2, 100.0, 0.175,
        0, -15, 2,
        -1, 4.0,
        1,
        ScaleMatrix(0.1, 1, 1));
  modelSoFar = Shape1(modelSoFar, rm, 0.2, 0.05, 100.0, 0.275 + ext,
        -30, -30, 2,
        -1, 2.0,
        1,
        ScaleMatrix(0.1, 1, 1));
}

void SetUpTongue(glm::mat4 modelSoFar, RenderManager &rm, float t) {
  float ext = sin(t/5) * 0.05;
  rm.SetColor(1, 78.0f/255.0f, 94.0f/255.0f);
  modelSoFar = modelSoFar * ScaleMatrix(1, 1, 0.1);
  Shape1(modelSoFar, rm, 0.075, 0.075, 100.0, 0.175,
        0, 0, 1,
        1, 4.0,
        1);
  Shape1(modelSoFar, rm, 0.075, 0.001, 100.0, 0.05+ext,
        0, 0, 1,
        -1, 4.0,
        1);
}

void SetUpMouth(glm::mat4 modelSoFar, RenderManager &rm, float t) {
  float yoffset = sin(t / 5)*0.0125;
  float zoffset = -sin(t / 5)*0.005;
  SetUpStache(modelSoFar, rm, /* sign= */ 1);
  modelSoFar = modelSoFar * TranslateMatrix(-0.25, 0, 0);
  SetUpStache(modelSoFar, rm, /* sign= */ -1);

  modelSoFar = modelSoFar * TranslateMatrix(0.125, -0.05, 0);
  SetUpJaw(modelSoFar, rm);

  modelSoFar = modelSoFar * TranslateMatrix(-0.02, -0.05, 0.1) * RotateMatrix(-20, 1, 0, 0);
  SetUpTongue(modelSoFar, rm, t);
}

glm::mat4 SetUpHead(glm::mat4 modelSoFar, RenderManager &rm, float t)
{
  // place center of head at X=3, Y=1, Z=0
  glm::mat4 translate = TranslateMatrix(1, 1, 0);
  //glm::mat4 translate = TranslateMatrix(0.5, 1, 3.5);
  SetUpHeadBase(modelSoFar*translate, rm, t);

  glm::mat4 mtranslate = TranslateMatrix(0.15, 0, 0);
  SetUpMouth(modelSoFar*translate*mtranslate, rm, t);

  glm::mat4 eartranslate = TranslateMatrix(0.0, 0.6, -0.45);
  SetUpEar(modelSoFar*translate*eartranslate*RotateMatrix(180, 0, 1, 0)*RotateMatrix(30, 0, 0, 1), rm, t*0.98);
  SetUpEar(modelSoFar*translate*eartranslate*RotateMatrix(30, 0, 0, 1), rm, t*1.01);
  
  glm::mat4 noseTranslate = TranslateMatrix(0, 0.15, -0.01);
  SetUpNose(modelSoFar*translate*noseTranslate, rm, t);
  
  glm::mat4 leftEyeTranslate = TranslateMatrix(-0.15, 0.25, -0.0);
  glm::mat4 rotateInFromLeft = RotateMatrix(15, 0, 1, 0);
  SetUpEyeball(modelSoFar*translate*leftEyeTranslate, rotateInFromLeft, rm, t);
  
  glm::mat4 rightEyeTranslate = TranslateMatrix(0.15, 0.25, -0.0);
  glm::mat4 rotateInFromRight = RotateMatrix(-15, 0, 1, 0);
  SetUpEyeball(modelSoFar*translate*rightEyeTranslate, rotateInFromRight, rm, t);

  return translate;
}

void SetUpRearLeg(glm::mat4 modelSoFar, RenderManager &rm, int sign) {
  rm.SetColor(245.0f/255.0f, 202.0f/255.0f, 175.0f/255.0f);
  glm::mat4 scale = ScaleMatrix(0.1, 0.1, 0.5);
  glm::mat4 shift = TranslateMatrix(0.0, 0.0, 0.0);
  glm::mat4 rotate = RotateMatrix(180, 0, 1, 0);
  rm.Render(RenderManager::CONE, modelSoFar*rotate*shift*scale);
}

void SetUpLeg(glm::mat4 modelSoFar, RenderManager &rm) {
  rm.SetColor(245.0f/255.0f, 202.0f/255.0f, 175.0f/255.0f);
  glm::mat4 scale = ScaleMatrix(0.1, 0.1, 0.5);
  glm::mat4 rotate = RotateMatrix(180, 0, 1, 0);
  rm.Render(RenderManager::CONE, modelSoFar*rotate*scale);
  glm::mat4 shift = TranslateMatrix(0, 0.0, 0);
  scale = ScaleMatrix(0.175, 0.1, 0.15);
  rm.Render(RenderManager::SPHERE, modelSoFar*shift*scale);
}

void SetUpLegs(glm::mat4 modelSoFar, RenderManager &rm, float bellyfactor) {
  float legfactor = bellyfactor / 2;
  modelSoFar = modelSoFar * TranslateMatrix(legfactor, legfactor, legfactor / 3);
  modelSoFar = modelSoFar * TranslateMatrix(0, -0.075, 0);
  // Left bottom
  glm::mat4 shift = TranslateMatrix(-0.3, 0.15, 0.3);
  glm::mat4 rotate = RotateMatrix(15, 0, 1, 0) * RotateMatrix(15, 1, 0, 0);
  SetUpLeg(modelSoFar * shift * rotate, rm);

  // Right bottom
  shift = TranslateMatrix(0.35, 0.15, 0.3);
  rotate = RotateMatrix(-15, 0, 1, 0) * RotateMatrix(15, 1, 0, 0);
  SetUpLeg(modelSoFar * shift * rotate, rm);

  // Left rear
  shift = TranslateMatrix(-0.4, -0.025, -0.2);
  rotate = RotateMatrix(10, 0, 1, 0) * RotateMatrix(20, 1, 0, 0);
  SetUpLeg(modelSoFar * shift * rotate, rm);

  // Right rear
  shift = TranslateMatrix(0.45, -0.025, -0.2);
  rotate = RotateMatrix(-10, 0, 1, 0) * RotateMatrix(20, 1, 0, 0);
  SetUpLeg(modelSoFar * shift * rotate, rm);
}

void SetUpBody(glm::mat4 modelSoFar, RenderManager &rm, float bellyfactor) {
  float maxgirth = 0.8 + bellyfactor;
  float yoffset = bellyfactor / 2;
  float neckgrowth = - bellyfactor / 4;
  glm::mat4 head_offset = modelSoFar;
  modelSoFar = modelSoFar * TranslateMatrix(0.0, -0.21+yoffset, -0.45);
  rm.SetColor(1, 232.0f/255.0f, 205.0f/255.0f);
  glm::mat4 neckscale = ScaleMatrix(0.505+neckgrowth, 0.15, 0.505+neckgrowth);
  rm.Render(RenderManager::SPHERE, modelSoFar*neckscale);
  modelSoFar = modelSoFar * TranslateMatrix(0.0, -0.1, 0);
  neckscale = ScaleMatrix(0.508+neckgrowth, 0.2, 0.508+neckgrowth);
  rm.Render(RenderManager::SPHERE, modelSoFar*neckscale);
  modelSoFar = modelSoFar * TranslateMatrix(0.0, -0.01, 0);
  modelSoFar = Shape1(modelSoFar, rm, 0.495, 0.575, 500.0, 0.3,
        0, 0, 1,
        -1);
  modelSoFar = Shape1(modelSoFar, rm, 0.575, maxgirth, 500.0, 0.6,
        0, 0, 1,
        -1, 2.0);
  glm::mat4 buttscale = ScaleMatrix(maxgirth, 0.15, maxgirth);
  rm.Render(RenderManager::SPHERE, modelSoFar*buttscale);

  // Legs
  glm::mat4 offset = head_offset * TranslateMatrix(0, -1.265, 0);
  SetUpLegs(offset, rm, bellyfactor);
}

void SetUpDog(RenderManager &rm, float t)
{
  float bellyfactor = sin(t/10) * 0.025;
  glm::mat4 identity(1.0f);
  
  glm::mat4 head_offset = SetUpHead(identity, rm, t);

  SetUpBody(head_offset, rm, bellyfactor);

  glm::mat4 shift = TranslateMatrix(-0.35, -2.25, 0.4);
  SetUpBone(head_offset*shift, rm, t);
}
    
const char *GetVertexShader()
{
   static char vertexShader[4096];
   strcpy(vertexShader, 
           "#version 400\n"
           "layout (location = 0) in vec3 vertex_position;\n"
           "layout (location = 1) in vec3 vertex_normal;\n"
           "uniform mat4 MVP;\n"
           "uniform mat4 model;\n"
           "uniform vec3 cameraloc;  // Camera position \n"
           "uniform vec3 lightdir;   // Lighting direction \n"
           "uniform vec4 lightcoeff; // Lighting coeff, Ka, Kd, Ks, alpha\n"
           "out float shading_amount;\n"
           "void main() {\n"
           "  gl_Position = MVP*vec4(vertex_position, 1.0);\n"
           // Calculate normal transform to prevent weird cylinder shadings
           "  vec3 normal = normalize(mat3(transpose(inverse(model))) * vertex_normal);\n"
           "  vec3 camera = normalize(cameraloc);\n"
           // Phong shading
           "  float Ka    = lightcoeff.x;\n"
           "  float Kd    = lightcoeff.y;\n"
           "  float Ks    = lightcoeff.z;\n"
           "  float alpha = lightcoeff.w;\n"
           #ifdef TWOSIDED
           "  float diffuse = max(dot(normal, lightdir), 0.0);\n"
           #else
           "  float diffuse = abs(dot(normal, lightdir));\n"
           #endif
           "  vec3 viewDir = normalize(cameraloc - vertex_position);\n"
           "  vec3 reflectDir = reflect(-lightdir, normal);\n"
           #ifdef TWOSIDED
           "  float specular = pow(abs(dot(viewDir, reflectDir)), alpha);\n"
           #else
           "  float specular = pow(max(dot(viewDir, reflectDir), 0.0), alpha);\n"
           #endif
           "  shading_amount = Ka + Kd * diffuse + Ks * specular;\n"
           "}\n"
         );
   return vertexShader;
}

const char *GetFragmentShader()
{
   static char fragmentShader[4096];
   strcpy(fragmentShader, 
           "#version 400\n"
           "in float shading_amount;\n"
           "uniform vec3 color;\n"
           "out vec4 frag_color;\n"
           "void main() {\n"
           "  frag_color = vec4(color * clamp(shading_amount, 0.0, 1.0), 1.0);\n"
           "}\n"
         );
   return fragmentShader;
}
