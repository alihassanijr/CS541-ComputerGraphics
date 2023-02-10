/*

Ali Hassani

Project 2-A

CS 441/541

*/
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "proj2A_data.h"
#define NUM_INDICES 44106
#define NUM_POINTS 77535
#define NUM_COLORS 25845

void _print_shader_info_log(GLuint shader_index) {
  int max_len = 2048;
  int actual_len = 0;
  char shader_log[2048];
  glGetShaderInfoLog(shader_index, max_len, &actual_len, shader_log);
  printf("shader info log for GL index %u:\n%s\n", shader_index, shader_log);
}

GLuint SetupDataForRendering()
{
  printf("Getting data\n");

  // Add data to VBOs and VAO for phase 3 here

  GLuint points_vbo = 0; // Points buffer object
  glGenBuffers(1, &points_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
  glBufferData(GL_ARRAY_BUFFER, NUM_POINTS * sizeof(float), tri_points, GL_STATIC_DRAW);

  GLuint data_vbo = 0; // Data buffer object
  glGenBuffers(1, &data_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, data_vbo);
  glBufferData(GL_ARRAY_BUFFER, NUM_COLORS * sizeof(float), tri_data, GL_STATIC_DRAW);

  GLuint normals_vbo = 0; // Normals buffer object
  glGenBuffers(1, &normals_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
  glBufferData(GL_ARRAY_BUFFER, NUM_POINTS * sizeof(float), tri_normals, GL_STATIC_DRAW);

  GLuint index_vbo; // Index buffer object
  glGenBuffers( 1, &index_vbo);
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, index_vbo);
  glBufferData( GL_ELEMENT_ARRAY_BUFFER, NUM_INDICES * sizeof(GLuint), tri_indices, GL_STATIC_DRAW );

  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Points
  glBindBuffer(GL_ARRAY_BUFFER, data_vbo);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL); // Data
  glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Normals
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, index_vbo );

  glEnableVertexAttribArray(0); // Points
  glEnableVertexAttribArray(1); // Data
  glEnableVertexAttribArray(2); // Normals

  return vao;
}

const char *proj2AVertexShader =
  "#version 400\n"
  "layout (location = 0) in vec3 vertex_position;\n"
  "layout (location = 1) in float vertex_data;\n"
  "layout (location = 2) in vec3 vertex_normal;\n"
  "uniform mat4 MVP;\n"
  "uniform vec3 cameraloc;  // Camera position \n"
  "uniform vec3 lightdir;   // Lighting direction \n"
  "uniform vec4 lightcoeff; // Lighting coeff, Ka, Kd, Ks, alpha\n"
  "out float data;\n"
  "out float shading_amount;\n"
  "void main() {\n"
  "  vec4 position = vec4(vertex_position, 1.0);\n"
  "  gl_Position = MVP*position;\n"
  "  data = vertex_data;\n"
  // Assign shading_amount a value by calculating phong shading
  // camaraloc  : is the location of the camera
  // lightdir   : is the direction of the light
  // lightcoeff : represents a vec4(Ka, Kd, Ks, alpha) from LightingParams of 1F
  "  float Ka    = lightcoeff.x;\n"
  "  float Kd    = lightcoeff.y;\n"
  "  float Ks    = lightcoeff.z;\n"
  "  float alpha = lightcoeff.w;\n"
  "  vec3 viewDirection = normalize(cameraloc - vertex_position);\n"
  "  float LN = dot(normalize(lightdir), normalize(vertex_normal));\n"
  "  float diffuse = max(0.0, LN);\n" // view direction does not affect this

  "  vec3 R = (normalize(vertex_normal) * (2*LN)) - normalize(lightdir);\n"
  "  float RV = max(0.0, dot(normalize(R), viewDirection));\n"

  "  float specular = pow(RV, alpha);\n" // Didn't need abs because RV is already >= 0
  "  shading_amount = Ka + Kd * diffuse + Ks * specular;\n"
  "}\n";

const char *proj2AFragmentShader =
  "#version 400\n"
  "in float data;\n"
  "in float shading_amount;\n"
  "out vec4 frag_color;\n"
  "void main() {\n"
  "  vec3 c1 = mix(vec3(0.25, 0.25,  1.0), vec3( 1.0,  1.0,  1.0), (data - 1.0) / (3.5)) * float(data >= 1.0 && data <= 4.5);\n"
  "  vec3 c2 = mix(vec3( 1.0,  1.0,  1.0), vec3( 1.0, 0.25, 0.25), (data - 4.5) / (1.5)) * float(data >= 4.5 && data <= 6.0);\n"
  // Update frag_color by mixing the shading factor
  "  frag_color = vec4((c1 + c2) * shading_amount, 1.0);\n"
  "}\n";

int main() {
  // start GL context and O/S window using the GLFW helper library
  if (!glfwInit()) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(700, 700, "CIS 541", NULL, NULL);
  if (!window) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    return 1;
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
  glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"

  GLuint vao = 0;
  vao = SetupDataForRendering();
  const char* vertex_shader = proj2AVertexShader;
  const char* fragment_shader = proj2AFragmentShader;

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

  GLuint shader_programme = glCreateProgram();
  glAttachShader(shader_programme, fs);
  glAttachShader(shader_programme, vs);
  glLinkProgram(shader_programme);

  glUseProgram(shader_programme);

  // Code block for camera transforms
  // Projection matrix : 30° Field of View
  // display size  : 1000x1000
  // display range : 5 unit <-> 200 units
  glm::mat4 Projection = glm::perspective(
      glm::radians(30.0f), (float)1000 / (float)1000,  5.0f, 200.0f);
  glm::vec3 camera(0, 40, 40);
  glm::vec3 origin(0, 0, 0);
  glm::vec3 up(0, 1, 0);
  // Camera matrix
  glm::mat4 View = glm::lookAt(
    camera, // Camera in world space
    origin, // looks at the origin
    up      // and the head is up
  );
  // Model matrix : an identity matrix (model will be at the origin)
  glm::mat4 Model = glm::mat4(1.0f);
  // Our ModelViewProjection : multiplication of our 3 matrices
  glm::mat4 mvp = Projection * View * Model;

  // Get a handle for our "MVP" uniform
  // Only during the initialisation
  GLuint mvploc = glGetUniformLocation(shader_programme, "MVP");
  // Send our transformation to the currently bound shader, in the "MVP" uniform
  // This is done in the main loop since each model will have a different MVP matrix
  // (At least for the M part)
  glUniformMatrix4fv(mvploc, 1, GL_FALSE, &mvp[0][0]);

 // Code block for shading parameters
  GLuint camloc = glGetUniformLocation(shader_programme, "cameraloc");
  glUniform3fv(camloc, 1, &camera[0]);
  glm::vec3 lightdir = glm::normalize(camera - origin);   // Direction of light
  GLuint ldirloc = glGetUniformLocation(shader_programme, "lightdir");
  glUniform3fv(ldirloc, 1, &lightdir[0]);
  glm::vec4 lightcoeff(0.3, 0.7, 2.8, 50.5); // Lighting coeff, Ka, Kd, Ks, alpha
  GLuint lcoeloc = glGetUniformLocation(shader_programme, "lightcoeff");
  glUniform4fv(lcoeloc, 1, &lightcoeff[0]);

  while (!glfwWindowShouldClose(window)) {
    // wipe the drawing surface clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(vao);
    // Draw triangles

    // Add correct number of indices
    glDrawElements( GL_TRIANGLES, NUM_INDICES, GL_UNSIGNED_INT, NULL );

    // update other events like input handling
    glfwPollEvents();
    // put the stuff we've been drawing onto the display
    glfwSwapBuffers(window);
  }

  // close GL context and any other GLFW resources
  glfwTerminate();
  return 0;
}

