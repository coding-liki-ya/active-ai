#include "neural_net.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <fstream>

static std::string loadFile(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static GLuint compileCompute(){
    std::string src = loadFile("shaders/neuron.comp");
    const char* csrc = src.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader,1,&csrc,nullptr); glCompileShader(shader);
    GLuint prog = glCreateProgram(); glAttachShader(prog, shader); glLinkProgram(prog);
    glDeleteShader(shader); return prog;
}

static GLuint compileRender(){
    std::string vsrc = loadFile("shaders/render.vert");
    std::string fsrc = loadFile("shaders/render.frag");
    const char* vs = vsrc.c_str();
    const char* fs = fsrc.c_str();
    GLuint vert = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vert,1,&vs,nullptr); glCompileShader(vert);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(frag,1,&fs,nullptr); glCompileShader(frag);
    GLuint prog = glCreateProgram(); glAttachShader(prog,vert); glAttachShader(prog,frag); glLinkProgram(prog);
    glDeleteShader(vert); glDeleteShader(frag); return prog;
}

int main(int argc, char** argv){
    if(!glfwInit()){ std::cerr << "GLFW init failed\n"; return 1; }
    GLFWwindow* win = glfwCreateWindow(800,600,"net",nullptr,nullptr);
    glfwMakeContextCurrent(win);
    if(glewInit()!=GLEW_OK){ std::cerr << "GLEW init failed\n"; return 1; }

    NeuralNet net;
    if(argc>1){ net.load(argv[1]); } else {
        net.randomize(4,10,2,1,3);
    }
    GLuint computeProg = compileCompute();
    GLuint renderProg = compileRender();

    GLuint bufEnergy,bufThres,bufType,bufFrom,bufTo,bufWeight;
    glGenBuffers(1,&bufEnergy); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufEnergy);
    std::vector<float> energies(net.neurons.size());
    for(size_t i=0;i<energies.size();++i) energies[i]=net.neurons[i].energy;
    glBufferData(GL_SHADER_STORAGE_BUFFER, energies.size()*sizeof(float), energies.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,bufEnergy);

    glGenBuffers(1,&bufThres); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufThres);
    std::vector<float> thrs(net.neurons.size());
    for(size_t i=0;i<thrs.size();++i) thrs[i]=net.neurons[i].threshold;
    glBufferData(GL_SHADER_STORAGE_BUFFER, thrs.size()*sizeof(float), thrs.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,bufThres);

    std::vector<int> types(net.neurons.size());
    for(size_t i=0;i<types.size();++i) types[i]=(int)net.neurons[i].type;
    glGenBuffers(1,&bufType); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufType);
    glBufferData(GL_SHADER_STORAGE_BUFFER, types.size()*sizeof(int), types.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,bufType);

    std::vector<int> from(net.connections.size()), to(net.connections.size());
    std::vector<float> weight(net.connections.size());
    for(size_t i=0;i<net.connections.size();++i){
        from[i]=net.connections[i].from; to[i]=net.connections[i].to; weight[i]=net.connections[i].weight;
    }
    glGenBuffers(1,&bufFrom); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufFrom);
    glBufferData(GL_SHADER_STORAGE_BUFFER, from.size()*sizeof(int), from.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,bufFrom);

    glGenBuffers(1,&bufTo); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufTo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, to.size()*sizeof(int), to.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,bufTo);

    glGenBuffers(1,&bufWeight); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufWeight);
    glBufferData(GL_SHADER_STORAGE_BUFFER, weight.size()*sizeof(float), weight.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,bufWeight);

    glUseProgram(computeProg);
    GLint ccLoc = glGetUniformLocation(computeProg,"connectionCount");
    glUniform1i(ccLoc, (int)net.connections.size());

    // setup geometry
    GLuint vao,posBuf;
    glGenVertexArrays(1,&vao);
    glBindVertexArray(vao);
    glGenBuffers(1,&posBuf); glBindBuffer(GL_ARRAY_BUFFER,posBuf);
    std::vector<glm::vec3> positions(net.neurons.size());
    for(size_t i=0;i<positions.size();++i) positions[i]=net.neurons[i].position;
    glBufferData(GL_ARRAY_BUFFER, positions.size()*sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, bufEnergy);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,0,nullptr);
    glEnableVertexAttribArray(1);

    GLuint connBuf;
    glGenBuffers(1,&connBuf); glBindBuffer(GL_ARRAY_BUFFER, connBuf);
    std::vector<glm::vec3> connVerts(net.connections.size()*2);
    for(size_t i=0;i<net.connections.size();++i){
        connVerts[i*2] = net.neurons[net.connections[i].from].position;
        connVerts[i*2+1] = net.neurons[net.connections[i].to].position;
    }
    glBufferData(GL_ARRAY_BUFFER, connVerts.size()*sizeof(glm::vec3), connVerts.data(), GL_STATIC_DRAW);

    glUseProgram(renderProg);
    GLint mvpLoc = glGetUniformLocation(renderProg,"mvp");
    glm::mat4 view = glm::lookAt(glm::vec3(0,0,5), glm::vec3(0,0,0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), 800.f/600.f, 0.1f, 100.f);
    glm::mat4 mvp = proj * view;

    while(!glfwWindowShouldClose(win)){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(computeProg);
        glUniform1i(ccLoc, (int)net.connections.size());
        glDispatchCompute((GLuint)net.connections.size()/64+1,1,1);

        glUseProgram(renderProg);
        glUniformMatrix4fv(mvpLoc,1,GL_FALSE,&mvp[0][0]);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS,0,(GLsizei)positions.size());

        glBindBuffer(GL_ARRAY_BUFFER, connBuf);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,nullptr);
        glDisableVertexAttribArray(1);
        glDrawArrays(GL_LINES,0,(GLsizei)connVerts.size());
        glEnableVertexAttribArray(1);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    if(argc>2) net.save(argv[2]);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
