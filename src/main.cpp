#include "neural_net.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <random>

struct Camera {
    glm::vec3 pos{0.f,0.f,5.f};
    float yaw{-90.f};
    float pitch{0.f};
    glm::vec3 front{0.f,0.f,-1.f};
    glm::vec3 up{0.f,1.f,0.f};
    void update(){
        glm::vec3 dir;
        dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        dir.y = sin(glm::radians(pitch));
        dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(dir);
    }
};

static Camera cam;
static bool firstMouse = true;
static double lastX = 400.0, lastY = 300.0;

static void mouseCallback(GLFWwindow*, double x, double y){
    if(firstMouse){ lastX = x; lastY = y; firstMouse = false; }
    float dx = static_cast<float>(x - lastX);
    float dy = static_cast<float>(lastY - y);
    lastX = x; lastY = y;
    const float sens = 0.1f;
    cam.yaw += dx * sens;
    cam.pitch += dy * sens;
    if(cam.pitch>89.f) cam.pitch=89.f;
    if(cam.pitch<-89.f) cam.pitch=-89.f;
    cam.update();
}

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
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(win, mouseCallback);
    if(glewInit()!=GLEW_OK){ std::cerr << "GLEW init failed\n"; return 1; }

    NeuralNet net;
    if(argc>1){ net.load(argv[1]); } else {
        net.randomize(4,10,2,1,3);
    }
    GLuint computeProg = compileCompute();
    GLuint renderProg = compileRender();
    double lastSpawn = glfwGetTime();

    GLuint bufEnergy,bufThres,bufType,bufFrom,bufTo,bufWeight;
    glGenBuffers(1,&bufEnergy); glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufEnergy);
    std::vector<float> energies(net.neurons.size());
    for(size_t i=0;i<energies.size();++i) energies[i]=net.neurons[i].energy;
    std::vector<glm::vec3> velocities(net.neurons.size(), glm::vec3(0));
    std::vector<bool> connected(net.neurons.size(), true);
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

    GLuint connBuf, connEnergyBuf;
    glGenBuffers(1,&connBuf); glBindBuffer(GL_ARRAY_BUFFER, connBuf);
    std::vector<glm::vec3> connVerts(net.connections.size()*2);
    std::vector<float> connEner(connVerts.size());
    for(size_t i=0;i<net.connections.size();++i){
        connVerts[i*2] = net.neurons[net.connections[i].from].position;
        connVerts[i*2+1] = net.neurons[net.connections[i].to].position;
        connEner[i*2]   = energies[net.connections[i].from];
        connEner[i*2+1] = energies[net.connections[i].to];
    }
    glBufferData(GL_ARRAY_BUFFER, connVerts.size()*sizeof(glm::vec3), connVerts.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1,&connEnergyBuf);
    glBindBuffer(GL_ARRAY_BUFFER, connEnergyBuf);
    glBufferData(GL_ARRAY_BUFFER, connEner.size()*sizeof(float), connEner.data(), GL_DYNAMIC_DRAW);

    glUseProgram(renderProg);
    GLint mvpLoc = glGetUniformLocation(renderProg,"mvp");
    GLint colorLoc = glGetUniformLocation(renderProg,"baseColor");
    GLint alphaLoc = glGetUniformLocation(renderProg,"alpha");
    GLint sizeLoc = glGetUniformLocation(renderProg,"pointSize");
    glm::mat4 view = glm::lookAt(glm::vec3(0,0,5), glm::vec3(0,0,0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), 800.f/600.f, 0.1f, 100.f);
    glm::mat4 mvp = proj * view;
    double lastTime = glfwGetTime();

    std::default_random_engine rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1.f,1.f);

    while(!glfwWindowShouldClose(win)){
        double t = glfwGetTime();
        float dt = static_cast<float>(t - lastTime);
        lastTime = t;

        float speed = 2.0f * dt;
        if(glfwGetKey(win,GLFW_KEY_W)==GLFW_PRESS) cam.pos += cam.front*speed;
        if(glfwGetKey(win,GLFW_KEY_S)==GLFW_PRESS) cam.pos -= cam.front*speed;
        glm::vec3 right = glm::normalize(glm::cross(cam.front, cam.up));
        if(glfwGetKey(win,GLFW_KEY_A)==GLFW_PRESS) cam.pos -= right*speed;
        if(glfwGetKey(win,GLFW_KEY_D)==GLFW_PRESS) cam.pos += right*speed;

        view = glm::lookAt(cam.pos, cam.pos+cam.front, cam.up);
        mvp = proj * view;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(computeProg);
        glUniform1i(ccLoc, (int)net.connections.size());
        glDispatchCompute((GLuint)net.connections.size()/64+1,1,1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufEnergy);
        float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        for(size_t i=0;i<energies.size();++i) energies[i]=ptr[i];
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glm::vec3 center(0.f); float sumE=0.f;
        for(size_t i=0;i<energies.size();++i){ center += energies[i]*positions[i]; sumE += energies[i]; }
        if(sumE>0.f) center /= sumE;

        for(size_t i=0;i<positions.size();++i){
            glm::vec3 dir = glm::normalize(center - positions[i]);
            velocities[i] += dir * dt;
            positions[i] += velocities[i]*dt;
            velocities[i] *= 0.98f;
            if(!connected[i] && glm::length(center - positions[i]) < 0.5f){
                for(size_t j=0;j<net.neurons.size();++j){
                    if(energies[j] >= thrs[j]){
                        Connection c; c.from=j; c.to=i; c.weight=0.5f+dist(rng);
                        net.connections.push_back(c);
                        from.push_back(c.from); to.push_back(c.to); weight.push_back(c.weight);
                    }
                }
                connected[i]=true;
                connVerts.resize(net.connections.size()*2);
                connEner.resize(connVerts.size());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufFrom);
                glBufferData(GL_SHADER_STORAGE_BUFFER, from.size()*sizeof(int), from.data(), GL_STATIC_DRAW);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufTo);
                glBufferData(GL_SHADER_STORAGE_BUFFER, to.size()*sizeof(int), to.data(), GL_STATIC_DRAW);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufWeight);
                glBufferData(GL_SHADER_STORAGE_BUFFER, weight.size()*sizeof(float), weight.data(), GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, connBuf);
                glBufferData(GL_ARRAY_BUFFER, connVerts.size()*sizeof(glm::vec3), connVerts.data(), GL_DYNAMIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, connEnergyBuf);
                glBufferData(GL_ARRAY_BUFFER, connEner.size()*sizeof(float), connEner.data(), GL_DYNAMIC_DRAW);
            }
        }

        if(t - lastSpawn > 5.0){
            Neuron n; n.position = glm::vec3(dist(rng)*3.f, dist(rng)*3.f, dist(rng)*3.f);
            n.type = NeuronType::Excitatory; n.energy = 0.f; n.threshold = 1.f;
            net.neurons.push_back(n);
            energies.push_back(0.f);
            thrs.push_back(1.f);
            types.push_back((int)n.type);
            velocities.push_back(glm::vec3(dist(rng),dist(rng),dist(rng)));
            connected.push_back(false);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufEnergy);
            glBufferData(GL_SHADER_STORAGE_BUFFER, energies.size()*sizeof(float), energies.data(), GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,bufEnergy);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufThres);
            glBufferData(GL_SHADER_STORAGE_BUFFER, thrs.size()*sizeof(float), thrs.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufType);
            glBufferData(GL_SHADER_STORAGE_BUFFER, types.size()*sizeof(int), types.data(), GL_STATIC_DRAW);
            positions.push_back(n.position);
            glBindBuffer(GL_ARRAY_BUFFER, posBuf);
            glBufferData(GL_ARRAY_BUFFER, positions.size()*sizeof(glm::vec3), positions.data(), GL_DYNAMIC_DRAW);
            lastSpawn = t;
        }

        for(size_t i=0;i<net.connections.size();++i){
            connVerts[i*2] = positions[net.connections[i].from];
            connVerts[i*2+1] = positions[net.connections[i].to];
            connEner[i*2] = energies[net.connections[i].from];
            connEner[i*2+1] = energies[net.connections[i].to];
        }

        glBindBuffer(GL_ARRAY_BUFFER, posBuf);
        glBufferSubData(GL_ARRAY_BUFFER,0,positions.size()*sizeof(glm::vec3),positions.data());
        glBindBuffer(GL_ARRAY_BUFFER, bufEnergy);
        glBufferSubData(GL_ARRAY_BUFFER,0,energies.size()*sizeof(float),energies.data());
        glBindBuffer(GL_ARRAY_BUFFER, connBuf);
        glBufferSubData(GL_ARRAY_BUFFER,0,connVerts.size()*sizeof(glm::vec3),connVerts.data());
        glBindBuffer(GL_ARRAY_BUFFER, connEnergyBuf);
        glBufferSubData(GL_ARRAY_BUFFER,0,connEner.size()*sizeof(float),connEner.data());

        glUseProgram(renderProg);
        glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(mvp));

        glUniform3f(colorLoc,1.f,0.f,0.f); glUniform1f(alphaLoc,0.5f); glUniform1f(sizeLoc,8.f);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,posBuf);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, bufEnergy);
        glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,0,nullptr);
        glDrawArrays(GL_POINTS,0,(GLsizei)positions.size());

        glUniform3f(colorLoc,0.f,1.f,0.f); glUniform1f(alphaLoc,1.0f); glUniform1f(sizeLoc,1.f);
        glBindBuffer(GL_ARRAY_BUFFER, connBuf);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, connEnergyBuf);
        glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,0,nullptr);
        glDrawArrays(GL_LINES,0,(GLsizei)connVerts.size());

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    if(argc>2) net.save(argv[2]);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
