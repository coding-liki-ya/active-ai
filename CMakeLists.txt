cmake_minimum_required(VERSION 3.10)
project(active_ai)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
add_executable(active_ai
    src/main.cpp
    src/neural_net.cpp
)
target_include_directories(active_ai PRIVATE ${GLM_INCLUDE_DIRS} src)
target_link_libraries(active_ai PRIVATE glfw OpenGL::GL GLEW::GLEW)
