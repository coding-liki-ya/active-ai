#pragma once
#include <vector>
#include <glm/vec3.hpp>
#include <string>

enum class NeuronType { Input, Excitatory, Inhibitory, Output };

struct Neuron {
    glm::vec3 position;
    float energy = 0.f;
    float threshold = 1.f;
    NeuronType type = NeuronType::Excitatory;
};

struct Connection {
    int from = 0;
    int to = 0;
    float weight = 1.f;
};

class NeuralNet {
public:
    std::vector<Neuron> neurons;
    std::vector<Connection> connections;

    void randomize(int inputs, int internals, int outputs,
                   int minLinks, int maxLinks);
    bool load(const std::string& path);
    bool save(const std::string& path) const;
};
