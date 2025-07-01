#include "neural_net.h"
#include <fstream>
#include <random>
#include <glm/geometric.hpp>

void NeuralNet::randomize(int inputs, int internals, int outputs,
                          int minLinks, int maxLinks) {
    neurons.clear();
    connections.clear();
    std::default_random_engine eng{std::random_device{}()};
    std::uniform_real_distribution<float> distPos(-1.f, 1.f);
    std::uniform_int_distribution<int> distLinks(minLinks, maxLinks);

    auto addNeuron = [&](NeuronType t){
        Neuron n; n.position = {distPos(eng), distPos(eng), distPos(eng)}; n.type = t; neurons.push_back(n); };
    for(int i=0;i<inputs;++i) addNeuron(NeuronType::Input);
    for(int i=0;i<internals;++i) addNeuron(NeuronType::Excitatory);
    for(int i=0;i<outputs;++i) addNeuron(NeuronType::Output);

    std::uniform_int_distribution<int> distNeuron(0, neurons.size()-1);
    for(size_t i=0;i<neurons.size();++i){
        int links = distLinks(eng);
        for(int l=0;l<links;++l){
            Connection c; c.from = i; c.to = distNeuron(eng); c.weight = 0.5f + distPos(eng); connections.push_back(c); }
    }
}

bool NeuralNet::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if(!f) return false;
    size_t n = neurons.size();
    size_t c = connections.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(&c), sizeof(c));
    f.write(reinterpret_cast<const char*>(neurons.data()), n*sizeof(Neuron));
    f.write(reinterpret_cast<const char*>(connections.data()), c*sizeof(Connection));
    return true;
}

bool NeuralNet::load(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    size_t n=0,c=0; f.read(reinterpret_cast<char*>(&n), sizeof(n));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    neurons.resize(n); connections.resize(c);
    f.read(reinterpret_cast<char*>(neurons.data()), n*sizeof(Neuron));
    f.read(reinterpret_cast<char*>(connections.data()), c*sizeof(Connection));
    return true;
}
