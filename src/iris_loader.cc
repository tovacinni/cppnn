#include <vector>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "iris_loader.h"
#include "network.h"

void Iris_Loader::load_csv(std::string filename){
  std::ifstream f(filename);
  std::string line;
  while (getline(f, line)){
    std::stringstream s(line);
    std::string w;
    Eigen::VectorXd input(4);
    Eigen::VectorXd output(3);
    int i = 0;
    while (getline(s, w, ',')){
      if (i < 4){
        input(i) = std::stod(w);
        ++i;
      } else {
        if (w == "Iris-setosa"){
          output << 1, 0, 0;
        } else if (w == "Iris-versicolor"){
          output << 0, 1, 0;
        } else if (w == "Iris-virginica"){
          output << 0, 0, 1;
        }
      }
    }
    inputs.push_back(input);
    outputs.push_back(output);
  }
}

std::vector<Eigen::VectorXd> Iris_Loader::get_inputs(){
  return inputs;
}

std::vector<Eigen::VectorXd> Iris_Loader::get_outputs(){
  return outputs;
}
