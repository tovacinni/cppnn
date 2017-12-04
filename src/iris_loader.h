#ifndef IRIS_H
#define IRIS_H
#include <vector>
#include <Eigen/Dense>
#include <string>

// this should probably inherit a generic loader class but oh well
class Iris_Loader {

private:

  std::vector<Eigen::VectorXd> inputs;
  std::vector<Eigen::VectorXd> outputs;

public:

  void load_csv(std::string filename);
  std::vector<Eigen::VectorXd> get_inputs();
  std::vector<Eigen::VectorXd> get_outputs();
};

#endif
