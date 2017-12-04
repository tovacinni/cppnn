#ifndef NET_H
#define NET_H
#include <vector>
#include <tuple>
#include <Eigen/Dense>

class Network {

private:

  int num_layers;
  double cost;
  int training;
  std::vector<int> sizes;
  std::vector<Eigen::VectorXd> biases;
  std::vector<Eigen::MatrixXd> weights;

public:

  Network(std::vector<int> s);
  Eigen::VectorXd feedforward(Eigen::VectorXd a);
  void sgd(std::vector<Eigen::VectorXd> &training_input,
           std::vector<Eigen::VectorXd> &training_output,
           int epochs, int mini_batch_size, double eta, bool log);
  static double sigmoid(double z);
  static double sigmoid_prime(double z);
  void update(std::vector<Eigen::VectorXd> &mini_input,
              std::vector<Eigen::VectorXd> &mini_output,
              double eta);
  void backprop(Eigen::VectorXd in, Eigen::VectorXd out,
                std::vector<Eigen::VectorXd> &nabla_b,
                std::vector<Eigen::MatrixXd> &nabla_w);
  int evaluate(std::vector<Eigen::VectorXd> &in,
               std::vector<Eigen::VectorXd> &out);
  void print_results(std::vector<Eigen::VectorXd> &in, 
                     std::vector<Eigen::VectorXd> &out);
};

#endif
