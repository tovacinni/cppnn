#include <random>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <math.h>
#include <iostream>
#include "network.h"

using namespace std;
using namespace Eigen;

Network::Network(vector<int> s){
  sizes = s;
  cost = 0;
  training = 0;
  num_layers = sizes.size();

  // allocate space
  biases.resize(num_layers-1);
  weights.resize(num_layers-1);

  // setup gaussian generators
  default_random_engine generator;
  normal_distribution<double> distribution(0.0, 1.0);
  
  // init w/ random gaussian vals
  for (int i = 1; i < num_layers; ++i){
    VectorXd v(sizes[i]);
    for (int a = 0; a < sizes[i]; ++a){
      v(a) = distribution(generator);
    }
    biases[i-1] = v;
    MatrixXd m(sizes[i], sizes[i-1]);
    for (int j = 0; j < sizes[i]; ++j){
      for (int k = 0; k < sizes[i-1]; ++k){
        m(j,k) = distribution(generator);
      }
    }
    weights[i-1] = m;
  }
}

double Network::sigmoid(double z){
  return 1 / (1 + exp(z));
}

double Network::sigmoid_prime(double z){
  // derivative of sigmoid
  return sigmoid(z) * (1 - sigmoid(z));
}

VectorXd Network::feedforward(VectorXd a){
  // this is just fancy matrix multiplication
  for (int i = 0; i < num_layers-1; ++i){
    a = (weights[i]*a) + biases[i];
    a = a.unaryExpr(ptr_fun(&Network::sigmoid));
  }
  return a;
}

void Network::sgd(vector<VectorXd> &training_input,
                  vector<VectorXd> &training_output,
                  int epochs, int mini_batch_size, double eta, bool log){
  int training_length = training_input.size();
  for (int i = 0; i < epochs; ++i){
    // make a vector of shuffled indexes
    vector<int> indexes;
    indexes.reserve(training_length);
    for (int j = 0; j < training_length; ++j){
      indexes.push_back(j);
    }
    random_shuffle(indexes.begin(), indexes.end());

    // make batches of inputs/outputs and update weights/biases
    for (int j = 0; j < training_length; j += mini_batch_size){
      vector<VectorXd> mini_training_input;
      vector<VectorXd> mini_training_output;
      for (int k = j; k < j + mini_batch_size; k++){
        mini_training_input.push_back(
          training_input[indexes[k%training_length]]);
        mini_training_output.push_back(
          training_output[indexes[k%training_length]]);
      }
      update(mini_training_input, mini_training_output, eta);
    }
    // if the log option is passed in, log partial results
    if (log){
      cout << "Epochs: " << i << endl;
      print_results(training_input, training_output);
      cout << "Cost: " << cost / (2 * training) << endl;
    }
  }
}

void Network::update(vector<VectorXd> &mini_input,
                     vector<VectorXd> &mini_output,
                     double eta){
  // create vectors of vectors/matrices
  vector<VectorXd> nabla_b;
  vector<MatrixXd> nabla_w;
  nabla_b.reserve(biases.size());
  nabla_w.reserve(weights.size());
  for (auto b : biases){
    VectorXd t = VectorXd::Zero(b.rows());
    nabla_b.push_back(t);
  }
  for (auto w : weights){
    MatrixXd t = MatrixXd::Zero(w.rows(), w.cols());
    nabla_w.push_back(t);
  }
  // iterate through input/output
  for (int i = 0; i < mini_input.size(); ++i){
    // backpropagate
    backprop(mini_input[i], mini_output[i], nabla_b, nabla_w);
  }
  for (int j = 0; j < biases.size(); ++j){
    weights[j] -= (eta/mini_input.size()) * nabla_w[j];
    biases[j] -= (eta/mini_input.size()) * nabla_b[j];
  }
}

double sqr(double x){
  return x * x;
}

void Network::backprop(VectorXd in, VectorXd out, 
                       vector<VectorXd> &nabla_b, vector<MatrixXd> &nabla_w){
  VectorXd activation = in;
  vector<VectorXd> activations{activation};
  vector<VectorXd> zs;
  zs.reserve(num_layers-1);

  // this does the same thing as feedforward, but saves the list of
  //   activations and zs (activation w/o sigmoid) for each layer for
  //   backpropagation
  for (int i = 0; i < num_layers-1; ++i){
    VectorXd z = (weights[i] * activation) + biases[i];
    zs.push_back(z);
    activation = z.unaryExpr(ptr_fun(&Network::sigmoid));
    activations.push_back(activation);
  }

  // log the training and cost
  VectorXd c = out - activations.back();
  cost += c.unaryExpr(ptr_fun(&sqr)).sum();
  training += 1;

  // now backpropagate, assuming quadratic cost func
  VectorXd dt = zs.back();
  VectorXd delta = (activations.back() - out)
    .cwiseProduct(dt.unaryExpr(ptr_fun(&Network::sigmoid_prime)));
  nabla_b[nabla_b.size()-1] = delta;
  nabla_w[nabla_w.size()-1] = delta * 
                              activations[activations.size()-2].transpose();

  // step backwards
  for (int i = 2; i < num_layers; ++i){
    VectorXd z = zs[zs.size()-i];
    VectorXd sp = z.unaryExpr(ptr_fun(&Network::sigmoid_prime));
    delta = (weights[weights.size()-i+1].transpose() * delta).cwiseProduct(sp);
    nabla_b[nabla_b.size()-i] += delta;
    nabla_w[nabla_w.size()-i] += 
      delta * activations[activations.size()-i-1].transpose();
  }
}

int maxCoeff(VectorXd v){
  // gets index w/ max element
  double max = 0;
  int index = 0;
  for (int i = 0; i < v.rows(); i++){
    if (v[i] > max){
      max = v[i];
      index = i;
    }
  }
  return index;
}

int Network::evaluate(vector<VectorXd> &in, vector<VectorXd> &out){
  // evaluates the network
  int correct = 0;
  for (int i = 0; i < in.size(); ++i){
    VectorXd o = feedforward(in[i]);
    if (maxCoeff(o) == maxCoeff(out[i])){
      correct += 1;
    }
  }
  return correct;
}

void Network::print_results(vector<VectorXd> &in, vector<VectorXd> &out){
  // print out results
  cout << "Result: " << evaluate(in, out) << "/" << in.size() << endl;
}
