#include <iostream>
#include <Eigen/Dense>
#include "network.h"
#include "iris_loader.h"

using namespace std;
using namespace Eigen;

int main(){
	vector<int> s{4, 144, 3};
	Network n(s);
  Iris_Loader i;
  Iris_Loader j;
  i.load_csv("../data/iris.data");
  vector<VectorXd> inputs = i.get_inputs();
  vector<VectorXd> outputs = i.get_outputs();
  // test/validation split should be automated but isn't :(
  j.load_csv("../data/iris.validation");
  vector<VectorXd> v_in = j.get_inputs();
  vector<VectorXd> v_out = j.get_outputs(); 
  n.sgd(inputs, outputs, 500, 30, 0.5, true);
  cout << "Results on Training Set:" << endl;
  n.print_results(inputs, outputs);
  cout << "Results on Validation Set:" << endl;
  n.print_results(v_in, v_out);
	return 0;
}