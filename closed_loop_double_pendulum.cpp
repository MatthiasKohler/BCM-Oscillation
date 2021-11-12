#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <Eigen/Dense>

#include "neural_network.h"
#include "double_pendulum.h"
#include "closed_loop_simulation.h"


int main()
{
    struct rabit::neural_network_parameters net_params;
    net_params.tau_mem = 5.0;
    net_params.n_neurons = 8;
    net_params.n_inputs = 8 + 8;
    net_params.learning_rate = 0.0001;
    
    double_pendulum p;

    closed_loop_simulation sim(net_params, p);
    sim.simulate();


    return 0;
}
