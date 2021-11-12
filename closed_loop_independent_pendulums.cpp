#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <Eigen/Dense>

#include "neural_network.h"
#include "independent_pendulums.h"
#include "closed_loop_simulation.h"

int main()
{
    struct rabit::neural_network_parameters net_params;
    net_params.tau_mem = 5.0;
    net_params.n_neurons = 8;
    net_params.n_inputs = 8 + 8;
    net_params.learning_rate = 0.0001;

    independent_pendulum_params s_params;
    s_params.friction_phi = 2.0;
    s_params.spring_constant_phi = 10.01;
    s_params.phi_0 = 0.0;
    
    independent_pendulum s(s_params);
  
    closed_loop_simulation sim(net_params, s);
    sim.simulate();


    return 0;
}
