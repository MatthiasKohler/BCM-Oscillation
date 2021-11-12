#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <iostream>
#include <vector>
#include <cmath>
#include <optional>
#include <random>

#include <boost/numeric/odeint.hpp>

#include <Eigen/Dense>

#include "recorder.h"

namespace rabit {

double sign(double x)
{
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}


double sigmoid(double x) 
{
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_diff(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double shifted_sigmoid(double x)
{
    x = (10.0 * x) - 5.0;
    return sigmoid(x);
}

double shifted_sigmoid_diff(double x)
{
    return shifted_sigmoid(x) * (1 - shifted_sigmoid(x)) * 10;
}

double relu(double x)
{
    if(x < 0.0)
        return 0.0;
    else if (x > 1.0)
        return 1.0;
    else
        return x;
}


struct neural_network_parameters {
    int n_inputs;
    int n_neurons;
    double tau_mem; /* Membrane time constant in milliseconds */
    double learning_rate;
};


class network {
    public:
 
    const int n_inputs;
    const int n_neurons;
    const double tau_mem; /* Membrane time constant in milliseconds */
    const double learning_rate;

    bool learning_enabled = true;

    Eigen::VectorXd V_inputs;  /* Input voltage */
    Eigen::VectorXd V_neurons; /* Neuron membrane voltage */
    Eigen::VectorXd firing_rate;
    Eigen::VectorXd I_in;
    Eigen::MatrixXd w_inputs;  /* Input weights */
    Eigen::MatrixXd w_neurons; /* Internal synaptic weights */

    std::shared_ptr<uuid_recorder> recorder;

    void reset_activity()
    {
        V_inputs  = Eigen::VectorXd::Zero(n_inputs);
        V_neurons = Eigen::VectorXd::Zero(n_neurons);
        firing_rate = Eigen::VectorXd::Zero(n_neurons);
        I_in = Eigen::VectorXd::Zero(n_neurons);
    }

    network(const struct neural_network_parameters &params) :         
        n_inputs(params.n_inputs),
        n_neurons(params.n_neurons),
        tau_mem(params.tau_mem),
        learning_rate(params.learning_rate)
    {
        reset_activity();
        w_inputs  = Eigen::MatrixXd::Zero(n_neurons, n_inputs);
        w_neurons = Eigen::MatrixXd::Zero(n_neurons, n_neurons);
    }

    void set_recorder(std::shared_ptr<uuid_recorder> recorder)
    {
        this->recorder = recorder;
        recorder->create_table("V_inputs", &V_inputs);        
        recorder->create_table("V_neurons", &V_neurons);
        recorder->create_table("firing_rate", &firing_rate);
        recorder->create_table("w_inputs", &w_inputs);
        recorder->create_table("w_neurons", &w_neurons);
    }

    void connect_neuron_to_neuron(const int src_neuron, const int dst_neuron, 
                                  double w)
    {
        assert(0 <= src_neuron);
        assert(src_neuron <= n_neurons);
        assert(0 <= dst_neuron);
        assert(dst_neuron <= n_neurons);

        w_neurons(src_neuron, dst_neuron) = w;
    }

    void connect_input_to_neuron(const int src_input, const int dst_neuron,
                                 double w)
    {
        assert(0 <= src_input);
        assert(src_input <= n_inputs);
        assert(0 <= dst_neuron);
        assert(dst_neuron <= n_neurons);

        w_inputs(dst_neuron, src_input) = w;
    }

    void connect_neurons_all_to_all(const std::vector<int> src, 
                                    const std::vector<int> dst, 
                                    double (*initalizer)())
    {
        for(int i : src)
            for(int j : dst)
                connect_neuron_to_neuron(i, j, initalizer()); 
    }

    void connect_neurons_to_neurons_random()
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-0.9, 0.9);

        for(int i = 0; i < n_neurons; i++) {
            for(int j = 0; j < n_neurons; j++) {
                if(i != j)
                    connect_neuron_to_neuron(i, j, dist(mt));               
            }
        }
    }

    void connect_inputs_to_neurons_all_to_all(const std::vector<int> src,
                                              const std::vector<int> dst,
                                              double (*initalizer)())
    {
        for(int input : src)
            for(int neuron : dst) 
                connect_input_to_neuron(input, neuron, initalizer());
    }

    void connect_inputs_to_neurons_random()
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.1, 0.9);

        for(int input = 0; input < n_inputs; input++){
            for(int neuron = 0; neuron < n_neurons; neuron++)
                connect_input_to_neuron(input, neuron, dist(mt));                
        }
    }

    void connect_fully_random()
    {
        connect_neurons_to_neurons_random();
        connect_inputs_to_neurons_random();
    }

    void set_inputs(const Eigen::VectorXd &inputs)
    {
        V_inputs = inputs;
    }

    void set_I_in(const Eigen::VectorXd &in)
    {
        I_in = in;
    }

    private:

    void grad_voltage(const Eigen::VectorXd &V, Eigen::VectorXd &dVdt, const double)
    {
        const Eigen::VectorXd firing_rate = V.unaryExpr(&relu);
        const Eigen::VectorXd a = (w_neurons * firing_rate) +
                                  (w_inputs * V_inputs) + 
                                  I_in;
        const Eigen::VectorXd b = Eigen::VectorXd::Ones(n_neurons) +
                                  (w_neurons.cwiseAbs() * firing_rate) +
                                  (w_inputs.cwiseAbs()  * V_inputs) +
                                  I_in;

        dVdt = (-V + a.cwiseQuotient(b)) / tau_mem;
    }

    virtual void grad_input_weights(const Eigen::MatrixXd &w, Eigen::MatrixXd &dwdt, const double)
    {
        dwdt = Eigen::MatrixXd::Zero(w.rows(), w.cols());
    }

    virtual void grad_neuron_weights(const Eigen::MatrixXd &w, Eigen::MatrixXd &dwdt, const double)
    {
        dwdt = Eigen::MatrixXd::Zero(w.rows(), w.cols());
    }

    public:

    void enable_learning() { learning_enabled = true; }
    void disable_learning() { learning_enabled = false; }

    /* Advance network state by time milliseconds */
    void step(const int time)
    {
        double start_t = 0.0;
        double end_t = start_t + double(time);
        double step_size = double(time) / 10.0;

        namespace odeint = boost::numeric::odeint;
        using vector_stepper = odeint::runge_kutta4<Eigen::VectorXd, double, 
                                                    Eigen::VectorXd, double,
                                                    odeint::vector_space_algebra>; 
        using matrix_stepper = odeint::runge_kutta4<Eigen::MatrixXd, double,
                                                    Eigen::MatrixXd, double,
                                                    odeint::vector_space_algebra>;
        vector_stepper voltage_stepper;
        
        auto voltage_system = [this] (const Eigen::VectorXd &V, 
                                            Eigen::VectorXd &dVdt,
                                      const double t)
            -> void {this->grad_voltage(V, dVdt, t);};
        
        odeint::integrate_const(voltage_stepper, voltage_system, V_neurons,
                                start_t, end_t, step_size);

        if(learning_enabled) {
            
            matrix_stepper w_neurons_stepper;
            auto w_neurons_system = [this] (const Eigen::MatrixXd &w,
                                                  Eigen::MatrixXd &dwdt,
                                            const double t)
                -> void {this->grad_neuron_weights(w, dwdt, t);};

            odeint::integrate_const(w_neurons_stepper, w_neurons_system, w_neurons,
                      start_t, end_t, step_size);


            matrix_stepper w_inputs_stepper;
            auto w_inputs_system = [this] (const Eigen::MatrixXd &w,
                                                 Eigen::MatrixXd &dwdt,
                                            const double t)
                -> void {this->grad_input_weights(w, dwdt, t);};
            odeint::integrate_const(w_inputs_stepper, w_inputs_system, w_inputs,
                                    start_t, end_t, step_size);
        }

        firing_rate = V_neurons.unaryExpr(&relu);
    }

};



class network_BCM : public network {

    private:

    using network::network;

    Eigen::VectorXd learning_threshold = Eigen::VectorXd::Ones(n_neurons);

    double tau_activity = 500.0;

    public:

    void set_recorder(std::shared_ptr<uuid_recorder> recorder)
    {
        network::set_recorder(recorder);
        recorder->create_table("learning_threshold", &learning_threshold);
    }

    private:

    void grad_learning_threshold(const Eigen::VectorXd &th, Eigen::VectorXd &dthdt, const double)
    {
        dthdt = (firing_rate.cwiseProduct(firing_rate) - th) / tau_activity;
    }

    void grad_weights(Eigen::MatrixXd &dwdt, 
                      const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::MatrixXd &w)
    {
        Eigen::MatrixXd learning_rates = w.unaryExpr([this](double elem) {
                        return  learning_rate;
                });
        
        Eigen::VectorXd postsynaptic_val = (y.cwiseProduct((0.5 * y) - learning_threshold));
        dwdt = learning_rates.cwiseProduct(postsynaptic_val * x.transpose());
    }

    void grad_input_weights(const Eigen::MatrixXd &w, Eigen::MatrixXd &dwdt, const double)
    {
        grad_weights(dwdt, V_inputs, firing_rate, w);
    }

    void grad_neuron_weights(const Eigen::MatrixXd &w, Eigen::MatrixXd &dwdt, const double)
    {
        grad_weights(dwdt, firing_rate, firing_rate, w);
        dwdt.diagonal() = Eigen::VectorXd::Zero(n_neurons);
    }

    public:

    void step(const int time)
    {
        double start_t = 0.0;
        double end_t = start_t + double(time);
        double step_size = double(time) / 10.0;

        namespace odeint = boost::numeric::odeint;

        if(learning_enabled) {

            using vector_stepper = odeint::runge_kutta4<Eigen::VectorXd, double, 
                                                        Eigen::VectorXd, double,
                                                        odeint::vector_space_algebra>;

            vector_stepper learning_threshold_stepper;
            
            auto learning_threshold_system = [this] (const Eigen::VectorXd &th, 
                                                     Eigen::VectorXd &dthdt,
                                                     const double t)
                -> void {this->grad_learning_threshold(th, dthdt, t);};
            
            odeint::integrate_const(learning_threshold_stepper, learning_threshold_system, learning_threshold,
                                    start_t, end_t, step_size);

        }
        network::step(time);
    }
};


};

#endif
