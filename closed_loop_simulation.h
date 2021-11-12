#ifndef CLOSED_LOOP_SIMULATION_H
#define CLOSED_LOOP_SIMULATION_H

#include <cmath>
#include <random>
#include <fstream>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <experimental/filesystem>

#include "mechanical_system.h"
#include "neural_network.h"
#include "recorder.h"
#include "util.h"


std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist_current(0.0, 0.9);
std::uniform_real_distribution<double> dist_input(1.5, 2.9);

double input_gain = 2.0;

class closed_loop_simulation {
    public:

    rabit::network_BCM net;
    mechanical_system &mech;

    unsigned long recording_interval = 100;

    unsigned int n_tests = 100;
    unsigned int test_steps = 100000;
    unsigned long learning_steps = 1000;
    unsigned long learning_repetitions = 2000;

    std::string simulation_id;
    std::string recording_directory;

    std::vector<Eigen::VectorXd> random_inputs;

    void sample_random_input(Eigen::VectorXd &random_input)
    {
        for(int i = 0; i < random_input.rows(); i++)
            random_input(i) = dist_current(mt) < 0.5 ? dist_current(mt) : 0.1;
    }

    closed_loop_simulation(rabit::neural_network_parameters net_params,
                           mechanical_system &mech) : net(net_params), mech(mech)
    {
        assert(net_params.n_inputs % 2 == 0);

        net.connect_neurons_to_neurons_random();

        for(int i = 0; i < net_params.n_inputs / 2; i++)
            for(int j = 0; j < net_params.n_inputs / 2; j++)
                net.connect_input_to_neuron(i, j, dist_input(mt));

        for(int i = 0; i < net_params.n_inputs / 2; i++)
            net.connect_input_to_neuron(i + (net_params.n_inputs / 2), i, dist_input(mt));

        simulation_id = boost::uuids::to_string(boost::uuids::random_generator()());
        recording_directory = "./results/" + simulation_id + '/';
        std::experimental::filesystem::create_directory(recording_directory);

        for(int i = 0; i < n_tests; i++) {
            Eigen::VectorXd random_input(net.n_inputs / 2);
            sample_random_input(random_input);
            random_inputs.push_back(random_input);
        }            
    }

    void run_closed_loop(std::shared_ptr<rabit::uuid_recorder> recorder,
                         unsigned int steps, Eigen::VectorXd &random_input)
    {
        //Eigen::VectorXd random_input(net.n_inputs / 2);
        Eigen::VectorXd net_input(net.n_inputs);
        Eigen::Vector2d mech_input;

        //for(int i = 0; i < random_input.rows(); i++)
        //    random_input(i) = dist_current(mt) < 0.5 ? dist_current(mt) : 0.1;

        for(unsigned int i = 0; i < steps; i++) {
            net_input <<    mech.state. unaryExpr(relu),
                         (- mech.state).unaryExpr(relu),
                            random_input;


            for(int j = 0; j < mech_input.rows(); j++)
                mech_input(j) =    net.firing_rate[2 * j + 0] + net.firing_rate[2 * j + 1]
                                 - net.firing_rate[2 * j + 2] - net.firing_rate[2 * j + 3];

                
            net.set_inputs(net_input * input_gain);
            mech.set_input(mech_input);

            net.step(1);
            mech.step(1);
            
            if(i % recording_interval == 0)
                recorder->record();
        }
    }

    void test(std::string directory)
    {
        for(unsigned int j = 0; j < n_tests; j++) {
            std::shared_ptr test_recorder = 
                std::make_shared<rabit::uuid_recorder>(directory, test_steps / recording_interval);

            net.reset_activity();
            mech.reset_activity();

            net.set_recorder(test_recorder);
            mech.set_recorder(test_recorder);

            run_closed_loop(test_recorder, test_steps, random_inputs[j]); 

            std::cout << "Test results written to: " << test_recorder->get_filename() << '\n';
        }
    }   

    void learn()
    {
        net.enable_learning();

        std::shared_ptr recorder = 
                std::make_shared<rabit::uuid_recorder>(recording_directory,
                        learning_steps * learning_repetitions / recording_interval);
        
        mech.set_recorder(recorder);
        net.set_recorder(recorder);

        for(unsigned long repetition = 0; repetition < learning_repetitions; repetition++) {
            std::cout << repetition << " of " << learning_repetitions << " Repetitions\r" << std::flush;
           
            Eigen::VectorXd random_input(net.n_inputs / 2);
            sample_random_input(random_input);

            run_closed_loop(recorder, learning_steps, random_input);
        }
        std::cout << std::endl;
        std::cout << "Results written to: " << recorder->get_filename() << '\n';
    }

    void simulate()
    {
        std::string directory_pre_learn  = recording_directory + "pre_learn/";
        std::string directory_post_learn = recording_directory + "post_learn/";
        std::experimental::filesystem::create_directory(directory_pre_learn);
        std::experimental::filesystem::create_directory(directory_post_learn);
        
        /*Test unlearnt network*/
        std::cout << "Test unlearnt network\n";
        
        net.disable_learning();
        test(directory_pre_learn);

        /*Learn network*/
        std::cout << "Learn network\n";
        
        learn();

        /*Test learnt network*/
        std::cout << "Test learnt network\n";

        net.disable_learning();
        test(directory_post_learn);

        std::ofstream out(recording_directory + mech.description);
        out << "";
        out.close();
    }
};

#endif
