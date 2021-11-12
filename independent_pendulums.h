#ifndef INDEPENDENT_PENDULUM_H 
#define INDEPENDENT_PENDULUM_H

#include <iostream>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>

#include "recorder.h"
#include "mechanical_system.h"

struct independent_pendulum_params {
    double friction_phi;
    double spring_constant_phi;
    double phi_0;
};

class independent_pendulum : public mechanical_system {
    private :

    const double slowdown = 1 / 1000.0;
    const double force_factor = 12.0;

    const struct independent_pendulum_params params;

    Eigen::Vector2d tau; /* Force input */

    public:
    
    std::string description = "Independent_Pendulums";
    
    void reset_activity()
    {
        state = Eigen::VectorXd(4);
        state << params.phi_0, params.phi_0, 0.0, 0.0;
        
        tau = Eigen::Vector2d::Zero();
    }

    independent_pendulum(const independent_pendulum_params params):
        params(params)
    {
        reset_activity();
    }

    void set_recorder(std::shared_ptr<rabit::uuid_recorder> recorder)
    {
        this->recorder = recorder;
        recorder->create_table("mech_system_state", &state);
    }

    void set_input(const Eigen::VectorXd &input)
    {
        this->tau = force_factor * input;
    }

    private:

    double kappa_phi(const double phi)
    {
        return (phi - params.phi_0) * params.spring_constant_phi;
    }

    double beta_phi(const double dphi)
    {
        return params.friction_phi * dphi;
    }

    void grad_state(const Eigen::VectorXd &state, Eigen::VectorXd &grad, const double /*t*/)
    {
        Eigen::Vector2d phi;
        phi << state(0), state(1);

        Eigen::Vector2d dphi;
        dphi << state(2), state(3);

        Eigen::Vector2d ddphi;
        for(int i = 0; i < 2; i++)
            ddphi(i) = - kappa_phi(phi(i)) - beta_phi(dphi(i)) + tau(i);
        
        Eigen::VectorXd res(4);
        res << dphi, ddphi;
        grad = slowdown * res;
    }

    public:

    void step(const int time)
    {
        double start_t = 0.0;
        double end_t = start_t + double(time);
        double step_size = double(time) / 10.0;


        namespace odeint = boost::numeric::odeint;
        using stepper = odeint::runge_kutta_fehlberg78<Eigen::VectorXd, double, 
                                             Eigen::VectorXd, double,
                                             odeint::vector_space_algebra>;
        stepper s;

        auto system = [this] (const Eigen::VectorXd &state, Eigen::VectorXd &grad, const double t)
            -> void {this -> grad_state(state, grad, t);};

        //odeint::integrate_const(s, system, state, start_t, end_t, step_size);
        odeint::integrate_adaptive(s, system, state, start_t, end_t, step_size);

        /*if(state(0) > 1.0)
            state(0) = 1.0;
        if(state(0) < -1.0)
            state(0) = -1.0;

        if(state(1) > 1.0)
            state(1) = 1.0;
        if(state(1) < -1.0)
            state(1) = -1.0;*/
    }
};

#endif
