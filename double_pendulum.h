#ifndef DOUBLE_PENDULUM_H
#define DOUBLE_PENDULUM_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>

#include "mechanical_system.h"
#include "recorder.h"


class double_pendulum : public mechanical_system {
    private:

    const double slowdown = 1 / 1000.0;
    const double force_factor = 6.0; 

    double g = 9.81;
    Eigen::Vector2d m;  //Masses
    Eigen::Vector2d l;  //Link lengths
    Eigen::Vector2d lc; //Moment of inertia
    Eigen::Vector2d II; //Center of masses

    //Not in Alins model
    Eigen::Vector2d damping;

    Eigen::Vector2d q;  //Joint angles
    Eigen::Vector2d dq; //Joint angular velocities

    //Torque to joints is applied by changing the stiffness of linear springs
    Eigen::Vector2d tau;

    public:

    std::string description = "Double_Pendulum";

    void reset_activity()
    {
        q = dq = tau = Eigen::Vector2d::Zero();

        state = Eigen::VectorXd::Zero(4);
    }

    double_pendulum()
    {
        m  << 1.0, 1.0;
        l  << 2.0, 2.0;
        lc << 1.0, 1.0;
        II << 1.0, 1.0;

        damping << 1.0, 1.0;

        reset_activity();
    }

    void set_input(const Eigen::VectorXd &input)
    {
        this->tau = force_factor * input;
    }

    private:

    void get_system(Eigen::Matrix2d &M, Eigen::Matrix2d &C, Eigen::Vector2d &G, Eigen::Vector2d &dq, Eigen::Vector2d &q)
    {
        double alpha = II(0) + II(1) + m(0)* std::pow(lc(0), 2) + m(1)*(std::pow(l(0), 2) + std::pow(lc(1), 2));
        double beta  = m(1)*l(0)*lc(1);
        double delta = II(1) + m(1)*std::pow(lc(1), 2);

        M(0,0) = alpha + 2*beta*cos(q(1));
        M(0,1) = delta + beta*cos(q(1));
        M(1,0) = M(0,1);
        M(1,1) = delta;

        double h = -m(1)*l(0)*lc(1)*sin(q(1));

        C(0,0) = h*dq(1);
        C(0,1) = h*(dq(0) + dq(1));
        C(1,0) = -h*dq(0);
        C(1,1) = 0;

        G(0) = (m(0) * lc(0) + m(1) * l(0)) * g * cos(q(0)) + m(1) * lc(1) * g * cos(q(0) + q(1));
        G(1) = m(1)*lc(1)*g*cos(q(0) + q(1));
    }

    void get_ddq(Eigen::Vector2d &ddq, Eigen::Vector2d &dq, Eigen::Vector2d &q)
    {
        Eigen::Matrix2d M;
        Eigen::Matrix2d C;
        Eigen::Vector2d G;

        get_system(M, C, G, dq, q);

        // M ddq + C dq + G = tau - damping * dq
        ddq = M.colPivHouseholderQr().solve(tau - (damping.cwiseProduct(dq)) - (C * dq) - G);
    }

    public:

    void step(const int time)
    {
        double start_t = 0.0;
        double end_t = start_t + double(time);
        double step_size = double(time) / 10.0;


        namespace odeint = boost::numeric::odeint;
        using stepper = odeint::runge_kutta_fehlberg78<Eigen::Vector4d, double, 
                                             Eigen::Vector4d, double,
                                             odeint::vector_space_algebra>;
        stepper s;

        state << dq, q;

        auto system = [this] (const Eigen::Vector4d &state, Eigen::Vector4d &grad, const double)
            -> void {
                Eigen::Vector2d ddq, dq, q;
                dq << state(0), state(1);
                q  << state(2), state(3);
                get_ddq(ddq, dq, q);
                grad << slowdown * ddq, slowdown * dq;
            };

        //odeint::integrate_const(s, system, state, start_t, end_t, step_size);
        odeint::integrate_adaptive(s, system, state, start_t, end_t, step_size);

        dq << state(0), state(1);
        q  << state(2), state(3);
    }
};

#endif
