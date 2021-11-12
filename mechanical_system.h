#ifndef MECHANICAL_SYSTEM_H
#define MECHANICAL_SYSTEM_H


#include <string>

#include <Eigen/Dense>

struct mechanical_system_params {};

class mechanical_system {
    public:

    Eigen::VectorXd state;
    virtual void set_input(const Eigen::VectorXd &/*input*/) {}
    virtual void reset_activity() {}

    std::string description;

    std::shared_ptr<rabit::uuid_recorder> recorder;
    
    virtual void set_recorder(std::shared_ptr<rabit::uuid_recorder> recorder)
    {
        this->recorder = recorder;
        recorder->create_table("mech_system_state", &state);
    }

    virtual void step(const int /*time*/) {}
};


#endif
