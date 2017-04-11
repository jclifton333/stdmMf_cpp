#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <caffe/caffe.hpp>
#include <caffe/layer.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "states.hpp"

namespace stdmMf {

template <typename State>
class NeuralNetwork {
protected:
    std::shared_ptr<caffe::Solver<double> > solver_;
    std::shared_ptr<caffe::Net<double> > eval_net_;

    const uint32_t batch_size_;
    const uint32_t num_nodes_;
    static const uint32_t num_input_per_node_;

    std::vector<double> state_trt_train_data_;
    std::vector<double> outcome_train_data_;

    std::vector<double> state_trt_eval_data_;
    std::vector<double> outcome_eval_data_;

public:
    NeuralNetwork(const std::string & model_file,
            const std::string & solver_file,
            const uint32_t & batch_size,
            const uint32_t & num_nodes);

    NeuralNetwork(const NeuralNetwork & other);

    virtual ~NeuralNetwork() = default;

    void train_data(const std::vector<StateAndTrt<State> > & state_trt,
            const std::vector<double> & outcome);

    void fit();

    double eval(const StateAndTrt<State> & state_trt);
};


} // namespace stdmMf


#endif // NEURAL_NETWORK_HPP
