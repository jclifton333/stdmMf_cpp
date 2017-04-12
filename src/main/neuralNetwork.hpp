#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <caffe/caffe.hpp>
#include <caffe/layer.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "states.hpp"

#include <njm_cpp/tools/random.hpp>

namespace stdmMf {

// TODO: write some test cases
template <typename State>
class NeuralNetwork : public njm::tools::RngClass {
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

    const uint32_t max_sweeps_;

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

    std::pair<boost::dynamic_bitset<>, double>
    sweep_max(const State & state, const uint32_t & num_trt);

    void set_new_treatment(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state);

    bool sweep_treatments(boost::dynamic_bitset<> & trt_bits,
            double & best_val,
            const uint32_t & num_trt,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state);

    using njm::tools::RngClass::rng;
    virtual void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // NEURAL_NETWORK_HPP
