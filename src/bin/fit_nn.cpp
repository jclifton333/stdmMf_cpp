#include "sim_data.pb.h"

#include "states.hpp"

#include "neuralNetwork.hpp"

#include <caffe/caffe.hpp>
#include <caffe/layer.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include <stdlib.h>

#include <numeric>

#include <boost/dynamic_bitset.hpp>

DEFINE_string(data_file, "", "Path to protobuf file");
DEFINE_string(model_file, "", "Path to model file");
DEFINE_string(solver_file, "", "Path to solver file");

using namespace stdmMf;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetCommandLineOption("GLOG_minloglevel", "2");

    // load data
    std::ifstream ifs(FLAGS_data_file);
    CHECK(ifs.good());
    const std::string content( (std::istreambuf_iterator<char>(ifs) ),
            (std::istreambuf_iterator<char>()    ) );
    ifs.close();

    SimData sd;
    sd.ParseFromString(content);

    // setup training data
    std::vector<StateAndTrt<InfShieldState> > state_trt_data;
    std::vector<double> reward_data;

    // extract into vectors
    uint32_t total = 0;
    for (uint32_t i = 0; i < sd.rep_size(); ++i) {
        const Observation & obs(sd.rep(i));
        for (uint32_t j = 0; j < obs.transition_size(); ++j) {
            ++total;

            const TransitionPB & trans(obs.transition(j));
            const uint32_t num_nodes(trans.curr_trt_bits().length());
            CHECK_EQ(num_nodes, 100);


            // curr state
            const boost::dynamic_bitset<> inf_bits(
                    trans.curr_state().inf_bits());
            const std::vector<double> shield(
                    trans.curr_state().shield().begin(),
                    trans.curr_state().shield().end());
            const InfShieldState curr_state(inf_bits, shield);


            // trt
            const boost::dynamic_bitset<> trt_bits(trans.curr_trt_bits());

            state_trt_data.emplace_back(curr_state, trt_bits);

            // reward
            {
                const boost::dynamic_bitset<> bits(
                        trans.next_state().inf_bits());
                float sum_reward(0.0);
                for (uint32_t k = 0; k < num_nodes; ++k) {
                    sum_reward -= bits.test(k) ? 1. : 0.;
                }

                reward_data.push_back(sum_reward / num_nodes);
            }
        }
    }

    NeuralNetwork<InfShieldState> nn(FLAGS_model_file, FLAGS_solver_file,
            100, 100);
    nn.train_data(state_trt_data, reward_data);

    nn.fit();

    std::cout << nn.eval(state_trt_data.at(0))
              << " == "
              << reward_data.at(0)
              << std::endl;

    return 0;
}
