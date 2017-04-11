#include "sim_data.pb.h"

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
    // google::InitGoogleLogging(argv[0]);
    // google::SetCommandLineOption("GLOG_minloglevel", "2");

    // load data
    std::ifstream ifs(FLAGS_data_file);
    CHECK(ifs.good());
    const std::string content( (std::istreambuf_iterator<char>(ifs) ),
            (std::istreambuf_iterator<char>()    ) );
    ifs.close();

    SimData sd;
    sd.ParseFromString(content);

    std::vector<float> state_trt_data;
    std::vector<float> reward_data;

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
            {
                const boost::dynamic_bitset<> bits(
                        trans.curr_state().inf_bits());
                for (uint32_t k = 0; k < num_nodes; ++k) {
                    state_trt_data.push_back(bits.test(k) ? 1 : 0);
                }
            }

            for (uint32_t k = 0; k < num_nodes; ++k) {
                state_trt_data.push_back(trans.curr_state().shield(k));
            }

            // trt
            {
                const boost::dynamic_bitset<> bits(trans.curr_trt_bits());
                for (uint32_t k = 0; k < num_nodes; ++k) {
                    state_trt_data.push_back(bits.test(k) ? 1 : 0);
                }
            }

            // reward
            {
                const boost::dynamic_bitset<> bits(
                        trans.next_state().inf_bits());
                float sum_reward(0.0);
                for (uint32_t k = 0; k < num_nodes; ++k) {
                    sum_reward -= bits.test(k) ? 1. : 0.;
                }

                reward_data.push_back(10 * sum_reward / num_nodes);
            }
        }
    }

    CHECK_EQ(reward_data.size(), total);
    CHECK_EQ(state_trt_data.size(), total * 3 * 100);

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver_file, &solver_param);
    solver_param.set_net(FLAGS_model_file);

    std::shared_ptr<caffe::Solver<float> > solver(
            caffe::SolverRegistry<float>::CreateSolver(solver_param));

    caffe::MemoryDataLayer<float> *input_data_layer =
        (caffe::MemoryDataLayer<float> *) (
                solver->net()->layer_by_name("input_data").get());

    input_data_layer->Reset(state_trt_data.data(), reward_data.data(), total);

    std::cout << "solving" << std::endl;
    solver->Solve();
    std::cout << "done" << std::endl;

    solver->net()->Forward();

    boost::shared_ptr<caffe::Blob<float> > input_layer(
            solver->net()->blob_by_name("reward"));
    boost::shared_ptr<caffe::Blob<float> > output_layer(
            solver->net()->blob_by_name("fc3"));
    CHECK_EQ(input_layer->count(), 200);
    CHECK_EQ(output_layer->count(), 200);
    for (uint32_t i = 0; i < 2; ++i) {
        std::cout << input_layer->cpu_data()[i]
                  << "  ==  "
                  << output_layer->cpu_data()[i]
                  << std::endl;
    }


    std::cout << "setup test net" << std::endl;
    std::shared_ptr<caffe::Net<float> > testnet;
    testnet.reset(new caffe::Net<float>(FLAGS_model_file, caffe::TEST));

   testnet->ShareTrainedLayersWith(solver->net().get());

    caffe::MemoryDataLayer<float> *input_test_data_layer =
        (caffe::MemoryDataLayer<float> *) (
                testnet->layer_by_name("input_data").get());
    input_test_data_layer->Reset(state_trt_data.data(),
            reward_data.data(), total);
    testnet->Forward();
    std::cout << testnet->blob_by_name("reward")->cpu_data()[0]
              << "  ==  "
              << testnet->blob_by_name("fc3")->cpu_data()[0]
              << std::endl;

    return 0;
}
