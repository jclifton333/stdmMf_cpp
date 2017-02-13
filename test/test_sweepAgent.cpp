#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>

#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "network.hpp"
#include "networkRunFeatures.hpp"
#include "sweepAgentSlow.hpp"
#include "sweepAgent.hpp"

namespace stdmMf {

TEST(TestSweepAgentSlow, ApplyTrt) {
    // generate network
    NetworkInit init;
    init.set_dim_x(5);
    init.set_dim_y(4);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::shared_ptr<NetworkRunFeatures> nrf(new NetworkRunFeatures(n, 1));

    // std::random_device rd;
    const uint32_t seed = 0; // crd();
    njm::tools::Rng rng;
    rng.seed(seed);

    // setup sweep agent
    std::vector<double> coef(nrf->num_features());
    std::for_each(coef.begin(), coef.end(),
            [&rng] (double & x) {
                x = rng.rnorm_01();
            });
    SweepAgentSlow sa(n, nrf, coef, 0);


    for (uint32_t reps = 0; reps < 50; ++reps) {
        // get random infection state
        boost::dynamic_bitset<> inf_bits(n->size());
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list = rng.sample_range(0, n->size(),
                num_inf);
        for (uint32_t i = 0; i < num_inf; ++i) {
            inf_bits.set(inf_list.at(i));
        }


        // get best by brute force
        double brute_best_val = std::numeric_limits<double>::lowest();
        std::string brute_bits;
        {
            std::vector<uint32_t> trt_list(n->size(), 0);
            std::fill(trt_list.begin(), trt_list.begin() + sa.num_trt(), 1);

            do {
                // get trt_bits
                boost::dynamic_bitset<> trt_bits(n->size());
                for (uint32_t i = 0; i < n->size(); ++i) {
                    if (trt_list.at(i) == 1) {
                        trt_bits.set(i);
                    }
                }

                ASSERT_EQ(trt_bits.count(), sa.num_trt());

                // calc features
                const std::vector<double> f = nrf->get_features(inf_bits,
                        trt_bits);

                const double val = njm::linalg::dot_a_and_b(coef, f);

                // assume no ties since coef are random
                if (val > brute_best_val) {
                    brute_best_val = val;
                    boost::to_string(trt_bits, brute_bits);
                }
            } while(std::prev_permutation(trt_list.begin(), trt_list.end()));
        }


        // get best from SweepAgentSlow
        double sweep_agent_best_val = 0.0;
        std::string sweep_bits;
        {
            const boost::dynamic_bitset<> trt_bits = sa.apply_trt(inf_bits,
                    std::vector<InfAndTrt>());
            boost::to_string(trt_bits, sweep_bits);

            const std::vector<double> f = nrf->get_features(inf_bits, trt_bits);
            sweep_agent_best_val = njm::linalg::dot_a_and_b(coef, f);
        }


        EXPECT_EQ(sweep_agent_best_val, brute_best_val)
            << std::endl
            << "sweep: " << sweep_bits << std::endl
            << "brute: " << brute_bits << std::endl;
    }
}

TEST(TestSweepAgent, TestEquality) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::shared_ptr<Features> f_slow(new NetworkRunFeatures(n, 4));
    std::shared_ptr<Features> f(new NetworkRunFeatures(n, 4));

    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);

    for (uint32_t i = 0; i < 5; ++i) {
        std::vector<double> coef(f->num_features(),0.);
        std::for_each(coef.begin(), coef.end(),
                [&rng](double & x) {
                    x = rng->rnorm_01();
                });

        SweepAgent sa(n, f, coef, 2, true);
        SweepAgentSlow sas(n, f_slow, coef, 2);

        for (uint32_t j = 0; j < 5; ++j) {
            // randomly infect 10 percent
            const std::vector<int> inf_bits_vec = rng->sample_range(0,
                    n->size(), n->size()*0.1);
            boost::dynamic_bitset<> inf_bits(n->size());
            for (uint32_t k = 0; k < inf_bits_vec.size(); ++k) {
                inf_bits.set(inf_bits_vec.at(k));
            }

            // apply trt with both agents
            const boost::dynamic_bitset<> trt_bits = sa.apply_trt(inf_bits);
            const boost::dynamic_bitset<> trt_bits_slow = sas.apply_trt(
                    inf_bits);

            EXPECT_EQ(trt_bits, trt_bits_slow);
        }
    }
}


TEST(TestSweepAgent, TestScaling) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::shared_ptr<Features> f(new NetworkRunFeatures(n, 4));

    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);

    for (uint32_t i = 0; i < 25; ++i) {
        // randomly infect 10 percent
        const std::vector<int> inf_bits_vec = rng->sample_range(0,
                n->size(), n->size()*0.1);
        boost::dynamic_bitset<> inf_bits(n->size());
        for (uint32_t k = 0; k < inf_bits_vec.size(); ++k) {
            inf_bits.set(inf_bits_vec.at(k));
        }

        std::vector<double> coef(f->num_features(),0.);
        std::for_each(coef.begin(), coef.end(),
                [&rng](double & x) {
                    x = rng->rnorm_01();
                });

        SweepAgent sa(n, f, coef, 2, true);

        const boost::dynamic_bitset<> trt_bits = sa.apply_trt(inf_bits);

        // scale parameters
        const double scalar = rng->runif_01();
        std::for_each(coef.begin(), coef.end(),
                [&scalar](double & x) {
                    x *= scalar;
                });

        SweepAgent sa_scaled(n, f, coef, 2, true);

        const boost::dynamic_bitset<> trt_bits_scaled = sa_scaled.apply_trt(
                inf_bits);

        EXPECT_EQ(trt_bits_scaled, trt_bits);
    }
}


TEST(TestSweepAgent, TestParallel) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);;
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::shared_ptr<Features> f(new NetworkRunFeatures(n, 4));

    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);

    for (uint32_t i = 0; i < 25; ++i) {
        // randomly infect 10 percent
        const std::vector<int> inf_bits_vec = rng->sample_range(0,
                n->size(), n->size()*0.1);
        boost::dynamic_bitset<> inf_bits(n->size());
        for (uint32_t k = 0; k < inf_bits_vec.size(); ++k) {
            inf_bits.set(inf_bits_vec.at(k));
        }

        std::vector<double> coef(f->num_features(),0.);
        std::for_each(coef.begin(), coef.end(),
                [&rng](double & x) {
                    x = rng->rnorm_01();
                });

        SweepAgent sa(n, f, coef, 0, false);

        const boost::dynamic_bitset<> trt_bits = sa.apply_trt(inf_bits);

        SweepAgent sa_parallel(n, f, coef, 0, false);
        sa_parallel.set_parallel(true, 1);

        const boost::dynamic_bitset<> trt_bits_parallel = sa_parallel.apply_trt(
                inf_bits);

        CHECK_EQ(trt_bits_parallel, trt_bits);
    }
}


} // namespace stdmMf



int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
