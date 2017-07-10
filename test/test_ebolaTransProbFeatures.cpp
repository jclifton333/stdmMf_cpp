#include <gtest/gtest.h>
#include <glog/logging.h>

#include <random>

#include <njm_cpp/tools/random.hpp>
#include "network.hpp"
#include "states.hpp"
#include "ebolaStateGravityModel.hpp"
#include "ebolaTransProbFeatures.hpp"
#include "ebolaData.hpp"

namespace stdmMf {


TEST(TestEbolaTransProbFeatures, UpdateFeatures) {
    EbolaData::deinit();
    EbolaData::init();
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);
    ASSERT_EQ(n->size(), 290);

    std::shared_ptr<Model<EbolaState> > mod(new EbolaStateGravityModel(n));
    std::vector<double> grav_par{-3.105, 1.434, 0.051, -1.117, -1.117};
    mod->par(grav_par);

    std::random_device device;
    const uint32_t seed = device();
    njm::tools::Rng rng;
    rng.seed(seed);

    EbolaTransProbFeatures ef_get(n, mod);
    EbolaTransProbFeatures ef_update(n, mod);
    for (uint32_t reps = 0; reps < 100; ++reps) {
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list =
            rng.sample_range(0, n->size(), num_inf);

        const uint32_t num_trt = rng.rint(0, n->size());
        const std::vector<int> trt_list =
            rng.sample_range(0, n->size(), num_trt);

        boost::dynamic_bitset<> inf_bits(n->size()), trt_bits(n->size());
        for (uint32_t i = 0; i < num_inf; ++i) {
            inf_bits.set(inf_list.at(i));
        }
        for (uint32_t i = 0; i < num_trt; ++i) {
            trt_bits.set(trt_list.at(i));
        }

        std::string inf_string, trt_string;
        boost::to_string(inf_bits, inf_string);
        boost::to_string(trt_bits, trt_string);

        ef_get.update_all_probs();
        ef_update.update_all_probs();


        std::vector<double> f_orig;

        // get features again to reset paths properly
        f_orig = ef_update.get_features(inf_bits,
                trt_bits);


        // flip trt
        boost::dynamic_bitset<> trt_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = ef_get.get_features(inf_bits,
                    trt_bits_flipped);

            // get features to reset masks properly
            f_orig = ef_update.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            ef_update.update_features(i, inf_bits, trt_bits_flipped, inf_bits,
                    trt_bits, f_upd);

            for (uint32_t j = 0; j < ef_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_new.at(j), 1e-12)
                    << "Flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }
        }

    }
}



TEST(TestEbolaTransProbFeatures, Copy) {
    EbolaData::deinit();
    EbolaData::init();
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);
    ASSERT_EQ(n->size(), 290);

    std::shared_ptr<Model<EbolaState> > mod(new EbolaStateGravityModel(n));
    std::vector<double> grav_par{-3.105, 1.434, 0.051, -1.117, -1.117};
    mod->par(grav_par);

    std::random_device device;
    const uint32_t seed = device();
    njm::tools::Rng rng;
    rng.seed(seed);

    EbolaTransProbFeatures ef_get(n, mod);
    EbolaTransProbFeatures ef_update(n, mod);

    EbolaTransProbFeatures ef_get_cpy(ef_get);
    EbolaTransProbFeatures ef_update_cpy(ef_update);

    EXPECT_EQ(ef_get.num_features(), ef_get_cpy.num_features());
    EXPECT_EQ(ef_update.num_features(),
            ef_update_cpy.num_features());

    for (uint32_t reps = 0; reps < 100; ++reps) {
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list =
            rng.sample_range(0, n->size(), num_inf);

        const uint32_t num_trt = rng.rint(0, n->size());
        const std::vector<int> trt_list =
            rng.sample_range(0, n->size(), num_trt);

        boost::dynamic_bitset<> inf_bits(n->size()), trt_bits(n->size());
        for (uint32_t i = 0; i < num_inf; ++i) {
            inf_bits.set(inf_list.at(i));
        }
        for (uint32_t i = 0; i < num_trt; ++i) {
            trt_bits.set(trt_list.at(i));
        }

        std::string inf_string, trt_string;
        boost::to_string(inf_bits, inf_string);
        boost::to_string(trt_bits, trt_string);

        ef_get.update_all_probs();
        ef_update.update_all_probs();

        ef_get_cpy.update_all_probs();
        ef_update_cpy.update_all_probs();

        std::vector<double> f_orig;
        std::vector<double> f_orig_cpy;

        // get features again to reset paths properly
        f_orig = ef_update.get_features(inf_bits,
                trt_bits);

        f_orig_cpy = ef_update_cpy.get_features(inf_bits,
                trt_bits);


        // flip trt
        boost::dynamic_bitset<> trt_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = ef_get.get_features(inf_bits,
                    trt_bits_flipped);

            const std::vector<double> f_new_cpy = ef_get_cpy.get_features(
                    inf_bits, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = ef_update.get_features(inf_bits,
                    trt_bits);
            f_orig_cpy = ef_update_cpy.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);

            std::vector<double> f_upd_cpy(f_orig_cpy);

            ef_update.update_features(i, inf_bits, trt_bits_flipped, inf_bits,
                    trt_bits, f_upd);

            ef_update_cpy.update_features(i, inf_bits, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd_cpy);

            for (uint32_t j = 0; j < ef_get.num_features(); ++j) {
                EXPECT_NEAR(f_new.at(j), f_new_cpy.at(j), 1e-12)
                    << "Copy failed for new";
            }

            for (uint32_t j = 0; j < ef_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_upd_cpy.at(j), 1e-12)
                    << "Copy failed for synchronous update";
            }
        }
    }
}



} // namespace stdmMf


int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
