#include "system.hpp"
#include "network.hpp"
#include "noCovEdgeModel.hpp"
#include "noCovEdgeOrSoModel.hpp"
#include "noCovEdgeXorSoModel.hpp"
#include "noCovEdgeSepSoModel.hpp"
#include "vfnMaxSimPerturbAgent.hpp"

#include "networkRunSymFeatures.hpp"

#include "result.hpp"

#include "pool.hpp"

#include <thread>

#include "trapperKeeper.hpp"

#include "projectInfo.hpp"

#include "adaptTestData.pb.h"

#include <google/protobuf/text_format.h>

#include "progress.hpp"

using namespace stdmMf;


void run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model> & mod_system,
        const std::shared_ptr<Model> & mod_agents,
        const uint32_t & rep,
        const uint32_t & num_points,
        Observation * const obs,
        const std::shared_ptr<Progress<std::ostream> > & progress) {
    std::shared_ptr<Rng> rng(new Rng);
    rng->set_seed(rep);

    std::shared_ptr<Features> features(
            new NetworkRunSymFeatures(net->clone(), 3));

    System s_orig(net->clone(), mod_system->clone());

    s_orig.set_rng(rng);
    VfnMaxSimPerturbAgent vmax_agent(net->clone(),
            features->clone(),
            mod_agents->clone(),
            2, num_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    vmax_agent.set_rng(rng);

    obs->set_num_points(num_points);

    std::string bits_str;

    s_orig.start();
    for (uint32_t t = 0; t < num_points; ++t) {
        State * state = obs->add_state();

        // starting infection
        boost::to_string(s_orig.inf_bits(), bits_str);
        state->set_inf_bits(bits_str);

        const boost::dynamic_bitset<> trt_bits = vmax_agent.apply_trt(
                s_orig.inf_bits(), s_orig.history());

        // treatment
        boost::to_string(trt_bits, bits_str);
        state->set_trt_bits(bits_str);

        CHECK_EQ(trt_bits.count(), vmax_agent.num_trt());

        s_orig.trt_bits(trt_bits);

        s_orig.turn_clock();
    }

    // final infection
    State * state = obs->add_state();

    boost::to_string(s_orig.inf_bits(), bits_str);
    state->set_inf_bits(bits_str);

    boost::to_string(s_orig.trt_bits(), bits_str);
    state->set_trt_bits(bits_str);

    progress->update();
}



int main(int argc, char *argv[]) {
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // network 2
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    // double vector since model depends on network
    typedef std::pair<std::shared_ptr<Model>,
                      std::shared_ptr<Model> > ModelPair;
    std::vector<std::pair<std::string,
                          std::vector<ModelPair> > > models;
    { // models
        // latent infections
        const double prob_inf_latent = 0.01;
        const double intcp_inf_latent =
            std::log(1. / (1. - prob_inf_latent) - 1);

        // neighbor infections
        const double prob_inf = 0.5;
        const uint32_t prob_num_neigh = 3;
        const double intcp_inf =
            std::log(std::pow(1. - prob_inf, -1. / prob_num_neigh) - 1.);

        const double trt_act_inf =
            std::log(std::pow(1. - prob_inf * 0.25, -1. / prob_num_neigh) - 1.)
            - intcp_inf;

        const double trt_pre_inf =
            std::log(std::pow(1. - prob_inf * 0.75, -1. / prob_num_neigh) - 1.)
            - intcp_inf;

        // recovery
        const double prob_rec = 0.25;
        const double intcp_rec = std::log(1. / (1. - prob_rec) - 1.);
        const double trt_act_rec =
            std::log(1. / ((1. - prob_rec) * 0.5) - 1.) - intcp_rec;


        std::vector<double> par =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             trt_act_rec,
             trt_pre_inf};

        std::vector<double> par_sep =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             -trt_act_inf,
             trt_act_rec,
             -trt_act_rec,
             trt_pre_inf,
             -trt_pre_inf};

        { // Correct: No So,  Postulated: No So
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_no_no", models_add));
        }

        { // Correct: OrSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_or_or", models_add));
        }

        { // Correct: XorSo,  Postulated: XorSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_xor_xor", models_add));
        }

        { // Correct: SepSo,  Postulated: SepSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par_sep);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_sep_sep", models_add));
        }

        { // Correct: SepSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_sep_or", models_add));
        }

        { // Correct: SepSo,  Postulated: XorSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_sep_xor", models_add));
        }

        { // Correct: SepSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_sep_no", models_add));
        }

        { // Correct: No So,  Postulated: SepSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par_sep);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_no_sep", models_add));
        }

        { // Correct: XorSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_xor_or", models_add));
        }

        { // Correct: XorSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_xor_no", models_add));
        }

        { // Correct: No So,  Postulated: XorSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_no_xor", models_add));
        }

        { // Correct: OrSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_or_no", models_add));
        }

        { // Correct: No So,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("model_no_or", models_add));
        }
    }

    Pool pool(std::thread::hardware_concurrency());

    std::shared_ptr<TrapperKeeper> tp(new TrapperKeeper(argv[0],
                    PROJECT_ROOT_DIR + "/data"));

    const uint32_t num_reps = 100;
    const uint32_t num_points = 50;

    AdaptData ad;

    std::shared_ptr<Progress<std::ostream> > progress(
            new Progress<std::ostream>(
                    networks.size() * models.size() * num_reps, &std::cout));

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);
        const std::string net_name = net->kind();

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).second.at(i));
            const std::string mod_name = models.at(j).first;

            SimData * sd = ad.add_sims();

            sd->set_true_model(mod_name);
            sd->set_network(net_name);

            for (uint32_t rep = 0; rep < num_reps; ++rep) {
                Observation * obs = sd->add_rep();
                pool.service()->post([=](){
                            run(net, mp.first, mp.second, rep,
                                    num_points, obs, progress);
                        });
            }
        }
    }

    pool.join();

    progress->done();


    std::string adapt_data_str;
    google::protobuf::TextFormat::PrintToString(ad, &adapt_data_str);
    Entry & entry = tp->entry("adapt_data.txt");
    entry << adapt_data_str;


    tp->finished();

    return 0;
}
