#include "system.hpp"
#include "network.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "vfnMaxSimPerturbAgent.hpp"

#include "networkRunSymFeatures.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/progress.hpp>

#include <thread>

#include <glog/logging.h>

#include "sim_data.pb.h"

#include <google/protobuf/text_format.h>

using namespace stdmMf;

using njm::tools::Rng;

void run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & rep,
        const uint32_t & num_points,
        Observation * const obs) {
    std::shared_ptr<Rng> rng(new Rng);
    rng->seed(rep);

    const uint32_t run_length = 2;

    std::shared_ptr<Features<InfShieldState> > features(
            new NetworkRunSymFeatures<InfShieldState> (net->clone(),
                    run_length));

    System<InfShieldState> s_orig(net->clone(), mod_system->clone());

    s_orig.rng(rng);
    VfnMaxSimPerturbAgent<InfShieldState> vmax_agent(net->clone(),
            features->clone(),
            mod_agents->clone(),
            2, num_points, num_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    vmax_agent.rng(rng);

    obs->set_num_points(num_points);

    std::string bits_str;

    s_orig.start();
    for (uint32_t t = 0; t < num_points; ++t) {
        TransitionPB * transition = obs->add_transition();

        // starting infection
        boost::to_string(s_orig.state().inf_bits, bits_str);
        transition->mutable_curr_state()->set_inf_bits(bits_str);
        // starting shield
        std::for_each(s_orig.state().shield.begin(),
                s_orig.state().shield.end(), [&](const double & val) {
                    transition->mutable_curr_state()->add_shield(val);
                });

        const boost::dynamic_bitset<> trt_bits = vmax_agent.apply_trt(
                s_orig.state(), s_orig.history());

        // treatment
        boost::to_string(trt_bits, bits_str);
        transition->set_curr_trt_bits(bits_str);

        CHECK_EQ(trt_bits.count(), vmax_agent.num_trt());

        s_orig.trt_bits(trt_bits);

        s_orig.turn_clock();

        // ending infection
        boost::to_string(s_orig.state().inf_bits, bits_str);
        transition->mutable_next_state()->set_inf_bits(bits_str);
        // ending shield
        std::for_each(s_orig.state().shield.begin(),
                s_orig.state().shield.end(), [&](const double & val) {
                    transition->mutable_next_state()->add_shield(val);
                });
    }
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
    typedef std::pair<std::shared_ptr<Model<InfShieldState> >,
                      std::shared_ptr<Model<InfShieldState> > > ModelPair;
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

        // shield
        const double shield_coef = 0.9;


        std::vector<double> par =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             trt_act_rec,
             trt_pre_inf,
             shield_coef};

        std::vector<double> par_sep =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             -trt_act_inf,
             trt_act_rec,
             -trt_act_rec,
             trt_pre_inf,
             -trt_pre_inf,
             shield_coef};

        { // Correct: NoIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_NoImNoSo_NoImNoSo",
                            models_add));
        }

        { // Correct: PosIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_PosImNoSo_PosImNoSo",
                            models_add));
        }

        { // Correct: PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_PosImNoSo_NoImNoSo",
                            models_add));
        }

        { // Correct: NoIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_NoImNoSo_PosImNoSo",
                            models_add));
        }
    }

    njm::thread::Pool pool(std::thread::hardware_concurrency());

    std::shared_ptr<njm::data::TrapperKeeper> tp(new njm::data::TrapperKeeper(
                    argv[0], njm::info::project::PROJECT_ROOT_DIR + "/data"));

    const uint32_t num_reps = 100;
    const uint32_t num_points = 50;

    SimDataGroup ad;

    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(
                    networks.size() * models.size() * num_reps, &std::cout));

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);
        const std::string net_name = net->kind();

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).second.at(i));
            const std::string mod_name = models.at(j).first;

            SimData * sd = ad.add_sim();

            sd->set_model(mod_name);
            sd->set_network(net_name);

            for (uint32_t rep = 0; rep < num_reps; ++rep) {
                Observation * obs = sd->add_rep();
                pool.service().post([=](){
                            run(net, mp.first, mp.second, rep,
                                    num_points, obs);
                            progress->update();
                        });
            }
        }
    }

    pool.join();

    progress->done();


    std::string adapt_data_str;
    google::protobuf::TextFormat::PrintToString(ad, &adapt_data_str);
    njm::data::Entry * entry = tp->entry("adapt_data.txt");
    *entry << adapt_data_str;


    tp->finished();

    return 0;
}
