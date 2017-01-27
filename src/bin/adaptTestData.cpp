#include "system.hpp"
#include "noCovEdgeModel.hpp"
#include "noCovEdgeOrSoModel.hpp"
#include "noCovEdgeXorSoModel.hpp"
#include "noCovEdgeSepSoModel.hpp"
#include "vfnMaxSimPerturbAgent.hpp"

#include "networkRunSymFeatures.hpp"

#include "result.hpp"

#include "pool.hpp"

#include "trapperKeeper.hpp"


void run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model> & mod_system,
        const std::shared_ptr<Model> & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & num_samples,
        const uint32_t & num_points,
        Entry & entry) {

    // header
    entry << "rep,time,sample,inf,trt,next_inf" << "\n";

    std::shared_ptr<Features> features(
            new NetworkRunSymFeatures(net->clone(), 3));

    for (uint32_t rep = 0; rep < num_reps; ++rep) {
        System s_orig(net->clone(), mod_system->clone());

        s_orig.set_seed(rep);
        VfnMaxSimPerturbAgent vmax_agent(net->clone(),
                features->clone(),
                mod_agents->clone(),
                2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7);
        vmax_agent.set_seed(rep);

        s_orig.start();

        for (uint32_t t = 0; t < num_points; ++t) {
            // rep, sample, time
            entry << rep << "," << t << "," << -1 << ",";

            // starting infection
            std::string bits_str;
            boost::to_string(s_orig.inf_bits(), bits_str);
            entry << bits_str << ",";

            const boost::dynamic_bitset<> trt_bits = vmax_agent.apply_trt(
                    s_orig.inf_bits(), s_orig.history());

            // treatment
            boost::to_string(trt_bits, bits_str);
            entry << bits_str << ",";

            CHECK_EQ(trt_bits.count(), vmax_agent.num_trt());

            s_orig.trt_bits(trt_bits);

            s_orig.turn_clock();

            // final infection
            boost::to_string(s_orig.inf_bits(), bits_str);
            entry << bits_str << "\n";
        }

        // estimate from history
        mod_agents->est_par(s_orig.inf_bits(), s_orig.history());

        // retreive last coefficients
        const std::vector<double> coef = vmax_agent.coef();

        for (uint32_t sample = 0; sample < sum_samples; ++sample) {
            // use agents model for simulating
            System s_sim(net->clone(), mod_agents->clone());

            // construct sweep agent
            SweepAgent sweep_agent(net->clone(), features->clone(),
                    coef, 0, false);

            for (uint32_t t = 0; t < num_points; ++t) {
                // set infection to observed value
                s_sim.inf_bits(s_orig.history().at(t).first);

                const boost::dynamic_bitset<> trt_bits = sweep_agent->apply_trt(
                        s_sim.inf_bits(), s_sim.history());

                CHECK_EQ(trt_bits.count(), sweep_agent.num_trt());

                s_sim.trt_bits(trt_bits);

                s_sim.turn_clock();
            }

        }
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
    typedef std::pair<std::shared_ptr<Model>,
                      std::shared_ptr<Model> > ModelPair;
    std::vector<std::vector<ModelPair> > models;
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
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
            models.push_back(models_add);
        }
    }





    return 0;
}
