#include "network.hpp"
#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>

#include <njm_cpp/info/project.hpp>

#include <njm_cpp/thread/pool.hpp>

#include <thread>

using namespace stdmMf;

void gen_tuples(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfShieldState> > & model,
        const uint32_t & num_starts,
        const uint32_t & num_points,
        njm::data::Entry * const entry) {

    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);

    System<InfShieldState> s(network, model);
    s.rng(rng);

    MyopicAgent<InfShieldState> myopic_agent(network, model->clone());
    ProximalAgent<InfShieldState> proximal_agent(network);
    RandomAgent<InfShieldState> random_agent(network);

    *entry << "rep,"
           << "time,"
           << "node,"
           << "curr_inf,"
           << "curr_shield,"
           << "trt,"
           << "next_inf,"
           << "next_shield\n";

    for (uint32_t i = 0; i < num_starts; ++i) {
        rng->seed(i);

        s.start();
        for (uint32_t j = 0; j < num_points; ++j) {
            const InfShieldState curr_state(s.state());

            boost::dynamic_bitset<> trt_bits;
            const auto draw = rng->rint(0, 3);
            if (draw == 0) {
                trt_bits = myopic_agent.apply_trt(s.state(), s.history());
            } else if (draw == 1) {
                trt_bits = proximal_agent.apply_trt(s.state(), s.history());
            } else if (draw == 2) {
                trt_bits = random_agent.apply_trt(s.state(), s.history());
            }

            s.trt_bits(trt_bits);

            s.turn_clock();

            const InfShieldState next_state(s.state());

            for (uint32_t n = 0; n < network->size(); ++n) {
                *entry << i << "," << j << "," << n << ","
                       << static_cast<uint32_t>(curr_state.inf_bits.test(n))
                       << "," << curr_state.shield.at(n) << ","
                       << static_cast<uint32_t>(trt_bits.test(n)) << ","
                    << static_cast<uint32_t>(next_state.inf_bits.test(n))
                    << "," << next_state.shield.at(n) << "\n";
            }
        }
    }
}


int main(int argc, char *argv[]) {
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(2);
        init.set_dim_y(2);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    typedef std::vector<std::shared_ptr<Model<InfShieldState> > > ModelVec;
    std::vector<std::pair<std::string, ModelVec> > models;

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

        {
            std::vector<std::shared_ptr<Model<InfShieldState> > > model_vec;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                std::shared_ptr<Model<InfShieldState> > m(
                        new InfShieldStateNoImNoSoModel(networks.at(i)));
                m->par(par);
                model_vec.push_back(std::move(m));
            }

            models.emplace_back("noim", std::move(model_vec));
        }

        {
            std::vector<std::shared_ptr<Model<InfShieldState> > > model_vec;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                std::shared_ptr<Model<InfShieldState> > m(
                        new InfShieldStatePosImNoSoModel(networks.at(i)));
                m->par(par);
                model_vec.push_back(std::move(m));
            }

            models.emplace_back("posim", std::move(model_vec));
        }
    }

    const uint32_t num_starts(10000);
    const uint32_t num_points(100);

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");

    njm::thread::Pool p(std::thread::hardware_concurrency());

    for (uint32_t i = 0; i < networks.size(); ++i) {
        for (uint32_t j = 0; j < models.size(); ++j) {
            njm::data::Entry * new_entry(tk.entry(
                            "tuples_" + networks.at(i)->kind() + "_"
                            + models.at(j).first + ".txt"));

            p.service().post([=]() {
                gen_tuples(networks.at(i)->clone(),
                        models.at(j).second.at(i)->clone(),
                        num_starts, num_points, new_entry);
            });
        }
    }

    p.join();

    tk.finished();

    return 0;
}
