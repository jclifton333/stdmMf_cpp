#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "epsAgent.hpp"

#include "networkRunSymFeatures.hpp"

#include "objFns.hpp"

#include <thread>

using namespace stdmMf;

int main(int argc, char *argv[]) {

    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> net(Network::gen_network(init));

    const std::shared_ptr<Model<InfShieldState> > mod(
            new InfShieldStateNoImNoSoModel(net));
    {
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
        mod->par(par);
    }

    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);
    rng->seed(0);

    // system
    System<InfShieldState> s(net, mod);
    s.rng(rng);

    // features
    std::shared_ptr<Features<InfShieldState> > features(
            new NetworkRunSymFeatures<InfShieldState>(net, 2));

    // eps agent
    std::shared_ptr<ProximalAgent<InfShieldState> > pa(
            new ProximalAgent<InfShieldState>(net));
    pa->rng(rng);
    std::shared_ptr<RandomAgent<InfShieldState> > ra(
            new RandomAgent<InfShieldState>(net));
    ra->rng(rng);
    EpsAgent<InfShieldState> ea(net, pa, ra, 0.2);
    ea.rng(rng);

    // set initial infections
    s.start();
    // simulate history
    for (uint32_t i = 0; i < 500; ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition<InfShieldState> > all_history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));

    // br min
    BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
            std::shared_ptr<Features<InfShieldState> >(
                    new NetworkRunSymFeatures<InfShieldState>(net->clone(), 2)),
            mod->clone(),
            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5, true, true, false, 0, 5, 5, 5);
    a.rng(rng);

    a.train(all_history);

    return 0;
}
