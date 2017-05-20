#include "network.hpp"
#include "model.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "system.hpp"
#include "features.hpp"
#include "networkRunSymFeatures.hpp"
#include "sweepAgent.hpp"
#include "brMinSimPerturbAgent.hpp"

#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "epsAgent.hpp"

#include "objFns.hpp"

#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/tools/experiment.hpp>
#include <njm_cpp/tools/progress.hpp>
#include <njm_cpp/info/project.hpp>

#include <algorithm>

#include <thread>

#include <glog/logging.h>


using namespace stdmMf;

std::vector<double> proj_a_on_b(const std::vector<double> & a,
        const std::vector<double> & b) {
    CHECK_EQ(a.size(), b.size());
    const double numer = njm::linalg::dot_a_and_b(a, b);
    const double denom = njm::linalg::dot_a_and_b(b, b);
    std::vector<double> proj(b);
    njm::linalg::mult_b_to_a(proj, numer / denom);
    return proj;
}


std::vector<std::vector<double> > get_orth_vectors(const uint32_t & num_vec,
        const uint32_t & vec_len) {
    njm::tools::Rng rng;
    rng.seed(0);

    // generate
    std::vector<std::vector<double> > orth_vectors(num_vec);
    for (uint32_t i = 0; i < num_vec; ++i) {
        // fill using standard normal
        std::vector<double> & vec_i(orth_vectors.at(i));
        vec_i.resize(vec_len);
        std::generate(vec_i.begin(), vec_i.end(),
                [&rng]() {
                    return rng.rnorm_01();
                });

        // orthogonalize against previous vectors
        std::vector<double> gs_adjust(vec_len, 0.0);
        for (uint32_t j = 0; j < i; ++j) {
            njm::linalg::add_b_to_a(gs_adjust,
                    proj_a_on_b(vec_i, orth_vectors.at(j)));
        }

        njm::linalg::mult_b_to_a(gs_adjust, -1.0);

        njm::linalg::add_b_to_a(vec_i, gs_adjust);

        njm::linalg::mult_b_to_a(vec_i, 1.0 / njm::linalg::l2_norm(vec_i));
    }


    return orth_vectors;
}



void generate_jitters(const uint32_t & seed,
        const uint32_t & run_length,
        const uint32_t & num_obs_a,
        const uint32_t & num_obs_b,
        const bool & gs_step,
        const bool & sq_total_br,
        const std::vector<double> & eps_values,
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfShieldState> > & model,
        njm::data::Entry * entry) {
    CHECK_GT(num_obs_b, num_obs_a);

    // set seed
    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng());
    rng->seed(seed);

    // system
    System<InfShieldState> s(network, model);
    s.rng(rng);

    // features
    std::shared_ptr<Features<InfShieldState> > features(
            new NetworkRunSymFeatures<InfShieldState>(network, run_length));

    // eps agent
    std::shared_ptr<ProximalAgent<InfShieldState> > pa(
            new ProximalAgent<InfShieldState>(network));
    pa->rng(rng);
    std::shared_ptr<RandomAgent<InfShieldState> > ra(
            new RandomAgent<InfShieldState>(network));
    ra->rng(rng);
    EpsAgent<InfShieldState> ea(network, pa, ra, 0.2);
    ea.rng(rng);


    // set initial infections
    s.start();
    // simulate history for obs_a
    for (uint32_t i = 0; i < num_obs_a; ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition<InfShieldState> > history_a(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));
    CHECK_EQ(history_a.size(), num_obs_a);

    // simulate history for obs_a
    for (uint32_t i = 0; i < (num_obs_b - num_obs_a); ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition<InfShieldState> > history_b(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));
    CHECK_EQ(history_b.size(), num_obs_b);

    // setup agent
    BrMinSimPerturbAgent<InfShieldState> brAgent(network, features, model,
            1e-1, 0.1, 1.41, 1, 0.85, 0.007150, false, gs_step, sq_total_br, 0,
            0, 0, 0, 0);
    brAgent.rng(rng);


    // train using first num_obs_a observations
    const std::vector<double> par_a = brAgent.train(history_a,
            std::vector<double>(features->num_features(), 0.0));

    // train using first num_obs_a observations
    const std::vector<double> par_b = brAgent.train(history_b,
            std::vector<double>(features->num_features(), 0.0));

    // train using first num_obs_a observations
    const std::vector<double> par_b_warm = brAgent.train(history_b, par_a);


    // generate orthonogal vectors
    const std::vector<std::vector<double> > orth_vectors(
            get_orth_vectors(features->num_features(),
                    features->num_features()));

    // calculate objective functionn
    for (uint32_t i = 0; i < eps_values.size(); ++i) {
        const double & eps = eps_values.at(i);
        for (uint32_t j = 0; j < orth_vectors.size(); ++j) {
            // prepare parameter vectors
            const std::vector<double> eps_orth_vec(
                    njm::linalg::mult_a_and_b(orth_vectors.at(j), eps));

            const double norm_a = njm::linalg::l2_norm(par_a);
            const std::vector<double> par_a_eps(
                    njm::linalg::add_a_and_b(
                            njm::linalg::mult_a_and_b(par_a, 1.0 - eps),
                            njm::linalg::mult_a_and_b(eps_orth_vec, norm_a)));

            const double norm_b = njm::linalg::l2_norm(par_b);
            const std::vector<double> par_b_eps(
                    njm::linalg::add_a_and_b(
                            njm::linalg::mult_a_and_b(par_b, 1.0 - eps),
                            njm::linalg::mult_a_and_b(eps_orth_vec, norm_b)));

            const double norm_b_warm = njm::linalg::l2_norm(par_b_warm);
            const std::vector<double> par_b_warm_eps(
                    njm::linalg::add_a_and_b(
                            njm::linalg::mult_a_and_b(par_b_warm, 1.0 - eps),
                            njm::linalg::mult_a_and_b(eps_orth_vec,
                                    norm_b_warm)));


            SweepAgent<InfShieldState> agent_a(network, features, par_a_eps,
                    njm::linalg::dot_a_and_b, 2, false);
            agent_a.rng(rng);
            auto q_fn_a = [&](const InfShieldState & state_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                              return njm::linalg::dot_a_and_b(par_a_eps,
                                      features->get_features(state_t,
                                              trt_bits_t));
                          };
            const double br_a_on_a = sq_bellman_residual<InfShieldState>(
                    history_a, & agent_a, 0.9, q_fn_a, q_fn_a);

            const double br_a_on_b = sq_bellman_residual<InfShieldState>(
                    history_b, & agent_a, 0.9, q_fn_a, q_fn_a);



            SweepAgent<InfShieldState> agent_b(network, features, par_b_eps,
                    njm::linalg::dot_a_and_b, 2, false);
            agent_b.rng(rng);
            auto q_fn_b = [&](const InfShieldState & state_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                              return njm::linalg::dot_a_and_b(par_b_eps,
                                      features->get_features(state_t,
                                              trt_bits_t));
                          };
            const double br_b_on_a = sq_bellman_residual<InfShieldState>(
                    history_a, & agent_b, 0.9, q_fn_b, q_fn_b);

            const double br_b_on_b = sq_bellman_residual<InfShieldState>(
                    history_b, & agent_b, 0.9, q_fn_b, q_fn_b);



            SweepAgent<InfShieldState> agent_b_warm(network, features,
                    par_b_warm_eps, njm::linalg::dot_a_and_b, 2, false);
            agent_b_warm.rng(rng);
            auto q_fn_b_warm = [&](const InfShieldState & state_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                                   return njm::linalg::dot_a_and_b(
                                           par_b_warm_eps,
                                           features->get_features(state_t,
                                                   trt_bits_t));
                               };
            const double br_b_warm_on_a = sq_bellman_residual<InfShieldState>(
                    history_a, & agent_b, 0.9, q_fn_b_warm, q_fn_b_warm);

            const double br_b_warm_on_b = sq_bellman_residual<InfShieldState>(
                    history_b, & agent_b, 0.9, q_fn_b_warm, q_fn_b_warm);


            (*entry) << seed << ", " << run_length << ", " << gs_step << ", "
                     << sq_total_br << ", " << eps << ", " << j << ", "
                     << br_a_on_a << ", " << br_a_on_b << ", "
                     << br_b_on_a << ", " << br_b_on_b << ", "
                     << br_b_warm_on_a << ", " << br_b_warm_on_b << "\n";
        }

    }
}


int main(int argc, char *argv[]) {
    // setup networks
    std::shared_ptr<Network> network;
    { // network 1
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        network = Network::gen_network(init);
    }

    std::shared_ptr<Model<InfShieldState> > model(
            new InfShieldStateNoImNoSoModel(
                    network));

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

        // set parameters
        model->par(par);
    }

    njm::tools::Experiment e;

    njm::thread::Pool p(std::thread::hardware_concurrency());

    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(&std::cout));

    {
        njm::tools::Experiment::FactorGroup * g = e.add_group();

        g->add_factor(std::vector<int>({1, 2, 3})); // run_length
        g->add_factor(std::vector<bool>({false, true})); // gs_step
        g->add_factor(std::vector<bool>({false, true})); // sq_total_br
    }


    std::shared_ptr<njm::data::TrapperKeeper> tk(
            new njm::data::TrapperKeeper(argv[0],
                    njm::info::project::PROJECT_ROOT_DIR + "/data"));

    const uint32_t num_obs_a = 10;
    const uint32_t num_obs_b = 100;
    const std::vector<double> eps_values({0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9, 1.0});

    njm::data::Entry * entry = tk->entry("inspectBrFn_results.csv");
    *entry << "seed, run_length, gs_step, sq_total_br, eps, orth_vector, "
           << "br_" << num_obs_a << "_on_" << num_obs_a << ", "
           << "br_" << num_obs_a << "_on_" << num_obs_b << ", "
           << "br_" << num_obs_b << "_on_" << num_obs_a << ", "
           << "br_" << num_obs_b << "_on_" << num_obs_b << ", "
           << "br_" << num_obs_b << "_warm_on_" << num_obs_a << ", "
           << "br_" << num_obs_b << "_warm_on_" << num_obs_b << "\n";

    e.start();

    uint32_t num_jobs = 0;
    do {
        const njm::tools::Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 50; ++rep) {
            uint32_t i = 0;
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_int);
            const uint32_t run_length = static_cast<uint32_t>(
                    f.at(i++).val.int_val);
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_bool);
            const bool gs_step = f.at(i++).val.bool_val;
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_bool);
            const bool sq_total_br = f.at(i++).val.bool_val;

            // check number of factors
            CHECK_EQ(i, f.size());

            njm::data::Entry * new_entry = tk->entry(
                    "inspectBrFn_results.csv");

            p.service().post([=]() {
                generate_jitters(rep, run_length, num_obs_a, num_obs_b, gs_step,
                        sq_total_br, eps_values, network->clone(),
                        model->clone(), new_entry);

                progress->update();
            });

            ++num_jobs;
        }

    } while (e.next());

    progress->total(num_jobs);

    p.join();
    progress->done();

    tk->finished();

    return 0;
}
