#include <memory>
#include <chrono>
#include <fstream>
#include <thread>

#include <njm_cpp/data/result.hpp>
#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/optim/simPerturb.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/tools/experiment.hpp>
#include <njm_cpp/tools/progress.hpp>
#include <njm_cpp/info/project.hpp>

#include "system.hpp"
#include "agent.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "networkRunSymFeatures.hpp"
#include "sweepAgent.hpp"
#include "randomAgent.hpp"
#include "proximalAgent.hpp"
#include "epsAgent.hpp"
#include "objFns.hpp"



using namespace stdmMf;

using njm::data::Result;
using njm::tools::Rng;
using njm::tools::Experiment;

void run_brmin(const std::shared_ptr<Result<std::pair<double, double> > > & r,
        const uint32_t & seed,
        const uint32_t & num_reps,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size,
        const uint32_t & run_length,
        const bool & do_sweep,
        const bool & gs_step,
        const bool & sq_total_br) {
    std::shared_ptr<Rng> rng(new Rng);
    rng->seed(seed);

    // setup network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> net = Network::gen_network(init);

    // model
    std::shared_ptr<Model<InfShieldState> > mod(
            new InfShieldStateNoImNoSoModel(net));
    mod->rng(rng);

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

    // system
    System<InfShieldState> s(net, mod);
    s.rng(rng);

    // features
    std::shared_ptr<Features<InfShieldState> > features(
            new NetworkRunSymFeatures<InfShieldState>(net, run_length));

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
    for (uint32_t i = 0; i < num_reps; ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition<InfShieldState> > all_history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));

    BrMinSimPerturbAgent<InfShieldState> brAgent(net, features, c, t, a, b, ell,
            min_step_size, do_sweep, gs_step, sq_total_br);
    brAgent.rng(rng);

    // start timer
    const std::chrono::time_point<std::chrono::steady_clock> tick =
        std::chrono::steady_clock::now();

    // train
    const std::vector<double> par = brAgent.train(all_history,
            std::vector<double>(features->num_features(), 0.));

    // end timer
    const std::chrono::time_point<std::chrono::steady_clock> tock =
        std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = tock - tick;


    SweepAgent<InfShieldState> agent(net, features, par, 2, do_sweep);
    agent.rng(rng);

    double val = 0.0;
    for (uint32_t i = 0; i < 50; ++i) {
        s.reset();
        s.start();

        val += runner(&s, &agent, 100, 1.0);
    }
    val /= 50.;

    r->set(std::pair<double, double>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                            elapsed).count(),
                    val));
}





int main(int argc, char *argv[]) {
    Experiment e;


    {
        // best
        Experiment::FactorGroup * g = e.add_group();


        g->add_factor(std::vector<int>(
            {5, 10, 50, 100, 500, 1000, 10000})); // num_reps
        g->add_factor(std::vector<double>({0.1})); // c
        g->add_factor(std::vector<double>({0.1})); // t
        g->add_factor(std::vector<double>({1.41})); // a
        g->add_factor(std::vector<double>({1})); // b
        g->add_factor(std::vector<double>({0.85})); // ell
        g->add_factor(std::vector<double>({0.00715})); // min_step_size
        g->add_factor(std::vector<int>({1, 2, 3})); // run_length
        g->add_factor(std::vector<bool>({false, true})); // do_sweeps
        g->add_factor(std::vector<bool>({false, true})); // gs_step
        g->add_factor(std::vector<bool>({false, true})); // sq_total_br
    }


    // {
    //     Experiment::FactorGroup * g = e.add_group();

    //     g->add_factor(std::vector<double>({0.2, 0.1, 0.05})); // c
    //     g->add_factor(std::vector<double>({0.05, 0.1, 0.15, 0.35})); // t
    //     g->add_factor(std::vector<double>({1.41e-0})); // a
    //     g->add_factor(std::vector<double>({1})); // b
    //     g->add_factor(std::vector<double>({0.85})); // ell
    //     g->add_factor(std::vector<double>(
    //                     {2.79e-2, 1.29e-2, 7.15e-3})); // min_step_size
    //     g->add_factor(std::vector<int>({1, 2})); // run_length
    //     g->add_factor(std::vector<bool>({false, true})); // do_sweeps
    //     g->add_factor(std::vector<bool>({false, true})); // gs_step
    //     g->add_factor(std::vector<bool>({false, true})); // sq_total_br
    // }

    // {
    //     Experiment::FactorGroup * g = e.add_group();

    //     g->add_factor(std::vector<double>({0.2, 0.1, 0.05})); // c
    //     g->add_factor(std::vector<double>({0.05, 0.1, 0.15, 0.35})); // t
    //     g->add_factor(std::vector<double>({1.41e-1})); // a
    //     g->add_factor(std::vector<double>({1})); // b
    //     g->add_factor(std::vector<double>({0.85})); // ell
    //     g->add_factor(std::vector<double>(
    //         {2.79e-3, 1.29e-3, 7.15e-4})); // min_step_size
    //     g->add_factor(std::vector<int>({1, 2})); // run_length
    //     g->add_factor(std::vector<bool>({false, true})); // do_sweeps
    //     g->add_factor(std::vector<bool>({false, true})); // gs_step
    //     g->add_factor(std::vector<bool>({false, true})); // sq_total_br
    // }

    njm::thread::Pool p(std::thread::hardware_concurrency());

    std::vector<std::shared_ptr<Result<std::pair<double, double> > > > results;
    std::vector<Experiment::Factor> factors;
    std::vector<uint32_t> factors_level;
    std::vector<uint32_t> rep_number;

    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(&std::cout));

    e.start();
    uint32_t level_num = 0;
    uint32_t num_jobs = 0;
    do {
        const Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 50; ++rep) {
            uint32_t i = 0;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const uint32_t num_reps = static_cast<uint32_t>(
                    f.at(i++).val.int_val);
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double c = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double t = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double a = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double b = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double ell = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double min_step_size = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const uint32_t run_length = static_cast<uint32_t>(
                    f.at(i++).val.int_val);
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_bool);
            const bool do_sweep = f.at(i++).val.bool_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_bool);
            const bool gs_step = f.at(i++).val.bool_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_bool);
            const bool sq_total_br = f.at(i++).val.bool_val;

            // check number of factors
            CHECK_EQ(i, f.size());

            std::shared_ptr<Result<std::pair<double, double> > >
                r(new Result<std::pair<double, double> >);
            results.push_back(r);
            factors.push_back(f);
            rep_number.push_back(rep);
            factors_level.push_back(level_num);
            p.service().post([=]() {
                run_brmin(r, rep, num_reps, c, t, a, b, ell, min_step_size,
                                run_length, do_sweep, gs_step, sq_total_br);
                        progress->update();
                    });

            ++num_jobs;
        }

        ++level_num;
    } while (e.next());

    progress->total(num_jobs);

    p.join();
    progress->done();

    CHECK_EQ(factors.size(), results.size());
    CHECK_EQ(factors.size(), factors_level.size());
    CHECK_EQ(factors.size(), rep_number.size());
    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");
    njm::data::Entry & entry = tk.entry("brMinExperiment_results.txt");
    entry << "level_num, rep_num, elapsed, value, num_reps, c, t, a, b, ell, "
          << "min_step_size, run_length, do_sweep, gs_step, sq_total_br\n";
    for (uint32_t i = 0; i < results.size(); ++i) {
        const std::pair<double, double> result_i = results.at(i)->get();
        entry << factors_level.at(i) << ", " << rep_number.at(i) << ", "
              << result_i.first << ", " << result_i.second;
        Experiment::Factor f = factors.at(i);
        for (uint32_t j = 0; j < f.size(); ++j) {
            entry << ", " << f.at(j);
        }
        entry << "\n";
    }

    return 0;
}
