#include <memory>
#include <chrono>
#include <fstream>
#include <thread>
#include "result.hpp"
#include "pool.hpp"
#include "system.hpp"
#include "agent.hpp"
#include "noCovEdgeModel.hpp"
#include "networkRunFeatures.hpp"
#include "sweepAgent.hpp"
#include "randomAgent.hpp"
#include "proximalAgent.hpp"
#include "epsAgent.hpp"
#include "objFns.hpp"
#include "simPerturb.hpp"
#include "experiment.hpp"
#include "utilities.hpp"


#include "trapperKeeper.hpp"
#include "projectInfo.hpp"
#include "progress.hpp"

using namespace stdmMf;


void run_brmin(const std::shared_ptr<Result<std::pair<double, double> > > & r,
        const uint32_t & seed,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size) {
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
    std::shared_ptr<Model> mod(new NoCovEdgeModel(net));

    mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});

    // system
    System s(net, mod);
    s.rng(rng);

    // features
    std::shared_ptr<Features> features(new NetworkRunFeatures(net, 4));

    // eps agent
    std::shared_ptr<ProximalAgent> pa(new ProximalAgent(net));
    pa->rng(rng);
    std::shared_ptr<RandomAgent> ra(new RandomAgent(net));
    ra->rng(rng);
    EpsAgent ea(net, pa, ra, 0.2);
    ea.rng(rng);


    // set initial infections
    s.start();
    // simulate history
    for (uint32_t i = 0; i < 500; ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.inf_bits(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition> history(
            Transition::from_sequence(s.history(), s.inf_bits()));


    auto min_fn = [&](const std::vector<double> & par,
            void * const data) {
        SweepAgent agent(net, features, par, 2, true);
        agent.rng(rng);

        // q function
        auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits) {
            return dot_a_and_b(par,features->get_features(inf_bits, trt_bits));
        };
        const double br = bellman_residual_sq(history, &agent, 0.9, q_fn);

        return br;
    };

    SimPerturb sp(min_fn, std::vector<double>(
                    features->num_features(), 0.),
            NULL, c, t, a, b, ell, min_step_size);
    sp.rng(rng);

    Optim::ErrorCode ec;
    const std::chrono::time_point<std::chrono::high_resolution_clock> tick =
        std::chrono::high_resolution_clock::now();

    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    const std::chrono::time_point<std::chrono::high_resolution_clock> tock =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = tock - tick;

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS)
        << "Failed with: c = " << c << ", t = " << t
        << ", a = " << a << ", b = " << b << ", ell = " << ell
        << ", min_step_size = " << min_step_size << ", completed_steps = "
        << sp.completed_steps();

    const std::vector<double> par = sp.par();

    SweepAgent agent(net, features, par, 2, true);
    agent.rng(rng);

    double val = 0.0;
    for (uint32_t i = 0; i < 50; ++i) {
        s.cleanse();
        s.wipe_trt();
        s.erase_history();
        s.start();

        val += runner(&s, &agent, 20, 1.0);
    }
    val /= 50.;

    // // q function
    // auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits,
    //         const boost::dynamic_bitset<> & trt_bits) {
    //     return dot_a_and_b(par,features->get_features(inf_bits, trt_bits));
    // };
    // const double br = bellman_residual_sq(history, &agent, 0.9, q_fn);

    r->set(std::pair<double, double>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                            elapsed).count(),
                    val));
}





int main(int argc, char *argv[]) {
    Experiment e;
    Experiment::FactorGroup * g_best = e.add_group();
    g_best->add_factor(std::vector<double>({1e-1}));
    g_best->add_factor(std::vector<double>({1.0}));
    g_best->add_factor(std::vector<double>({1e-3}));
    g_best->add_factor(std::vector<double>({1}));
    g_best->add_factor(std::vector<double>({0.85}));
    g_best->add_factor(std::vector<double>({1e-5}));

    {
        const std::vector<double> c_list = {2e-1};
        const std::vector<double> t_list = {0.75, 0.875, 1.0};
        const std::vector<double> a_list = {1e-3};
        const std::vector<double> b_list = {1};
        const std::vector<double> ell_list = {1.0, 0.95, 0.85};
        const std::vector<double> min_step_size_list = {9.13e-6, 6.473e-6,
                                                        5.072e-6};

        Experiment::FactorGroup * g = e.add_group();
        g->add_factor(c_list);
        g->add_factor(t_list);
        g->add_factor(a_list);
        g->add_factor(b_list);
        g->add_factor(ell_list);
        g->add_factor(min_step_size_list);
    }

    {
        const std::vector<double> c_list = {2e-1};
        const std::vector<double> t_list = {0.75, 0.875, 1.0};
        const std::vector<double> a_list = {1e-3, 1.41e-3, 1.8e-3};
        const std::vector<double> b_list = {1};
        const std::vector<double> ell_list = {1.0, 0.95, 0.85};
        const std::vector<double> min_step_size_list = {9.13e-6};

        Experiment::FactorGroup * g = e.add_group();
        g->add_factor(c_list);
        g->add_factor(t_list);
        g->add_factor(a_list);
        g->add_factor(b_list);
        g->add_factor(ell_list);
        g->add_factor(min_step_size_list);
    }

    Pool p(std::thread::hardware_concurrency());

    std::vector<std::shared_ptr<Result<std::pair<double, double> > > > results;
    std::vector<Experiment::Factor> factors;
    std::vector<uint32_t> factors_level;
    std::vector<uint32_t> rep_number;

    std::shared_ptr<Progress<std::ostream> > progress(
            new Progress<std::ostream>(&std::cout));

    e.start();
    uint32_t level_num = 0;
    uint32_t num_jobs = 0;
    do {
        const Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 50; ++rep) {
            uint32_t i = 0;
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

            std::shared_ptr<Result<std::pair<double, double> > >
                r(new Result<std::pair<double, double> >);
            results.push_back(r);
            factors.push_back(f);
            rep_number.push_back(rep);
            factors_level.push_back(level_num);
            p.service()->post([=]() {
                        run_brmin(r, rep, c, t, a, b, ell, min_step_size);
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
    TrapperKeeper tk(argv[0], PROJECT_ROOT_DIR + "/data");
    Entry & entry = tk.entry("brMinExperiment_results.txt");
    entry << "level_num, rep_num, elapsed, value, c, t, a, b, ell, "
          << "min_step_size\n";
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
