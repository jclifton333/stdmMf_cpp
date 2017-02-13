#include <memory>
#include <chrono>
#include <fstream>
#include <thread>

#include <njm_cpp/data/result.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/optim/simPerturb.hpp>
#include <njm_cpp/tools/experiment.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/tools/progress.hpp>
#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/info/project.hpp>

#include "system.hpp"
#include "agent.hpp"
#include "noCovEdgeModel.hpp"
#include "networkRunSymFeatures.hpp"
#include "sweepAgent.hpp"
#include "randomAgent.hpp"
#include "proximalAgent.hpp"
#include "myopicAgent.hpp"
#include "epsAgent.hpp"
#include "objFns.hpp"


using namespace stdmMf;

using njm::tools::Rng;
using njm::data::Result;
using njm::tools::Progress;
using njm::tools::Experiment;

void run_adapt(const std::shared_ptr<Result<std::pair<double, double> > > & r,
        const uint32_t & seed,
        const uint32_t & path_len,
        const uint32_t & num_reps_vfn,
        const double & c_vfn,
        const double & t_vfn,
        const double & a_vfn,
        const double & b_vfn,
        const double & ell_vfn,
        const double & min_step_size_vfn,
        const double & c_br,
        const double & t_br,
        const double & a_br,
        const double & b_br,
        const double & ell_br,
        const double & min_step_size_br,
        const uint32_t & step_scale_br,
        const double & gamma_br,
        const std::shared_ptr<Progress<std::ostream> > & progress) {
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


        const std::vector<double> par =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             trt_act_rec,
             trt_pre_inf};

        // mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});
        mod->par(par);
    }


    std::vector<double> optim_par;

    // value function maximization
    {
        // features
        std::shared_ptr<Features> features(new NetworkRunSymFeatures(net,
                        path_len));

        auto min_fn = [&](const std::vector<double> & par,
                void * const data) {
            SweepAgent agent(net, features, par, 2, true);
            agent.rng(rng);
            System s(net, mod);
            s.rng(rng);
            double val = 0.0;
            for (uint32_t i = 0; i < num_reps_vfn; ++i) {
                s.cleanse();
                s.wipe_trt();
                s.erase_history();
                s.start();

                val += runner(&s, &agent, 20, 1.0);
            }
            val /= num_reps_vfn;

            // return negative since optim minimizes functions
            return -val;
        };

        njm::optim::SimPerturb sp(min_fn, std::vector<double>(
                        features->num_features(), 0.),
                NULL, c_vfn, t_vfn, a_vfn, b_vfn, ell_vfn, min_step_size_vfn);
        sp.rng(rng);

        njm::optim::ErrorCode ec;
        const std::chrono::time_point<std::chrono::high_resolution_clock> tick =
            std::chrono::high_resolution_clock::now();
        do {
            ec = sp.step();
        } while (ec == njm::optim::ErrorCode::CONTINUE);
        const std::chrono::time_point<std::chrono::high_resolution_clock> tock =
            std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double> elapsed = tock - tick;

        CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS);

        optim_par = sp.par();
    }


    // bellman residual minimization
    std::chrono::duration<double> elapsed;
    {
        // system
        System s(net, mod);
        s.rng(rng);

        // features
        std::shared_ptr<Features> features(new NetworkRunSymFeatures(net,
                        path_len));

        // eps agent
        std::shared_ptr<MyopicAgent> ma(new MyopicAgent(net, mod->clone()));
        ma->rng(rng);
        std::shared_ptr<ProximalAgent> pa(new ProximalAgent(net));
        pa->rng(rng);
        EpsAgent ea(net, ma, pa, 0.2);
        ea.rng(rng);


        // set initial infections
        s.start();
        // simulate history
        const uint32_t num_points_for_br = 50;
        for (uint32_t i = 0; i < num_points_for_br; ++i) {
            const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.inf_bits(),
                    s.history());
            s.trt_bits(trt_bits);
            s.turn_clock();
        }

        std::vector<Transition> history(
                Transition::from_sequence(s.history(), s.inf_bits()));
        CHECK_EQ(history.size(), num_points_for_br);


        // function for br min
        auto min_fn = [&](const std::vector<double> & par,
                void * const data) {
            SweepAgent agent(net, features, par, 2, true);
            agent.rng(rng);

            // q function
            auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits,
                    const boost::dynamic_bitset<> & trt_bits) {
                return njm::linalg::dot_a_and_b(par,features->get_features(
                                inf_bits, trt_bits));
            };
            const double br = bellman_residual_sq(history, &agent, gamma_br,
                    q_fn);

            return br;
        };

        njm::optim::SimPerturb sp(min_fn, optim_par,
                NULL, c_br, t_br, a_br, b_br, ell_br, min_step_size_br);
        sp.rng(rng);

        njm::optim::ErrorCode ec;
        const std::chrono::time_point<std::chrono::high_resolution_clock> tick =
            std::chrono::high_resolution_clock::now();

        do {
            ec = sp.step();
        } while (ec == njm::optim::ErrorCode::CONTINUE
                && sp.completed_steps() < (num_points_for_br * step_scale_br));

        const std::chrono::time_point<std::chrono::high_resolution_clock> tock =
            std::chrono::high_resolution_clock::now();

        elapsed = tock - tick;

        CHECK(ec == njm::optim::ErrorCode::SUCCESS ||
                ec == njm::optim::ErrorCode::CONTINUE)
            << ec << std::endl
            << "br tuning paramters: "
            << c_br << ", "
            << t_br << ", "
            << a_br << ", "
            << b_br << ", "
            << ell_br << ", "
            << min_step_size_br << ", "
            << gamma_br << ", "
            << step_scale_br;

        optim_par = sp.par();
    }

    double val = 0.0;
    {
        System s(net, mod);
        s.rng(rng);

        // features
        std::shared_ptr<Features> features(new NetworkRunSymFeatures(net,
                        path_len));

        SweepAgent agent(net, features, optim_par, 2, true);
        agent.rng(rng);

        for (uint32_t i = 0; i < 50; ++i) {
            s.cleanse();
            s.wipe_trt();
            s.erase_history();
            s.start();

            val += runner(&s, &agent, 20, 1.0);
        }
        val /= 50.;
    }

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

    progress->update();
}





int main(int argc, char *argv[]) {
    Experiment e;

    {
        const std::vector<int> path_len_list = {2, 3, 4};

        const std::vector<int> num_reps_vfn_list = {2};
        const std::vector<double> c_vfn_list = {10.0};
        const std::vector<double> t_vfn_list = {0.1};
        const std::vector<double> a_vfn_list = {5.0};
        const std::vector<double> b_vfn_list = {1.0};
        const std::vector<double> ell_vfn_list = {0.4};
        const std::vector<double> min_step_size_vfn_list = {0.7};

        const std::vector<double> c_br_list = {2e-1};
        const std::vector<double> t_br_list = {0.75, 0.875, 1.0};
        const std::vector<double> a_br_list = {1e-3};
        const std::vector<double> b_br_list = {1};
        const std::vector<double> ell_br_list = {1.0, 0.95, 0.85};
        const std::vector<double> min_step_size_br_list = {9.13e-6, 6.473e-6,
                                                           5.072e-6};
        const std::vector<int> step_scale_br_list = {1, 10, 100};
        const std::vector<double> gamma_br_list = {0.9, 0.99};

        Experiment::FactorGroup * g = e.add_group();
        g->add_factor(path_len_list);

        g->add_factor(num_reps_vfn_list);
        g->add_factor(c_vfn_list);
        g->add_factor(t_vfn_list);
        g->add_factor(a_vfn_list);
        g->add_factor(b_vfn_list);
        g->add_factor(ell_vfn_list);
        g->add_factor(min_step_size_vfn_list);

        g->add_factor(c_br_list);
        g->add_factor(t_br_list);
        g->add_factor(a_br_list);
        g->add_factor(b_br_list);
        g->add_factor(ell_br_list);
        g->add_factor(min_step_size_br_list);
        g->add_factor(step_scale_br_list);
        g->add_factor(gamma_br_list);
    }

    {
        const std::vector<int> path_len_list = {2, 3, 4};

        const std::vector<int> num_reps_vfn_list = {2};

        const std::vector<double> c_vfn_list = {10.0};
        const std::vector<double> t_vfn_list = {0.1};
        const std::vector<double> a_vfn_list = {5.0};
        const std::vector<double> b_vfn_list = {1.0};
        const std::vector<double> ell_vfn_list = {0.4};
        const std::vector<double> min_step_size_vfn_list = {0.7};

        const std::vector<double> c_br_list = {2e-1};
        const std::vector<double> t_br_list = {0.75, 0.875, 1.0};
        const std::vector<double> a_br_list = {1e-3, 1.41e-3, 1.8e-3};
        const std::vector<double> b_br_list = {1};
        const std::vector<double> ell_br_list = {1.0, 0.95, 0.85};
        const std::vector<double> min_step_size_br_list = {9.13e-6};
        const std::vector<int> step_scale_br_list = {1, 10, 100};
        const std::vector<double> gamma_br_list = {0.9, 0.99};

        Experiment::FactorGroup * g = e.add_group();
        g->add_factor(path_len_list);

        g->add_factor(num_reps_vfn_list);
        g->add_factor(c_vfn_list);
        g->add_factor(t_vfn_list);
        g->add_factor(a_vfn_list);
        g->add_factor(b_vfn_list);
        g->add_factor(ell_vfn_list);
        g->add_factor(min_step_size_vfn_list);

        g->add_factor(c_br_list);
        g->add_factor(t_br_list);
        g->add_factor(a_br_list);
        g->add_factor(b_br_list);
        g->add_factor(ell_br_list);
        g->add_factor(min_step_size_br_list);
        g->add_factor(step_scale_br_list);
        g->add_factor(gamma_br_list);
    }

    njm::thread::Pool p(std::thread::hardware_concurrency());

    std::vector<std::shared_ptr<Result<std::pair<double, double> > > > results;
    std::vector<Experiment::Factor> factors;
    std::vector<uint32_t> factors_level;
    std::vector<uint32_t> rep_number;

    std::shared_ptr<Progress<std::ostream> > progress(
            new Progress<std::ostream>(0, &std::cout));

    uint32_t num_jobs = 0;

    e.start();
    uint32_t level_num = 0;
    do {
        const Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 50; ++rep) {
            uint32_t i = 0;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const int path_len = f.at(i++).val.int_val;

            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const int num_reps_vfn = f.at(i++).val.int_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double c_vfn = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double t_vfn = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double a_vfn = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double b_vfn = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double ell_vfn = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double min_step_size_vfn = f.at(i++).val.double_val;

            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double c_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double t_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double a_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double b_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double ell_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double min_step_size_br = f.at(i++).val.double_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const int step_scale_br = f.at(i++).val.int_val;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_double);
            const double gamma_br = f.at(i++).val.double_val;

            CHECK_EQ(i, f.size());

            std::shared_ptr<Result<std::pair<double, double> > >
                r(new Result<std::pair<double, double> >);
            results.push_back(r);
            factors.push_back(f);
            rep_number.push_back(rep);
            factors_level.push_back(level_num);
            p.service()->post(std::bind(&run_adapt, r, rep, path_len,
                            num_reps_vfn, c_vfn, t_vfn, a_vfn, b_vfn,
                            ell_vfn, min_step_size_vfn,
                            c_br, t_br, a_br, b_br,
                            ell_br, min_step_size_br,
                            step_scale_br, gamma_br, progress));

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
    njm::data::Entry & results_entry = tk.entry(
            "vfnBrAdaptExperiment_results.txt");
    results_entry
        << "level_num, rep_num, elapsed, value, path_len_vfn, num_reps_vfn, "
        << "c_vfn, t_vfn, a_vfn, b_vfn, ell_vfn, min_step_size_vfn, "
        << "c_br, t_br, a_br, b_br, ell_br, min_step_size_br, step_scale_br, "
        << "gamma_br"
        << "\n";

    for (uint32_t i = 0; i < results.size(); ++i) {
        const std::pair<double, double> result_i = results.at(i)->get();
        results_entry << factors_level.at(i) << ", " << rep_number.at(i) << ", "
                      << result_i.first << ", " << result_i.second;
        Experiment::Factor f = factors.at(i);
        for (uint32_t j = 0; j < f.size(); ++j) {
            results_entry << ", " << f.at(j);
        }
        results_entry << "\n";
    }

    return 0;
}
