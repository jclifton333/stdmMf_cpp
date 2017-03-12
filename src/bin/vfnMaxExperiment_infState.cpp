#include <memory>
#include <chrono>
#include <fstream>
#include <thread>

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/data/result.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/optim/simPerturb.hpp>
#include <njm_cpp/tools/experiment.hpp>

#include "system.hpp"
#include "agent.hpp"
#include "infStateNoSoModel.hpp"
#include "networkRunFeatures.hpp"
#include "sweepAgent.hpp"
#include "objFns.hpp"

using namespace stdmMf;

using njm::data::Result;
using njm::tools::Rng;
using njm::tools::Experiment;

void run_vmax(const std::shared_ptr<Result<std::pair<double, double> > > & r,
        const uint32_t & seed,
        const int & num_reps,
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
    std::shared_ptr<Model<InfState> > mod(new InfStateNoSoModel(net));

    mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});


    // features
    std::shared_ptr<Features<InfState> > features(
            new NetworkRunFeatures<InfState>(net, 4));

    auto min_fn = [&](const std::vector<double> & par,
            const std::vector<double> & par_orig) {
        SweepAgent<InfState> agent(net, features, par, 2, true);
        agent.rng(rng);
        System<InfState> s(net, mod);
        s.rng(rng);
        double val = 0.0;
        for (uint32_t i = 0; i < num_reps; ++i) {
            s.reset();
            s.start();

            val += runner(&s, &agent, 20, 1.0);
        }
        val /= num_reps;

        // return negative since optim minimizes functions
        return -val;
    };

    njm::optim::SimPerturb sp(min_fn, std::vector<double>(
                    features->num_features(), 0.),
            c, t, a, b, ell, min_step_size);
    sp.rng(rng);

    njm::optim::ErrorCode ec;
    const std::chrono::time_point<std::chrono::steady_clock> tick =
        std::chrono::steady_clock::now();
    do {
        ec = sp.step();
    } while (ec == njm::optim::ErrorCode::CONTINUE);
    const std::chrono::time_point<std::chrono::steady_clock> tock =
        std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = tock - tick;

    CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS);

    const std::vector<double> par = sp.par();

    SweepAgent<InfState> agent(net, features, par, 2, true);
    agent.rng(rng);
    System<InfState> s(net, mod);
    s.rng(rng);
    double val = 0.0;
    for (uint32_t i = 0; i < 50; ++i) {
        s.reset();
        s.start();

        val += runner(&s, &agent, 20, 1.0);
    }
    val /= 50.;

    r->set(std::pair<double, double>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                            elapsed).count(),
                    val));
}





int main(int argc, char *argv[]) {
    const std::vector<int> num_reps_list = {2};
    const std::vector<double> c_list = {10.0};
    const std::vector<double> t_list = {0.1};
    const std::vector<double> a_list = {5, 10};
    const std::vector<double> b_list = {1};
    const std::vector<double> ell_list = {0.4};
    const std::vector<double> min_step_size_list = {0.3, 0.7, 1.0};

    Experiment e;
    Experiment::FactorGroup * g0 = e.add_group();
    g0->add_factor(num_reps_list);
    g0->add_factor(c_list);
    g0->add_factor(t_list);
    g0->add_factor(a_list);
    g0->add_factor(b_list);
    g0->add_factor(ell_list);
    g0->add_factor(min_step_size_list);

    njm::thread::Pool p(std::thread::hardware_concurrency());

    std::vector<std::shared_ptr<Result<std::pair<double, double> > > > results;
    std::vector<Experiment::Factor> factors;
    std::vector<uint32_t> factors_level;
    std::vector<uint32_t> rep_number;

    e.start();
    uint32_t level_num = 0;
    do {
        const Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 20; ++rep) {
            uint32_t i = 0;
            CHECK_EQ(f.at(i).type, Experiment::FactorLevel::Type::is_int);
            const int num_reps = f.at(i++).val.int_val;
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
            p.service().post(std::bind(&run_vmax, r, rep, num_reps, c, t, a, b,
                            ell, min_step_size));
        }

        ++level_num;
    } while (e.next());

    p.join();

    CHECK_EQ(factors.size(), results.size());
    CHECK_EQ(factors.size(), factors_level.size());
    CHECK_EQ(factors.size(), rep_number.size());
    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");
    njm::data::Entry * results_entry = tk.entry(
            "vfnMaxExperiment_results.txt");
    *results_entry
        << "level_num, rep_num, elapsed, value, num_reps, c, t, a, b, ell, "
        << "min_step_size\n";
    for (uint32_t i = 0; i < results.size(); ++i) {
        const std::pair<double, double> result_i = results.at(i)->get();
        *results_entry << factors_level.at(i) << ", " << rep_number.at(i)
                       << ", " << result_i.first << ", " << result_i.second;
        Experiment::Factor f = factors.at(i);
        for (uint32_t j = 0; j < f.size(); ++j) {
            *results_entry << ", " << f.at(j);
        }
        *results_entry << "\n";
    }

    return 0;
}
