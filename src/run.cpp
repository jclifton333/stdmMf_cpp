#include <memory>
#include <chrono>
#include <fstream>
#include "result.hpp"
#include "pool.hpp"
#include "system.hpp"
#include "agent.hpp"
#include "noCovEdgeModel.hpp"
#include "networkRunFeatures.hpp"
#include "sweepAgent.hpp"
#include "objFns.hpp"
#include "simPerturb.hpp"
#include "experiment.hpp"

using namespace stdmMf;


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
    rng->set_seed(seed);

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


    // features
    std::shared_ptr<Features> features(new NetworkRunFeatures(net, 4));

    auto min_fn = [&](const std::vector<double> & par,
            void * const data) {
        SweepAgent agent(net, features, par, 2);
        agent.set_rng(rng);
        System s(net, mod);
        s.set_rng(rng);
        double val = 0.0;
        for (uint32_t i = 0; i < num_reps; ++i) {
            s.cleanse();
            s.wipe_trt();
            s.erase_history();

            val += runner(&s, &agent, 20, 1.0);
        }
        val /= num_reps;

        // return negative since optim minimizes functions
        return -val;
    };

    SimPerturb sp(min_fn, std::vector<double>(
                    features->num_features(), 0.),
            NULL, c, t, a, b, ell, min_step_size);
    sp.set_rng(rng);

    Optim::ErrorCode ec;
    const std::chrono::time_point<std::chrono::high_resolution_clock> tick =
        std::chrono::high_resolution_clock::now();
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);
    const std::chrono::time_point<std::chrono::high_resolution_clock> tock =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = tock - tick;

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    const std::vector<double> par = sp.par();

    SweepAgent agent(net, features, par, 2);
    agent.set_rng(rng);
    System s(net, mod);
    s.set_rng(rng);
    double val = 0.0;
    for (uint32_t i = 0; i < 50; ++i) {
        s.cleanse();
        s.wipe_trt();
        s.erase_history();

        val += runner(&s, &agent, 20, 1.0);
    }
    val /= 50.;

    r->set(std::pair<double, double>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                            elapsed).count(),
                    val));
}





int main(int argc, char *argv[]) {
    const std::vector<int> num_reps_list = {5,10};
    const uint32_t final_t_list = 20;
    const std::vector<double> c_list = {10.0, 15.0};
    const std::vector<double> t_list = {0.1, 0.5};
    const std::vector<double> a_list = {5, 10};
    const std::vector<double> b_list = {1};
    const std::vector<double> ell_list = {1.0, 0.5};
    const std::vector<double> min_step_size_list = {1.0, 0.5};

    Experiment e;
    e.add_factor(num_reps_list);
    e.add_factor(c_list);
    e.add_factor(t_list);
    e.add_factor(a_list);
    e.add_factor(b_list);
    e.add_factor(ell_list);
    e.add_factor(min_step_size_list);

    Pool p(4);

    std::vector<std::shared_ptr<Result<std::pair<double, double> > > > results;
    std::vector<Experiment::Factor> factors;
    std::vector<uint32_t> factors_level;
    std::vector<uint32_t> rep_number;

    e.start();
    uint32_t level_num = 0;
    do {
        const Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 10; ++rep) {
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
            p.service()->post(std::bind(&run_vmax, r, rep, num_reps, c, t, a, b,
                            ell, min_step_size));
        }

        ++level_num;
    } while (e.next());

    p.join();

    CHECK_EQ(factors.size(), results.size());
    CHECK_EQ(factors.size(), factors_level.size());
    CHECK_EQ(factors.size(), rep_number.size());
    std::ofstream out;
    out.open("results.txt");
    out << "level_num, rep_num, elapsed, value, num_reps, c, t, a, b, ell, "
        << "min_step_size\n";
    for (uint32_t i = 0; i < results.size(); ++i) {
        const std::pair<double, double> result_i = results.at(i)->get();
        out << factors_level.at(i) << ", " << rep_number.at(i) << ", "
            << result_i.first << ", " << result_i.second;
        Experiment::Factor f = factors.at(i);
        for (uint32_t j = 0; j < f.size(); ++j) {
            out << ", " << f.at(j);
        }
        out << "\n";
    }

    return 0;
}
