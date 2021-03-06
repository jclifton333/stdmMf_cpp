#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <njm_cpp/tools/random.hpp>

#include "ebolaStateGravityModel.hpp"
#include "system.hpp"
#include "randomAgent.hpp"
#include "proximalAgent.hpp"
#include "epsAgent.hpp"
#include "objFns.hpp"
#include "ebolaData.hpp"

namespace stdmMf {

template <typename State>
class GradientChecker {
public:
    Model<State> * m;
    int wiggle_var;
    std::vector<Transition<State> > * history;
    std::vector<double> par;
};


template <typename State>
class HessianChecker {

public:
    Model<State> * m;
    int wiggle_var;
    int gradient_var;
    std::vector<Transition<State> > * history;
    std::vector<double> par;
};

const double eps = 1e-4;


template <typename State>
double f (double x, void * params) {
    GradientChecker<State> * gc =
        static_cast<GradientChecker<State>*>(params);
    std::vector<double> par = gc->par;
    par.at(gc->wiggle_var) = x;
    gc->m->par(par);
    return gc->m->ll(*gc->history);
}


template <typename State>
double f_grad (double x, void * params) {
    HessianChecker<State> * hc =
        static_cast<HessianChecker<State>*>(params);
    std::vector<double> par = hc->m->par();
    par.at(hc->wiggle_var) = x;
    hc->m->par(par);
    return hc->m->ll_grad(*hc->history).at(hc->gradient_var);
}


TEST(TestEbolaStateGravityModel, TestPar) {
    EbolaData::deinit();
    // dummy ebola network
    EbolaData::init({"a"}, {"1"}, {1}, {0}, {1.0}, {0.0}, {0.0}, {});
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    EbolaStateGravityModel m(n);

    ASSERT_EQ(m.par().size(), 5);

    std::vector<double> par (m.par());
    for (uint32_t i = 0; i < par.size(); ++i) {
        par.at(i) = i;
    }

    m.par(par);
    par = m.par();
    for (uint32_t i = 0; i < par.size(); ++i) {
        ASSERT_EQ(par.at(i), static_cast<double>(i));
    }
}


TEST(TestEbolaStateGravityModel,TestLLGradient) {
    njm::tools::Rng rng;
    {
        // set up mock ebola network from a grid
        NetworkInit init;
        init.set_type(NetworkInit_NetType_GRID);
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);

        std::shared_ptr<Network> grid = Network::gen_network(init);

        std::vector<std::string> str_var(grid->size(), "blank");

        std::vector<uint32_t> region(grid->size());
        std::generate(region.begin(), region.end(),
                [&rng] () {return (uint32_t)rng.rint(0, 10);});

        std::vector<int> outbreaks(grid->size());
        std::generate(outbreaks.begin(), outbreaks.end(),
                [&rng] () {return rng.rint(0, 100);});

        std::vector<double> population(grid->size());
        std::generate(population.begin(), population.end(),
                [&rng] () {return rng.runif(3.0, 5.0);});

        std::vector<double> x(grid->size());
        std::vector<double> y(grid->size());

        for (uint32_t i = 0; i < grid->size(); ++i) {
            x.at(i) = grid->get_node(i).x();
            y.at(i) = grid->get_node(i).y();
        }

        EbolaData::deinit();
        EbolaData::init(str_var, str_var, region,
                outbreaks, population, x, y, {});
    }
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    // set par
    std::vector<double> par(m->par());
    par.at(0) = -4.5;
    par.at(1) = 4.0;
    par.at(2) = std::log(0.5);
    par.at(3) = -3.0;
    par.at(4) = -2.0;
    m->par(par);

    System<EbolaState> s(n,m);
    s.start();

    RandomAgent<EbolaState> a(n);

    const uint32_t num_points = 10;
    runner(&s, &a, num_points, 1.0);
    std::cout << "count: " << s.state().inf_bits.count() << std::endl;

    std::vector<Transition<EbolaState>> history(
            Transition<EbolaState>::from_sequence(s.history(), s.state()));
    ASSERT_EQ(history.size(), num_points);

    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x += rng.rnorm(0.0, std::abs(x) / 10.0);
            });
    m->par(par);
    const std::vector<double> grad_val =
        m->ll_grad(history);

    for (uint32_t i = 0; i < par.size(); ++i) {
        GradientChecker<EbolaState> gc;
        gc.m = m.get();
        gc.wiggle_var = i;
        gc.history = &history;
        gc.par = par;

        gsl_function F;
        F.function = &f<EbolaState>;
        F.params = &gc;

        double result;
        double abserr;
        gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

        std::cout << i << ": " << grad_val.at(i) << std::endl;

        EXPECT_NEAR(grad_val.at(i), result, eps)
            << "gradient failed for parameter " << i;
    }
}


TEST(TestEbolaStateGravityModel,TestLLHessian) {
    njm::tools::Rng rng;
    {
        // set up mock ebola network from a grid
        NetworkInit init;
        init.set_type(NetworkInit_NetType_GRID);
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);

        std::shared_ptr<Network> grid = Network::gen_network(init);

        std::vector<std::string> str_var(grid->size(), "blank");

        std::vector<uint32_t> region(grid->size());
        std::generate(region.begin(), region.end(),
                [&rng] () {return (uint32_t)rng.rint(0, 10);});

        std::vector<int> outbreaks(grid->size());
        std::generate(outbreaks.begin(), outbreaks.end(),
                [&rng] () {return rng.rint(0, 100);});

        std::vector<double> population(grid->size());
        std::generate(population.begin(), population.end(),
                [&rng] () {return rng.runif(3.0, 5.0);});

        std::vector<double> x(grid->size());
        std::vector<double> y(grid->size());

        for (uint32_t i = 0; i < grid->size(); ++i) {
            x.at(i) = grid->get_node(i).x();
            y.at(i) = grid->get_node(i).y();
        }

        EbolaData::deinit();
        EbolaData::init(str_var, str_var, region,
                outbreaks, population, x, y, {});
    }

    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    // set par
    std::vector<double> par(m->par());
    par.at(0) = -4.5;
    par.at(1) = 4.0;
    par.at(2) = std::log(0.5);
    par.at(3) = -3.0;
    par.at(4) = -2.0;
    m->par(par);

    System<EbolaState> s(n,m);
    s.start();

    RandomAgent<EbolaState> a(n);

    const uint32_t num_points = 10;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<EbolaState>> history(
            Transition<EbolaState>::from_sequence(s.history(), s.state()));
    ASSERT_EQ(history.size(), num_points);


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x += rng.rnorm(0.0, std::abs(x) / 10);
            });
    m->par(par);
    const std::vector<double> hess_val =
        m->ll_hess(history);


    for (uint32_t i = 0; i < par.size(); ++i) {
        for (uint32_t j = 0; j < par.size(); ++j) {
            HessianChecker<EbolaState> hc;
            hc.m = m.get();
            hc.wiggle_var = i;
            hc.gradient_var = j;
            hc.history = &history;
            hc.par = par;

            gsl_function F;
            F.function = &f_grad<EbolaState>;
            F.params = &hc;

            double result;
            double abserr;
            gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);
            std::cout << result << " == "
                      << hess_val.at(i * par.size() + j) << std::endl;

            const double diff(hess_val.at(i * par.size() + j) - result);
            EXPECT_LT(std::abs(diff / result), eps)
                << "hessian failed for parameters " << i << " and " << j;
        }
    }
}


// TEST(TestEbolaStateGravityModel, EstPar) {
//     njm::tools::Rng rng;
//     {
//         // set up mock ebola network from a grid
//         NetworkInit init;
//         init.set_type(NetworkInit_NetType_GRID);
//         init.set_dim_x(10);
//         init.set_dim_y(10);
//         init.set_wrap(false);

//         std::shared_ptr<Network> grid = Network::gen_network(init);

//         std::vector<std::string> str_var(grid->size(), "blank");

//         std::vector<uint32_t> region(grid->size());
//         std::generate(region.begin(), region.end(),
//                 [&rng] () {return (uint32_t)rng.rint(0, 10);});

//         std::vector<int> outbreaks(grid->size());
//         std::generate(outbreaks.begin(), outbreaks.end(),
//                 [&rng] () {return rng.rint(0, 100);});

//         std::vector<double> population(grid->size());
//         std::generate(population.begin(), population.end(),
//                 [&rng] () {return rng.runif(3.0, 5.0);});

//         std::vector<double> x(grid->size());
//         std::vector<double> y(grid->size());

//         for (uint32_t i = 0; i < grid->size(); ++i) {
//             x.at(i) = grid->get_node(i).x();
//             y.at(i) = grid->get_node(i).y();
//         }

//         EbolaData::deinit();
//         EbolaData::init(str_var, str_var, str_var, region,
//                 outbreaks, population, x, y, {});
//     }

//     // network
//     NetworkInit init;
//     init.set_type(NetworkInit_NetType_EBOLA);

//     const std::shared_ptr<Network> n = Network::gen_network(init);

//     const std::shared_ptr<EbolaStateGravityModel> m(
//             new EbolaStateGravityModel(n));

//     std::vector<double> par(m->par_size());
//     par.at(0) = -4.5;
//     par.at(1) = 4.0;
//     par.at(2) = std::log(0.5);
//     par.at(3) = -3.0;
//     par.at(4) = -2.0;
//     m->par(par);

//     System<EbolaState> s(n,m);
//     s.start();

//     const std::shared_ptr<ProximalAgent<EbolaState> > pa(
//             new ProximalAgent<EbolaState>(n));
//     const std::shared_ptr<RandomAgent<EbolaState> > ra(
//             new RandomAgent<EbolaState>(n));
//     EpsAgent<EbolaState> ea(n, pa, ra, 0.1);

//     const uint32_t num_points = 100;
//     runner(&s, &ea, num_points, 1.0);

//     std::vector<Transition<EbolaState> > history(
//             Transition<EbolaState>::from_sequence(s.history(), s.state()));
//     ASSERT_EQ(history.size(), num_points);

//     // scale paramters
//     std::vector<double> start_par = par;
//     std::for_each(start_par.begin(), start_par.end(),
//             [&rng] (double & x) {
//                 x += rng.rnorm(0.0, std::abs(x) / 10.0);
//             });
//     m->par(start_par);

//     m->est_par(history);

//     const std::vector<double> est_par = m->par();
//     for (uint32_t i = 0; i < m->par_size(); ++i) {
//         const double diff(par.at(i) - est_par.at(i));
//         EXPECT_LT(std::abs(diff / par.at(i)), 0.1)
//             << "Par " << i << " failed with truth " << par.at(i)
//             << " and estimate " << est_par.at(i);
//     }
// }



} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
