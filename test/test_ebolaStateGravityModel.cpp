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
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    EbolaStateGravityModel m(n);

    ASSERT_EQ(m.par().size(), 5);

    std::vector<double> par (m.par());
    for (uint32_t i = 0; i < par.size(); ++i) {
        ASSERT_EQ(par.at(i), 0.);
        par.at(i) = i;
    }

    m.par(par);
    par = m.par();
    for (uint32_t i = 0; i < par.size(); ++i) {
        ASSERT_EQ(par.at(i), static_cast<double>(i));
    }
}

TEST(TestEbolaStateGravityModel,TestLLGradient) {
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    par.at(0) = -5.246;
    par.at(1) = -155.8;
    par.at(2) = 0.186;
    par.at(3) = -2.0;
    par.at(4) = -1.0;
    m->par(par);

    System<EbolaState> s(n,m);

    RandomAgent<EbolaState> a(n);

    const uint32_t num_points = 50;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<EbolaState>> history(
            Transition<EbolaState>::from_sequence(s.history(), s.state()));
    ASSERT_EQ(history.size(), num_points);


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
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

        std::cout << result << std::endl;

        EXPECT_NEAR(grad_val.at(i), result, eps)
            << "gradient failed for parameter " << i;
    }
}


TEST(TestEbolaStateGravityModel,TestLLHessian) {
    // generate network
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    par.at(0) = -5.246;
    par.at(1) = -155.8;
    par.at(2) = 0.341;
    par.at(3) = -2.0;
    par.at(4) = -1.0;
    m->par(par);

    System<EbolaState> s(n,m);

    RandomAgent<EbolaState> a(n);

    const uint32_t num_points = 3;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<EbolaState>> history(
            Transition<EbolaState>::from_sequence(s.history(), s.state()));
    ASSERT_EQ(history.size(), num_points);


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x += rng.rnorm_01();
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

            EXPECT_NEAR(hess_val.at(i * par.size() + j), result, eps)
                << "hessian failed for parameters " << i << " and " << j;
        }
    }
}


TEST(TestEbolaStateGravityModel, EstPar) {
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    const std::shared_ptr<Network> n = Network::gen_network(init);

    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    njm::tools::Rng rng;
    std::vector<double> par(m->par_size());
    par.at(0) = -5.246;
    par.at(1) = -155.8;
    par.at(2) = 0.341;
    par.at(3) = 2.0;
    par.at(4) = 2.5;
    m->par(par);

    System<EbolaState> s(n,m);

    const std::shared_ptr<ProximalAgent<EbolaState> > pa(
            new ProximalAgent<EbolaState>(n));
    const std::shared_ptr<RandomAgent<EbolaState> > ra(
            new RandomAgent<EbolaState>(n));
    EpsAgent<EbolaState> ea(n, pa, ra, 0.1);

    const uint32_t num_points = 100;
    runner(&s, &ea, num_points, 1.0);

    std::vector<Transition<EbolaState> > history(
            Transition<EbolaState>::from_sequence(s.history(), s.state()));
    ASSERT_EQ(history.size(), num_points);

    // scale paramters
    std::vector<double> start_par = par;
    std::for_each(start_par.begin(), start_par.end(),
            [] (double & x) {
                x *= 10.0;
            });

    m->est_par(history);

    const std::vector<double> est_par = m->par();
    for (uint32_t i = 0; i < m->par_size(); ++i) {
        const double diff = std::abs(par.at(i) - est_par.at(i));
        EXPECT_LT(diff / par.at(i), 0.1)
            << "Par " << i << " failed with truth " << par.at(i)
            << " and estimate " << est_par.at(i);
    }
}



} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
