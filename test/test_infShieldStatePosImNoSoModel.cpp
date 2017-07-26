#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <njm_cpp/tools/random.hpp>

#include "infShieldStatePosImNoSoModel.hpp"
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


TEST(TestInfShieldStatePosImNoSoModel, TestPar) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    InfShieldStatePosImNoSoModel m(n);

    CHECK_EQ(m.par().size(), 7);

    std::vector<double> par (m.par());
    for (uint32_t i = 0; i < par.size(); ++i) {
        CHECK_EQ(par.at(i), 0.);
        par.at(i) = i;
    }

    m.par(par);
    par = m.par();
    for (uint32_t i = 0; i < par.size(); ++i) {
        CHECK_EQ(par.at(i), static_cast<double>(i));
    }
}

TEST(TestInfShieldStatePosImNoSoModel,TestLLGradient) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<InfShieldStatePosImNoSoModel> m(
            new InfShieldStatePosImNoSoModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);

    System<InfShieldState> s(n,m);

    RandomAgent<InfShieldState> a(n);

    const uint32_t num_points = 50;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<InfShieldState>> history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));
    CHECK_EQ(history.size(), num_points);


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);
    const std::vector<double> grad_val =
        m->ll_grad(history);

    for (uint32_t i = 0; i < par.size(); ++i) {
        GradientChecker<InfShieldState> gc;
        gc.m = m.get();
        gc.wiggle_var = i;
        gc.history = &history;
        gc.par = par;

        gsl_function F;
        F.function = &f<InfShieldState>;
        F.params = &gc;

        double result;
        double abserr;
        gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

        EXPECT_NEAR(grad_val.at(i), result, eps)
            << "gradient failed for parameter " << i;
    }
}


TEST(TestInfShieldStatePosImNoSoModel,TestLLHessian) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<InfShieldStatePosImNoSoModel> m(
            new InfShieldStatePosImNoSoModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);

    System<InfShieldState> s(n,m);

    RandomAgent<InfShieldState> a(n);

    const uint32_t num_points = 3;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<InfShieldState>> history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));
    CHECK_EQ(history.size(), num_points);


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);
    const std::vector<double> hess_val =
        m->ll_hess(history);

    for (uint32_t i = 0; i < par.size(); ++i) {
        for (uint32_t j = 0; j < par.size(); ++j) {
            HessianChecker<InfShieldState> hc;
            hc.m = m.get();
            hc.wiggle_var = i;
            hc.gradient_var = j;
            hc.history = &history;
            hc.par = par;

            gsl_function F;
            F.function = &f_grad<InfShieldState>;
            F.params = &hc;

            double result;
            double abserr;
            gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

            EXPECT_NEAR(hess_val.at(i * par.size() + j), result, eps)
                << "hessian failed for parameters " << i << " and " << j;
        }
    }
}


TEST(TestInfShieldStatePosImNoSoModel, EstPar) {
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> n = Network::gen_network(init);

    const std::shared_ptr<InfShieldStatePosImNoSoModel> m(
            new InfShieldStatePosImNoSoModel(n));

    njm::tools::Rng rng;
    // latent infections
    const double prob_inf_latent = 0.01;
    const double intcp_inf_latent =
        std::log(1. / (1. - prob_inf_latent) - 1);

    // neighbor infections
    const double prob_inf = 0.5;
    const uint32_t prob_num_neigh = 3;
    const double intcp_inf =
        std::log(std::pow((1. - prob_inf) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.);

    const double trt_act_inf =
        std::log(std::pow((1. - prob_inf * 0.25) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.)
        - intcp_inf;

    const double trt_pre_inf =
        std::log(std::pow((1. - prob_inf * 0.75) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.)
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

    m->par(par);

    System<InfShieldState> s(n,m);

    const std::shared_ptr<ProximalAgent<InfShieldState> > pa(
            new ProximalAgent<InfShieldState>(n));
    const std::shared_ptr<RandomAgent<InfShieldState> > ra(
            new RandomAgent<InfShieldState>(n));
    EpsAgent<InfShieldState> ea(n, pa, ra, 0.1);

    const uint32_t num_points = 100;
    runner(&s, &ea, num_points, 1.0);

    std::vector<Transition<InfShieldState> > history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));
    CHECK_EQ(history.size(), num_points);

    // scale paramters
    std::vector<double> start_par = par;
    std::for_each(start_par.begin(), start_par.end(),
            [] (double & x) {
                x *= 10.0;
            });

    m->est_par(history);

    const std::vector<double> est_par = m->par();
    for (uint32_t i = 0; i < m->par_size(); ++i) {
        const double diff(par.at(i) - est_par.at(i));
        EXPECT_LT(std::abs(diff / par.at(i)), 0.2)
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
