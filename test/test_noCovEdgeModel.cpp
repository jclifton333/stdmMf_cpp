#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>
#include "noCovEdgeModel.hpp"
#include "random.hpp"
#include "system.hpp"
#include "randomAgent.hpp"
#include "proximalAgent.hpp"
#include "epsAgent.hpp"
#include "runner.hpp"

namespace stdmMf {


template <typename T>
class GradientChecker {
public:
    T * m;
    int gradient_var;
    std::vector<BitsetPair> * history;
    std::vector<double> par;
};

template <typename T>
class HessianChecker {
public:
    T * m;
    int gradient_var;
    int hessian_var;
    std::vector<BitsetPair> * history;
    std::vector<double> par;
};

const double eps = 1e-6;

template <typename T>
double f (double x, void * params) {
    GradientChecker<T> * gc = static_cast<GradientChecker<T>*>(params);
    std::vector<double> par = gc->par;
    par.at(gc->gradient_var) = x;
    gc->m->par(par);
    return gc->m->ll(*gc->history);
}

template <typename T>
double fGrad (double x, void * params) {
    HessianChecker<T> * hc = static_cast<HessianChecker<T>*>(params);
    std::vector<double> par = hc->par();
    par.at(hc->hessian_var) = x;
    hc->m->par(par);
    return hc->m->ll_grad(*hc->history);
}

TEST(TestNoCovEdgeModel, TestPar) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NoCovEdgeModel m(n);

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


TEST(TestNoCovEdgeModel,TestLLGradient) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<NoCovEdgeModel> m(new NoCovEdgeModel(n));

    // set par
    Rng rng;
    std::vector<double> par(m->par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);

    System s(n,m);

    RandomAgent a(n);

    runner(&s, &a, 50);

    std::vector<BitsetPair> history = s.history();
    history.push_back(BitsetPair(s.inf_bits(), s.trt_bits()));


    // generate new parameters so gradient is not zero
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);
    const std::vector<double> grad_val =
        m->ll_grad(history);

    for (uint32_t i = 0; i < par.size(); ++i) {
        GradientChecker<NoCovEdgeModel> gc;
        gc.m = m.get();
        gc.gradient_var = i;
        gc.history = &history;
        gc.par = par;

        gsl_function F;
        F.function = &f<NoCovEdgeModel>;
        F.params = &gc;

        double result;
        double abserr;
        gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

        EXPECT_NEAR(grad_val.at(i), result, eps)
            << "gradient failed for parameter " << i;
    }
}

TEST(TestNoCovEdgeModel, EstPar) {
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> n = Network::gen_network(init);

    const std::shared_ptr<NoCovEdgeModel> m(new NoCovEdgeModel(n));

    Rng rng;
    std::vector<double> par;
    for (uint32_t i = 0; i < m->par_size(); ++i) {
        par.push_back(rng.rnorm(-2.0, 1.0));
    }

    m->par(par);

    System s(n,m);

    const std::shared_ptr<ProximalAgent> pa(new ProximalAgent(n));
    const std::shared_ptr<RandomAgent> ra(new RandomAgent(n));
    EpsAgent ea(n, pa, ra, 0.1);

    runner(&s, &ea, 100);

    std::vector<BitsetPair> history = s.history();
    history.push_back(BitsetPair(s.inf_bits(), s.trt_bits()));

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
