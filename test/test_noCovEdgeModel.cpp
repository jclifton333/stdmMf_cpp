#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>
#include "noCovEdgeModel.hpp"
#include "random.hpp"

namespace stdmMf {


template <typename T>
class GradientChecker {
public:
    T * m;
    int gradient_var;
    std::vector<BitsetPair> * history;
};

template <typename T>
class HessianChecker {
public:
    T * m;
    int gradient_var;
    int hessian_var;
    std::vector<BitsetPair> * history;
};

const double eps = 1e-6;

template <typename T>
double f (double x, void * params) {
    GradientChecker<T> * gc = static_cast<GradientChecker<T>*>(params);
    std::vector<double> par = gc->m->par();
    par.at(gc->gradient_var) = x;
    gc->m->par(par);
    return gc->m->ll(*gc->history);
}

template <typename T>
double fGrad (double x, void * params) {
    HessianChecker<T> * hc = static_cast<HessianChecker<T>*>(params);
    std::vector<double> par = hc->m->par();
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
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    NoCovEdgeModel m(n);

    // set par
    Rng rng;
    std::vector<double> par(m.par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.runif_01();
            });

    // generate history
    std::vector<BitsetPair> history;
    boost::dynamic_bitset<> inf_bits(n->size());
    boost::dynamic_bitset<> trt_bits(n->size());
    // t = 0
    history.push_back(BitsetPair(inf_bits, trt_bits));

    // t = 1
    inf_bits.flip(0);
    inf_bits.flip(1);
    inf_bits.flip(3);

    trt_bits.flip(0);
    trt_bits.flip(4);

    history.push_back(BitsetPair(inf_bits, trt_bits));

    // t = 2
    inf_bits.flip(4);
    inf_bits.flip(2);

    trt_bits.reset();
    trt_bits.flip(4);
    trt_bits.flip(5);

    history.push_back(BitsetPair(inf_bits, trt_bits));


    // calculate gradient
    const std::vector<double> grad_val =
        m.ll_grad(history);

    for (uint32_t i = 0; i < par.size(); ++i) {
        m.par(par);

        GradientChecker<NoCovEdgeModel> gc;
        gc.m = &m;
        gc.gradient_var = i;
        gc.history = &history;

        gsl_function F;
        F.function = &f<NoCovEdgeModel>;
        F.params = &gc;

        double result;
        double abserr;
        gsl_deriv_central(&F, par.at(i), 1e-8, &result, &abserr);

        EXPECT_NEAR(grad_val.at(i), result, eps);
    }
}



} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
