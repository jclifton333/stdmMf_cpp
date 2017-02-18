#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <njm_cpp/tools/random.hpp>

#include "infStateXorSoModel.hpp"
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

TEST(TestInfStateXorSoModel, TestPar) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    InfStateXorSoModel m(n);

    CHECK_EQ(m.par().size(), 6);

    std::vector<double> par (m.par());
    for (uint32_t i = 0; i < par.size(); ++i) {
        EXPECT_EQ(par.at(i), 0.);
        par.at(i) = i;
    }

    m.par(par);
    par = m.par();
    for (uint32_t i = 0; i < par.size(); ++i) {
        EXPECT_EQ(par.at(i), static_cast<double>(i));
    }
}


TEST(TestInfStateXorSoModel,TestLLGradient) {
    // generate network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<InfStateXorSoModel> m(new InfStateXorSoModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);

    System<InfState> s(n,m);

    RandomAgent<InfState> a(n);

    const uint32_t num_points = 50;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<InfState> > history(
            Transition<InfState>::from_sequence(s.history(), s.state()));
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
        GradientChecker<InfState> gc;
        gc.m = m.get();
        gc.wiggle_var = i;
        gc.history = &history;
        gc.par = par;

        gsl_function F;
        F.function = &f<InfState>;
        F.params = &gc;

        double result;
        double abserr;
        gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

        EXPECT_NEAR(grad_val.at(i), result, eps)
            << "gradient failed for parameter " << i;
    }
}


TEST(TestInfStateXorSoModel,TestLLHessian) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<InfStateXorSoModel> m(new InfStateXorSoModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    std::for_each(par.begin(),par.end(),
            [&rng](double & x) {
                x = rng.rnorm_01();
            });
    m->par(par);

    System<InfState> s(n,m);

    RandomAgent<InfState> a(n);

    const uint32_t num_points = 3;
    runner(&s, &a, num_points, 1.0);

    std::vector<Transition<InfState> > history(
            Transition<InfState>::from_sequence(s.history(), s.state()));
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
            HessianChecker<InfState> hc;
            hc.m = m.get();
            hc.wiggle_var = i;
            hc.gradient_var = j;
            hc.history = &history;
            hc.par = par;

            gsl_function F;
            F.function = &f_grad<InfState>;
            F.params = &hc;

            double result;
            double abserr;
            gsl_deriv_central(&F, par.at(i), 1e-3, &result, &abserr);

            EXPECT_NEAR(hess_val.at(i * par.size() + j), result, eps)
                << "hessian failed for parameters " << i << " and " << j;
        }
    }
}


TEST(TestInfStateXorSoModel, EstPar) {
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> n = Network::gen_network(init);

    const std::shared_ptr<InfStateXorSoModel> m(new InfStateXorSoModel(n));

    njm::tools::Rng rng;
    std::vector<double> par;
    for (uint32_t i = 0; i < m->par_size(); ++i) {
        par.push_back(rng.rnorm(-2.0, 1.0));
    }

    m->par(par);

    System<InfState> s(n,m);

    const std::shared_ptr<ProximalAgent<InfState> > pa(
            new ProximalAgent<InfState>(n));
    const std::shared_ptr<RandomAgent<InfState> > ra(
            new RandomAgent<InfState>(n));
    EpsAgent<InfState> ea(n, pa, ra, 0.1);

    const uint32_t num_points = 100;
    runner(&s, &ea, num_points, 1.0);

    std::vector<Transition<InfState>> history(
            Transition<InfState>::from_sequence(s.history(), s.state()));
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
        const double diff = std::abs(par.at(i) - est_par.at(i));
        EXPECT_LT(diff / par.at(i), 0.1)
            << "Par " << i << " failed with truth " << par.at(i)
            << " and estimate " << est_par.at(i);
    }
}


TEST(TestInfStateXorSoModel, Spillover) {
    NetworkInit init;
    init.set_dim_x(5);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> n = Network::gen_network(init);

    const std::shared_ptr<InfStateXorSoModel> m(new InfStateXorSoModel(n));

    njm::tools::Rng rng;
    std::vector<double> par;
    for (uint32_t i = 0; i < m->par_size(); ++i) {
        par.push_back(rng.rnorm(-2.0, 1.0));
    }

    m->par(par);
    for (uint32_t i = 0; i < n->size(); ++i) {
        EXPECT_LT(n->size(), 20); // make sure no overflow in next line
        const boost::dynamic_bitset<> inf_bits(n->size(),
                rng.rint(0, 1 << n->size()));

        boost::dynamic_bitset<> trt_bits(n->size());

        // check latent infection
        trt_bits.reset();
        const double prob_not = m->probs(inf_bits, trt_bits).at(i);
        trt_bits.reset();
        trt_bits.set(i);
        const double prob_trt = m->probs(inf_bits, trt_bits).at(i);

        EXPECT_NE(prob_trt, prob_not);

        const Node & node_i = n->get_node(i);
        const uint32_t num_neigh = node_i.neigh_size();
        for (uint32_t j = 0; j < num_neigh; ++j) {
            const uint32_t neigh = node_i.neigh(j);

            {
                // set both
                trt_bits.reset();
                trt_bits.set(i);
                trt_bits.set(neigh);

                double prob_neigh_trt = m->probs(inf_bits, trt_bits).at(i);

                if (inf_bits.test(i) == inf_bits.test(neigh)) {
                    EXPECT_EQ(prob_not, prob_neigh_trt);
                } else if (!inf_bits.test(i)) {
                    EXPECT_NE(prob_trt, prob_neigh_trt);
                    EXPECT_NE(prob_not, prob_neigh_trt);
                } else {
                    EXPECT_EQ(prob_trt, prob_neigh_trt);
                }
            }
            {
                // set only neigh
                trt_bits.reset();
                trt_bits.set(neigh);

                double prob_neigh_trt = m->probs(inf_bits, trt_bits).at(i);

                if (inf_bits.test(i) == inf_bits.test(neigh)) {
                    EXPECT_EQ(prob_trt, prob_neigh_trt);
                } else if (!inf_bits.test(i)) {
                    EXPECT_NE(prob_trt, prob_neigh_trt);
                    EXPECT_NE(prob_not, prob_neigh_trt);
                } else {
                    EXPECT_EQ(prob_not, prob_neigh_trt);
                }
            }
        }
    }
}



} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
