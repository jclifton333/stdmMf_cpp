#ifndef MODEL_HPP
#define MODEL_HPP

#include "states.hpp"

#include "network.hpp"

#include <njm_cpp/tools/random.hpp>

#include <cstdint>
#include <gsl/gsl_multimin.h>

namespace stdmMf {


template <typename State>
class Model : public njm::tools::RngClass {
protected:
    const uint32_t par_size_;

    const std::shared_ptr<const Network> network_;
    const uint32_t num_nodes_;

public:
    Model(const uint32_t & par_size,
            const std::shared_ptr<const Network> & network);

    Model(const Model & other);

    virtual ~Model() = default;

    virtual std::shared_ptr<Model<State> > clone() const = 0;

    virtual std::vector<double> par() const = 0;

    virtual void par(const std::vector<double> & par) = 0;

    uint32_t par_size() const;

    virtual void est_par(const std::vector<Transition<State> > & history);

    void est_par(const std::vector<StateAndTrt<State> > & history,
            const State & state);

    virtual std::vector<double> probs(
            const State & state,
            const boost::dynamic_bitset<> & trt_status) const = 0;

    virtual double ll(
            const std::vector<Transition<State> > & history) const = 0;

    virtual std::vector<double> ll_grad(
            const std::vector<Transition<State> > & history) const = 0;

    virtual std::vector<double> ll_hess(
            const std::vector<Transition<State> > & history) const = 0;

    virtual State turn_clock(const State & curr_state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    using njm::tools::RngClass::rng;
    virtual void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};

template <typename State>
class ModelFit {
private:
    Model<State> * const model_;
    const std::vector<Transition<State> > & history_;

public:

    ModelFit(Model<State> * const model,
            const std::vector<Transition<State> > & history);

    static double obj_fn(const gsl_vector * x, void * params);

    static void obj_fn_grad(const gsl_vector * x, void * params,
            gsl_vector * g);

    static void obj_fn_both(const gsl_vector * x, void * params,
            double * f, gsl_vector * g);

};


} // namespace stdmMf


#endif // MODEL_HPP__
