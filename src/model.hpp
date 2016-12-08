#ifndef MODEL_HPP
#define MODEL_HPP

#include "types.hpp"

#include <cstdint>
#include <gsl/gsl_multimin.h>

namespace stdmMf {


class Model {
public:
    virtual std::vector<double> par() const = 0;

    virtual void par(const std::vector<double> & par) = 0;

    virtual uint32_t par_size() const = 0;

    virtual void est_par(const std::vector<BitsetPair> & history) = 0;

    virtual std::vector<double> probs(
            const boost::dynamic_bitset<> & inf_status,
            const boost::dynamic_bitset<> & trt_status) const = 0;

    virtual double ll(const std::vector<BitsetPair> & history) const = 0;

    virtual std::vector<double> ll_grad(
            const std::vector<BitsetPair> & history) const = 0;
};

class ModelFit {
public:
    Model & model_;
    std::vector<BitsetPair> history_;

    static double obj_fn(const gsl_vector * x, void * params);

    static void obj_fn_grad(const gsl_vector * x, void * params,
            gsl_vector * g);

    static void obj_fn_both(const gsl_vector * x, void * params,
            double * f, gsl_vector * g);

};


} // namespace stdmMf


#endif // MODEL_HPP__
