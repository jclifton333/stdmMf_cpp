#ifndef MODEL_HPP
#define MODEL_HPP

#include "types.hpp"

#include <cstdint>

namespace stdmMf {


class Model {
public:
    virtual std::vector<double> par() const = 0;

    virtual void par(const std::vector<double> & par) = 0;

    virtual uint32_t par_size() const = 0;

    virtual void est_par(const std::vector<inf_trt_pair> & history) = 0;

    virtual std::vector<double> probs(
            const boost::dynamic_bitset<> & inf_status,
            const boost::dynamic_bitset<> & trt_status) const = 0;

    virtual double ll(const std::vector<inf_trt_pair> & history) const = 0;

    virtual std::vector<double> ll_grad(
            const std::vector<inf_trt_pair> & history) const = 0;
};


} // namespace stdmMf


#endif // MODEL_HPP__
