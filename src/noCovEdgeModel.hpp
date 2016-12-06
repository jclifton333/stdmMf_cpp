#ifndef NO_COV_EDGE_MODEL_HPP
#define NO_COV_EDGE_MODEL_HPP

#include "model.hpp"

namespace stdmMf {


class NoCovEdgeModel : public Model {
private:
    const uint32_t par_size_;
    std::vector<double> par_;

public:
    NoCovEdgeModel();

    virtual std::vector<double> par() const;

    virtual void par(const std::vector<double> & par);

    virtual uint32_t par_size() const;

    virtual void est_par(const std::vector<inf_trt_pair> & history);

    virtual std::vector<double> probs(
            const boost::dynamic_bitset<> & inf_status,
            const boost::dynamic_bitset<> & trt_status) const;

    virtual double ll(const std::vector<inf_trt_pair> & history) const;

    virtual std::vector<double> ll_grad(
            const std::vector<inf_trt_pair> & history) const;
};


} // namespace stdmMf


#endif // NO_COV_EDGE_MODEL_HPP
