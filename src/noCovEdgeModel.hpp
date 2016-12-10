#ifndef NO_COV_EDGE_MODEL_HPP
#define NO_COV_EDGE_MODEL_HPP

#include "types.hpp"
#include "model.hpp"
#include "network.hpp"

namespace stdmMf {


class NoCovEdgeModel : public Model {
private:
    const uint32_t par_size_;

    double intcp_inf_latent_;
    double intcp_inf_;
    double intcp_rec_;
    double trt_act_inf_;
    double trt_act_rec_;
    double trt_pre_inf_;

    const std::shared_ptr<const Network> network_;
    const uint32_t num_nodes_;

    double inf_b(const uint32_t & b_node, const bool & b_trt) const;

    double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt) const;

    double rec_b(const uint32_t & b_node, const bool & b_trt) const;

    std::vector<double> inf_b_grad(const uint32_t & b_node,
            const bool & b_trt) const;

    std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt) const;

    std::vector<double> rec_b_grad(
            const uint32_t & b_node, const bool & b_trt) const;


public:
    NoCovEdgeModel(const std::shared_ptr<const Network> & network);

    virtual std::vector<double> par() const;

    virtual void par(const std::vector<double> & par);

    virtual uint32_t par_size() const;

    virtual std::vector<double> probs(
            const boost::dynamic_bitset<> & inf_status,
            const boost::dynamic_bitset<> & trt_status) const;

    virtual double ll(const std::vector<BitsetPair> & history) const;

    virtual std::vector<double> ll_grad(
            const std::vector<BitsetPair> & history) const;
};


} // namespace stdmMf


#endif // NO_COV_EDGE_MODEL_HPP
