#ifndef INF_STATE_SEP_SO_MODEL_HPP
#define INF_STATE_SEP_SO_MODEL_HPP

#include "types.hpp"
#include "model.hpp"
#include "network.hpp"

namespace stdmMf {


class InfStateSepSoModel : public Model {
private:
    double intcp_inf_latent_;
    double intcp_inf_;
    double intcp_rec_;
    double trt_act_inf_;
    double trt_act_inf_so_;
    double trt_act_rec_;
    double trt_act_rec_so_;
    double trt_pre_inf_;
    double trt_pre_inf_so_;

    double inf_b(const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    double rec_b(const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> inf_b_grad(const uint32_t & b_node,
            const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> rec_b_grad(
            const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;


    std::vector<double> inf_b_hess(const uint32_t & b_node,
            const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> rec_b_hess(
            const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override;


public:
    InfStateSepSoModel(const std::shared_ptr<const Network> & network);

    InfStateSepSoModel(const InfStateSepSoModel & other);

    ~InfStateSepSoModel() override = default;

    std::shared_ptr<Model<State> > clone() const override;

    std::vector<double> par() const override;

    void par(const std::vector<double> & par) override;
};


} // namespace stdmMf


#endif // INF_STATE_SEP_SO_MODEL_HPP
