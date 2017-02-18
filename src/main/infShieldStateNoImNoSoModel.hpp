#ifndef INF_SHIELD_STATE_NO_IM_NO_SO_MODEL_HPP
#define INF_SHIELD_STATE_NO_IM_NO_SO_MODEL_HPP

#include "states.hpp"
#include "infShieldStateModel.hpp"

namespace stdmMf {


class InfShieldStateNoImNoSoModel : public InfShieldStateModel {
private:
    double intcp_inf_latent_;
    double intcp_inf_;
    double intcp_rec_;
    double trt_act_inf_;
    double trt_act_rec_;
    double trt_pre_inf_;

    double shield_coef_;


    double inf_b(const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    double rec_b(const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;


    std::vector<double> inf_b_grad(const uint32_t & b_node,
            const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> rec_b_grad(
            const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> inf_b_hess(const uint32_t & b_node,
            const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    std::vector<double> rec_b_hess(
            const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    double shield_draw(const uint32_t & loc,
            const InfShieldState & curr_state) const override;

    double shield_prob(
            const uint32_t & loc, const InfShieldState & curr_state,
            const InfShieldState & next_state,
            const bool & log_scale = false) const override;

    std::vector<double> shield_grad(
            const uint32_t & loc, const InfShieldState & curr_state,
            const InfShieldState & next_state) const override;

    std::vector<double> shield_hess(
            const uint32_t & loc, const InfShieldState & curr_state,
            const InfShieldState & next_state) const override;


public:
    InfShieldStateNoImNoSoModel(
            const std::shared_ptr<const Network> & network);

    InfShieldStateNoImNoSoModel(const InfShieldStateNoImNoSoModel & other);

    ~InfShieldStateNoImNoSoModel() override = default;

    std::shared_ptr<Model<InfShieldState> > clone() const override;

    std::vector<double> par() const override;

    void par(const std::vector<double> & par) override;
};


} // namespace stdmMf


#endif // INF_SHIELD_STATE_NO_IM_NO_SO_MODEL_HPP
