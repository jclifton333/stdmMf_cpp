#ifndef INF_STATE_NO_SO_MODEL
#define INF_STATE_NO_SO_MODEL

#include "states.hpp"
#include "network.hpp"
#include "infStateModel.hpp"

namespace stdmMf {


class InfStateNoSoModel : public InfStateModel {
private:
    double intcp_inf_latent_;
    double intcp_inf_;
    double intcp_rec_;
    double trt_act_inf_;
    double trt_act_rec_;
    double trt_pre_inf_;

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
    InfStateNoSoModel(const std::shared_ptr<const Network> & network);

    InfStateNoSoModel(const InfStateNoSoModel & other);

    ~InfStateNoSoModel() override = default;

    std::shared_ptr<Model<InfState> > clone() const override;

    std::vector<double> par() const override;

    void par(const std::vector<double> & par) override;
};


} // namespace stdmMf


#endif // INF_STATE_NO_SO_MODEL
