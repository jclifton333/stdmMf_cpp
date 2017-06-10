#ifndef EBOLA_STATE_GRAVITY_MODEL_HPP
#define EBOLA_STATE_GRAVITY_MODEL_HPP

#include "ebolaStateModel.hpp"

namespace stdmMf {


class EbolaStateGravityModel : public EbolaStateModel {
protected:
    double beta_0_, beta_1_, beta_2_;
    double trt_pre_, trt_act_;

    virtual double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    virtual std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    virtual std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_bits) const override;

public:
    EbolaStateGravityModel(const std::shared_ptr<const Network> & network);

    EbolaStateGravityModel(const EbolaStateGravityModel & other);

    virtual ~EbolaStateGravityModel() override = default;

    virtual std::shared_ptr<Model<EbolaState> > clone() const override;

    virtual std::vector<double> par() const override;

    virtual void par(const std::vector<double> & par) override;
};


}


#endif // EBOLA_STATE_GRAVITY_MODEL_HPP
