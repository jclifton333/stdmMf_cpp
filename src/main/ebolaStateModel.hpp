#ifndef EBOLA_STATE_MODEL_HPP
#define EBOLA_STATE_MODEL_HPP

#include "model.hpp"
#include "states.hpp"

namespace stdmMf {


class EbolaStateModel : public Model<EbolaState> {
protected:
    virtual double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

public:
    EbolaStateModel(const uint32_t & par_size,
            const std::shared_ptr<const Network> & network);

    EbolaStateModel(const EbolaStateModel & other);

    virtual ~EbolaStateModel() override = default;

    virtual std::shared_ptr<Model<EbolaState> > clone() const override = 0;

    virtual std::vector<double> par() const override = 0;

    virtual void par(const std::vector<double> & par) override = 0;

    std::vector<double> probs(
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_status) const override;

    double ll(const std::vector<Transition<EbolaState> > & history)
        const override;

    std::vector<double> ll_grad(
            const std::vector<Transition<EbolaState> > & history)
        const override;

    std::vector<double> ll_hess(
            const std::vector<Transition<EbolaState> > & history)
        const override;

    virtual EbolaState turn_clock(const EbolaState & curr_state,
            const boost::dynamic_bitset<> & trt_bits) const override;

};


} // namespace stdmMf


#endif // EBOLA_STATE_MODEL_HPP
