#ifndef INF_STATE_MODEL_HPP
#define INF_STATE_MODEL_HPP

#include "model.hpp"
#include "states.hpp"

namespace stdmMf {


class InfStateModel : public Model<InfState> {
private:
    virtual double inf_b(const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual double rec_b(const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;


    virtual std::vector<double> inf_b_grad(const uint32_t & b_node,
            const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual std::vector<double> rec_b_grad(
            const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;


    virtual std::vector<double> inf_b_hess(const uint32_t & b_node,
            const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

    virtual std::vector<double> rec_b_hess(
            const uint32_t & b_node, const bool & b_trt,
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) const override = 0;

public:
    InfStateModel(const std::shared_ptr<const Network> & network);

    InfStateModel(const InfStateModel & other);

    virtual ~InfStateModel() override = default;

    virtual std::shared_ptr<Model<InfState> > clone() const override = 0;

    virtual std::vector<double> par() const override = 0;

    virtual void par(const std::vector<double> & par) override = 0;

    virtual std::vector<double> probs(
            const InfState & state,
            const boost::dynamic_bitset<> & trt_status) const override;

    virtual double ll(
            const std::vector<Transition<InfState> > & history) const override;

    virtual std::vector<double> ll_grad(
            const std::vector<Transition<InfState> > & history) const override;

    virtual std::vector<double> ll_hess(
            const std::vector<Transition<InfState> > & history) const override;

    virtual InfState turn_clock(const InfState & curr_state
            const boost::dynamic_bitset<> & trt_bits) const override = 0;
};


} // namespace stdmMf


#endif // INF_STATE_MODEL_HPP
