#ifndef INF_SHIELD_STATE_MODEL_HPP
#define INF_SHIELD_STATE_MODEL_HPP


namespace stdmMf {


class InfShieldStateModel : public Model<InfShieldState> {
private:
    virtual double inf_b(const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual double a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual double rec_b(const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;


    virtual std::vector<double> inf_b_grad(const uint32_t & b_node,
            const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> a_inf_b_grad(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> rec_b_grad(
            const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> inf_b_hess(const uint32_t & b_node,
            const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> a_inf_b_hess(
            const uint32_t & a_node, const uint32_t & b_node,
            const bool & a_trt, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual std::vector<double> rec_b_hess(
            const uint32_t & b_node, const bool & b_trt,
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_bits) const = 0;

    virtual double shield_prob(
            const uint32_t & loc, const InfShield & state) const = 0;

public:
    InfShieldStateModel(const uint32_t & par_size,
            const std::shared_ptr<const Network> & network);

    InfShieldStateModel(const InfShieldStateModel & other);

    virtual ~InfStateShieldModel() override = default;

    virtual std::shared_ptr<Model<InfShieldState> > clone() const override = 0;

    virtual std::vector<double> par() const override = 0;

    virtual void par(const std::vector<double> & par) override = 0;

    std::vector<double> probs(
            const InfShieldState & state,
            const boost::dynamic_bitset<> & trt_status) const override;

    double ll(const std::vector<Transition<InfShieldState> > & history)
        const override;

    std::vector<double>
    ll_grad(const std::vector<Transition<InfShieldState> > & history)
        const override;

    std::vector<double>
    ll_hess(const std::vector<Transition<InfShieldState> > & history)
        const override;

    virtual InfShieldState turn_clock(const InfShieldState & curr_state,
            const boost::dynamic_bitset<> & trt_bits) const override;
};


} // namespace stdmMf


#endif // INF_SHIELD_STATE_MODEL_HPP
