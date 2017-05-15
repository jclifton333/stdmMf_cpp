#ifndef MIXTURE_MODEL_HPP
#define MIXTURE_MODEL_HPP

#include "model.hpp"

namespace stdmMf {


template <typename State>
class MixtureModel : public Model<State> {
protected:
    const std::vector<std::shared_ptr<Model<State> > > models_;

    const std::vector<double> weights_;


public:
    MixtureModel(const std::vector<std::shared_ptr<Model<State> > > & models,
            const std::vector<double> & weights,
            const std::shared_ptr<const Network> & network);

    MixtureModel(const MixtureModel & other);

    virtual ~MixtureModel() override = default;

    virtual std::shared_ptr<Model<State> > clone() const override;

    virtual std::vector<double> par() const override;

    virtual void par(const std::vector<double> & par) override;

    virtual std::vector<double> probs(
            const State & state,
            const boost::dynamic_bitset<> & trt_status) const override;

    virtual double ll(
            const std::vector<Transition<State> > & history) const override;

    virtual std::vector<double> ll_grad(
            const std::vector<Transition<State> > & history) const override;

    virtual std::vector<double> ll_hess(
            const std::vector<Transition<State> > & history) const override;

    virtual State turn_clock(const State & curr_state,
            const boost::dynamic_bitset<> & trt_bits) const override;

    using njm::tools::RngClass::rng;
    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // MIXTURE_MODEL_HPP
