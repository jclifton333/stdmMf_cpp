#include "mixtureModel.hpp"

#include "utilities.hpp"

#include "glog/logging.h"

namespace stdmMf {


template <typename State>
MixtureModel<State>::MixtureModel(
        const std::vector<std::shared_ptr<Model<State> > > & models,
        const std::vector<double> & weights,
        const std::shared_ptr<const Network> & network)
    : Model<State>(std::accumulate(models.begin(), models.end(), 0u,
                    [] (const std::shared_ptr<Model<State> > & mod_) {
                        return mod_.par_size();
                    }),
            network),
      models_(models), weights_(weights) {
    CHECK(std::all_of(weights.begin(), weights.end(),
                    [] (const double & x_) {
                        return x_ >= 0;
                    }));
    CHECK_NEAR(std::accumulate(weights.begin(), weights.end(), 0.0,
                    [] (const double & x_) {
                        return x_;
                    }), 1.0, 1e-10);
}


template <typename State>
MixtureModel<State>::MixtureModel(const MixtureModel<State> & other)
    : Model<State>(other.par_size(), other.network_),
      models_(clone_vec(other.models_)), weights_(other.weights_) {
}


template <typename State>
std::shared_ptr<Model<State> > MixtureModel<State>::clone() const {
    return std::shared_ptr<Model<State> > (new MixtureModel<State>(*this));
}


template <typename State>
std::vector<double> MixtureModel<State>::par() const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <typename State>
void MixtureModel<State>::par(const std::vector<double> & par) {
    LOG(FATAL) << "NOT IMPLEMENTED";
}


template <typename State>
double MixtureModel<State>::ll(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return 0.0;
}


template <typename State>
std::vector<double> MixtureModel<State>::ll_grad(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <typename State>
std::vector<double> MixtureModel<State>::ll_hess(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <typename State>
void MixtureModel<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
    std::for_each(this->models_.begin(), this->models_.end(),
            [&rng] (const std::shared_ptr<Model<State> > & mod_) {
                mod_->rng(rng);
            });
}






} // namespace stdmMf
