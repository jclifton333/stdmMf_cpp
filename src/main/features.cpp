#include "features.hpp"

namespace stdmMf {


template <typename State>
void Features<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
}


template class Features<InfState>;
template class Features<InfShieldState>;
template class Features<EbolaState>;

} // namespace stdmMf
