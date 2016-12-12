#include "networkRunFeatures.hpp"

namespace stdmMf {

NetworkRunFeatures::NetworkRunFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t & run_length)
    : network_(network), num_nodes_(network->size()), run_length_(run_length),
      paths_(network->runs_of_len(run_length)),
      paths_by_node_(network->split_by_node(this->paths_)) {
}

} // namespace stdmMf
