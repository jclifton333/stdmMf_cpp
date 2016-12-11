#include "networkRunFeatures.hpp"

namespace stdmMf {

NetworkRunFeatures::NetworkRunFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t & run_length)
    : network_(network), num_nodes(network->size()), run_length_(run_length),
      paths_(network->paths(run_length)),
      paths_by_node_(Network::split_by_node(paths_)) {
}

} // namespace stdmMf
