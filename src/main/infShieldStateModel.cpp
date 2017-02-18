#include "infShieldStateModel.hpp"

namespace stdmMf {


InfShieldStateModel::InfShieldStateModel(const uint32_t & par_size,
        const std::shared_ptr<const Network> & network)
    : Model<InfShieldState>(par_size, network) {
}


InfShieldStateModel::InfShieldStateModel(const InfShieldStateModel & other)
    : Model<InfShieldState>(par_size, network) {
}


std::vector<double> InfStateModel::probs(
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_status) const {
    const boost::dynamic_bitset<> & inf_status(state.inf_bits);

    std::vector<double> probs;
    const std::vector<uint32_t> status = njm::tools::combine_sets(
            state.inf_bits, trt_status);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const uint32_t status_i = status.at(i);
        const bool trt_i = status_i % 2 == 1;
        if (status_i < 2) {
            // not infected -> infection probability
            const Node & node = this->network_->get_node(i);
            const uint32_t num_neigh = node.neigh_size();

            double prob = 1.0 - this->inf_b(i, trt_i, inf_status,
                    trt_status); // latent prob

            // factor in neighbors
            for (uint32_t j = 0; j < num_neigh; ++j) {
                const uint32_t neigh = node.neigh(j);
                const uint32_t status_neigh = status.at(neigh);
                if (status_neigh >= 2) {
                    // if neighbor is infected
                    const bool trt_neigh = status_neigh % 2 == 1;
                    prob *= 1.0 - this->a_inf_b(neigh, i, trt_neigh, trt_i,
                            inf_status, trt_status);
                }
            }
            probs.push_back(1.0 - prob);
        } else {
            // infected -> recovery probability
            const double prob = this->rec_b(i, trt_i, inf_status, trt_status);
            probs.push_back(prob);
        }
    }
    return probs;
}


} // namespace stdmMf
