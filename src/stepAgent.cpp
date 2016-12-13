#include "stepAgent.hpp"
#include <glog/logging.h>

namespace stdmMf {


StepAgent::StepAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features> & features,
        const uint32_t & max_sweeps,
        const std::vector<double> & coef)
    : Agent(network), features_(features), max_sweeps_(max_sweeps) {
    CHECK_EQ(coef.size(), features->num_features());
}


boost::dynamic_bitset<> StepAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);


}




} // namespace stdmMf
