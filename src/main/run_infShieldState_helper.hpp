#ifndef RUN_INF_SHIELD_STATE_HELPER
#define RUN_INF_SHIELD_STATE_HELPER

#include "network.hpp"
#include "states.hpp"
#include "model.hpp"

namespace stdmMf {

typedef std::pair<std::shared_ptr<Model<InfShieldState> >,
                  std::shared_ptr<Model<InfShieldState> > > ModelPair;

void run_infShieldState_sim(
        const std::string & sim_name,
        const std::vector<std::shared_ptr<const Network> > & networks,
        const std::vector<std::pair<std::string,
        std::vector<ModelPair> > > & models);


} // namespace stdmMf


#endif // RUN_INF_SHIELD_STATE_HELPER
