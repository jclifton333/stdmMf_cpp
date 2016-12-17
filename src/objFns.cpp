#include "objFns.hpp"

namespace stdmMf {


double runner(System * system, Agent * agent, const uint32_t & final_time) {
    double value = 0.0;
    for (uint32_t i = 0; i < final_time; ++i) {
        const boost::dynamic_bitset<> trt_bits = agent->apply_trt(
                system->inf_bits(), system->history());

        system->trt_bits(trt_bits);

        system->turn_clock();

        value += system->inf_bits().count();
    }
    return value;
}


} // namespace stdmMf
