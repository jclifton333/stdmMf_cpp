#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cstdint>
#include <boost/dynamic_bitset.hpp>

#include <njm_cpp/tools/random.hpp>

#include "types.hpp"
#include "model.hpp"
#include "network.hpp"

namespace stdmMf {

class System : public njm::tools::RngClass {
private:
    const std::shared_ptr<const Network> network_;
    const std::shared_ptr<Model> model_;

    const uint32_t num_nodes_;

    State state_;
    boost::dynamic_bitset<> trt_bits_;

    std::vector<StateAndTrt> history_;

    uint32_t time_;

public:
    System(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model> & model);

    System(const System & other);

    std::shared_ptr<System> clone() const;

    uint32_t n_inf() const;

    uint32_t n_not() const;

    uint32_t n_trt() const;

    void reset();

    void wipe_trt();

    const State & state() const;

    void State(const State & state);

    const boost::dynamic_bitset<> & trt_bits() const;

    void trt_bits(const boost::dynamic_bitset<> & trt_bits);

    const std::vector<StateAndTrt> & history() const;

    void start();

    void update_history();

    void turn_clock();

    void turn_clock(const State & next_state);

};


} // namespace stdmMf


#endif // SYSTEM_HPP__
