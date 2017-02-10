#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cstdint>
#include <boost/dynamic_bitset.hpp>

#include "types.hpp"
#include "random.hpp"
#include "model.hpp"
#include "network.hpp"

namespace stdmMf {

class System : public RngClass {
private:
    const std::shared_ptr<const Network> network_;
    const std::shared_ptr<Model> model_;

    const uint32_t num_nodes_;

    boost::dynamic_bitset<> inf_bits_;
    boost::dynamic_bitset<> trt_bits_;

    std::vector<InfAndTrt> history_;

    uint32_t time_;

public:
    System(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model> & model);

    System(const System & other);

    std::shared_ptr<System> clone() const;

    uint32_t n_inf() const;

    uint32_t n_not() const;

    uint32_t n_trt() const;

    void cleanse();

    void plague();

    void wipe_trt();

    void erase_history();

    const boost::dynamic_bitset<> & inf_bits() const;

    void inf_bits(const boost::dynamic_bitset<> & inf_bits);

    const boost::dynamic_bitset<> & trt_bits() const;

    void trt_bits(const boost::dynamic_bitset<> & trt_bits);

    const std::vector<InfAndTrt> & history() const;

    void start();

    void update_history();

    void turn_clock();

    void turn_clock(const std::vector<double> & probs);

};


} // namespace stdmMf


#endif // SYSTEM_HPP__
