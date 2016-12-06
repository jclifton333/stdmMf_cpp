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

    boost::dynamic_bitset<> inf_status_;
    boost::dynamic_bitset<> trt_status_;

    std::vector<inf_trt_pair> history_;

    uint32_t time_;

public:
    System(std::shared_ptr<const Network> network,
            std::shared_ptr<Model> model);


    uint32_t n_inf() const;

    uint32_t n_not() const;

    uint32_t n_trt() const;

    std::vector<uint32_t> inf_nodes() const;

    std::vector<uint32_t> not_nodes() const;

    std::vector<uint32_t> status() const;

    void cleanse();

    void plague();

    void wipe_trt();

    void erase_history();

    boost::dynamic_bitset<> inf_status() const;

    boost::dynamic_bitset<> trt_status() const;

    void start();

    void update_history();

    void turn_clock();

    void turn_clock(const std::vector<double> & probs);

};


} // namespace stdmMf


#endif // SYSTEM_HPP__
