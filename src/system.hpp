#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cstdint>
#include <boost/dynamic_bitset.hpp>

#include "random.hpp"
#include "model.hpp"
#include "network.hpp"

namespace stdmMf {


class System : public RngClass {
private:
    std::shared_ptr<const Network> network;
    std::shared_ptr<Model> model;

    const uint32_t num_nodes;

    boost::dynamic_bitset<> inf_status;
    boost::dynamic_bitset<> trt_status;

    std::vector<std::pair< boost::dynamic_bitset<>,
                           boost::dynamic_bitset<> > > history;

    uint32_t time;

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

    boost::dynamic_bitset<> get_inf_status() const;

    boost::dynamic_bitset<> get_trt_status() const;

    void start();

    void update_history();

    void turn_clock();

    void turn_clock(const std::vector<double> & probs);

};


} // namespace stdmMf


#endif // SYSTEM_HPP__
