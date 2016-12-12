#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>
#include <cstdint>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/dynamic_bitset.hpp>
#include "stdmMf.pb.h"

namespace stdmMf {

struct NetworkRun
{
    std::vector<uint32_t> nodes;
    boost::dynamic_bitset<> mask;
};

class Network {
private:
    // number of nodes
    uint32_t num_nodes;
    // list of nodes
    NodeList node_list;

    // adjacency matrix
    // row to column
    boost::numeric::ublas::mapped_matrix<uint32_t> adj;

    void runs_of_len_helper(std::vector<NetworkRun> & runs,
            const std::vector<uint32_t> & curr_run,
            const uint32_t & curr_len, const uint32_t & target_len) const;


public:
    // number of nodes
    uint32_t size() const;

    // Retrieve index-th node in the network
    const Node & get_node(const uint32_t index) const;

    // Retrieve the adjacency matrix
    boost::numeric::ublas::mapped_matrix<int> get_adj() const;

    // runs of length run_length
    std::vector<NetworkRun> runs_of_len(const uint32_t & run_length) const;

    // split runs by node
    std::vector<std::vector<NetworkRun> > split_by_node(
            const std::vector<NetworkRun> & runs) const;

    // generate a grid type network
    static std::shared_ptr<Network> gen_grid(
            const uint32_t dim_x, const uint32_t dim_y, const bool wrap);

    // generate a network from NetworkInit data
    static std::shared_ptr<Network> gen_network(const NetworkInit & init);

};

} // namespace stdmMf


#endif // NETWORK_HPP
