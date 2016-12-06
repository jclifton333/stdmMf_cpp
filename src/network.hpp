#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>
#include <cstdint>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "stdmMf.pb.h"

namespace stdmMf {

class Network {
private:
    // number of nodes
    uint32_t num_nodes;
    // list of nodes
    NodeList node_list;

    // adjacency matrix
    // row to column
    boost::numeric::ublas::mapped_matrix<uint32_t> adj;

public:
    // number of nodes
    uint32_t size() const;

    // Retrieve index-th node in the network
    const Node & get_node(const uint32_t index) const;

    // Retrieve the adjacency matrix
    boost::numeric::ublas::mapped_matrix<int> get_adj() const;

    // generate a grid type network
    static std::shared_ptr<Network> gen_grid(
            const uint32_t dim_x, const uint32_t dim_y, const bool wrap);

    // generate a network from NetworkInit data
    static std::shared_ptr<Network> gen_network(const NetworkInit & init);
};

} // namespace stdmMf


#endif // NETWORK_HPP
