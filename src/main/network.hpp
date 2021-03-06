#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>
#include <cstdint>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/dynamic_bitset.hpp>
#include "network.pb.h"

namespace stdmMf {

struct NetworkRun
{
    std::vector<uint32_t> nodes;
    boost::dynamic_bitset<> mask;
    uint32_t len;
};

class Network {
private:
    std::string kind_;

    // number of nodes
    uint32_t num_nodes_;
    // list of nodes
    NodeList node_list_;

    // adjacency matrix
    // row to column
    boost::numeric::ublas::mapped_matrix<uint32_t> adj_;

    std::vector<std::vector<double> > dist_;

    std::vector<std::vector<double> > calc_dist() const;

    void runs_of_len_helper(std::vector<NetworkRun> & runs,
            const std::vector<uint32_t> & curr_run,
            const uint32_t & curr_len, const uint32_t & target_len) const;

public:
    std::shared_ptr<Network> clone() const;

    // a string name for the type of network
    std::string kind() const;

    // number of nodes
    uint32_t size() const;

    // Retrieve index-th node in the network
    const Node & get_node(const uint32_t index) const;

    // Retrieve the adjacency matrix
    boost::numeric::ublas::mapped_matrix<uint32_t> get_adj() const;

    const std::vector<std::vector<double> > & dist() const;

    // runs of length run_length
    std::vector<NetworkRun> runs_of_len(const uint32_t & run_length) const;

    // runs of length run_length
    std::vector<NetworkRun> runs_of_len_cumu(const uint32_t & run_length) const;

    // split runs by node
    std::vector<std::vector<NetworkRun> > split_by_node(

            const std::vector<NetworkRun> & runs) const;

    // edge pairs
    std::vector<std::pair<uint32_t, uint32_t> > edges() const;

    // generate a network from NetworkInit data
    static std::shared_ptr<Network> gen_network(const NetworkInit & init);

    // generate a grid type network
    static std::shared_ptr<Network> gen_grid(
            const uint32_t dim_x, const uint32_t dim_y, const bool wrap);
    // generate a grid type network
    static std::shared_ptr<Network> gen_random(const uint32_t size);


    // generate a barabasi type network
    static std::shared_ptr<Network> gen_barabasi(const uint32_t size);

    // generate network for ebola
    static std::shared_ptr<Network> gen_ebola();

    // join all subnetworks
    static void join_subnetworks(const std::shared_ptr<Network> & network);

    static uint32_t check_network(const std::shared_ptr<Network> & network);

    const NodeList & node_list() const;
};

} // namespace stdmMf


#endif // NETWORK_HPP
