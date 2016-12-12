#include "network.hpp"
#include <algorithm>
#include <glog/logging.h>


namespace stdmMf {

uint32_t Network::size() const {
    return this->num_nodes;
}

const Node & Network::get_node(const uint32_t index) const {
    return this->node_list.nodes(index);
}

// Retrieve the adjacency matrix
boost::numeric::ublas::mapped_matrix<int> Network::get_adj() const {
    return this->adj;
}



// runs of length run_length
std::vector<NetworkRun> Network::runs_of_len(
        const uint32_t & run_length) const {
    std::vector<NetworkRun> runs;
    CHECK_GT(run_length, 0);
    if (run_length == 1) {
        for (uint32_t i = 0; i < this->size(); ++i) {
            NetworkRun nr;
            nr.nodes.push_back(i);
            nr.mask.resize(this->size());
            nr.mask.set(i);
            runs.push_back(nr);
        }
    } else {
        runs_of_len_helper(runs, std::vector<uint32_t>(), 0, run_length);
    }
    return runs;
}


std::vector<NetworkRun> Network::runs_of_len_cumu(
        const uint32_t & run_length) const {
    std::vector<NetworkRun> runs;
    for (uint32_t i = 0; i < run_length; ++i) {
        std::vector<NetworkRun> runs_to_add;
        runs_to_add = this->runs_of_len(i+1);
        runs.insert(runs.end(),runs_to_add.begin(), runs_to_add.end());
    }
    return runs;
}


// runs of length run_length
void Network::runs_of_len_helper(std::vector<NetworkRun> & runs,
        const std::vector<uint32_t> & curr_run,
        const uint32_t & curr_len, const uint32_t & target_len) const {
    if (curr_len == target_len && curr_run.at(0) < curr_run.at(target_len-1)) {
        // create NetworkRun
        NetworkRun nr;
        nr.nodes = curr_run;
        nr.mask.resize(this->size());
        boost::dynamic_bitset<> mask(this->size());
        // set mask
        std::for_each(nr.nodes.begin(), nr.nodes.end(),
                [&nr] (const uint32_t & node) {
                    nr.mask.set(node);
                });

        runs.push_back(nr);

    } else if (curr_len > 0) {
        // get the last node in the list
        const Node & last_node = this->get_node(curr_run.at(curr_len - 1));
        const uint32_t num_neigh = last_node.neigh_size();
        // iterate over neighbors of last node
        for (uint32_t i = 0; i < num_neigh; ++i) {
            const uint32_t neigh = last_node.neigh(i);
            auto neigh_it = std::find(curr_run.begin(), curr_run.end(), neigh);
            // if a neighbor isn't already in the list
            if (neigh_it == curr_run.end()) {
                std::vector<uint32_t> next_run = curr_run;
                next_run.push_back(neigh);
                // proceed with run
                runs_of_len_helper(runs, next_run, curr_len + 1, target_len);
            }
        }
    } else {
        for (uint32_t i = 0; i < this->size(); ++i) {
            std::vector<uint32_t> next_run;
            next_run.push_back(i);
            runs_of_len_helper(runs, next_run, curr_len + 1, target_len);
        }
    }
}

// split runs by node
std::vector<std::vector<NetworkRun> > Network::split_by_node(
        const std::vector<NetworkRun> & runs) const {
    std::vector<std::vector<NetworkRun> > by_node;
    by_node.resize(this->size());
    const uint32_t num_runs = runs.size();
    for (uint32_t i = 0; i < num_runs; ++i) {
        const NetworkRun nr = runs.at(i);
        const uint32_t run_length = nr.nodes.size();
        for (uint32_t j = 0; j < run_length; ++j) {
            by_node.at(nr.nodes.at(j)).push_back(nr);
        }
    }

    return by_node;
}



std::shared_ptr<Network> Network::gen_network(
        const NetworkInit & init) {
    CHECK(init.has_type());

    // call appropriate initializer
    switch (init.type()) {
    case NetworkInit_NetType_GRID: {
        CHECK(init.has_dim_x());
        CHECK(init.has_dim_y());
        CHECK(init.has_wrap());

        return gen_grid(init.dim_x(),init.dim_y(),init.wrap());
        break;
    }
    default:
        LOG(ERROR) << "Don't know how to initialize grid of type "
                   << init.type() << ".";
        break;
    }
}


std::shared_ptr<Network> Network::gen_grid(
        const uint32_t dim_x, const uint32_t dim_y, const bool wrap) {
    std::shared_ptr<Network> network = std::shared_ptr<Network>(new Network());

    // Either dimension of 1 is not a grid.  Define this in another
    // network.  Such as a "Line Netowrk".
    CHECK_GT(dim_x,1);
    CHECK_GT(dim_y,1);

    // iterate through grid column first
    network->num_nodes = dim_x * dim_y;
    network->adj = boost::numeric::ublas::mapped_matrix<uint32_t>(
            network->num_nodes,network->num_nodes);

    for (uint32_t x = 0, i = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y, ++i) {
            Node * n = network->node_list.add_nodes();
            n->set_index(i);
            n->set_x(static_cast<double>(x)/(dim_x - 1));
            n->set_y(static_cast<double>(y)/(dim_y - 1));

            // nothing action
            network->adj(i,i) = 1;

            // up
            if (y > 0) {
                const uint32_t neigh = i-1;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else if(wrap) {
                const uint32_t neigh = i + dim_y - 1;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else {
                n->add_neigh(i);
            }

            // down
            if (y < (dim_y - 1)) {
                const uint32_t neigh = i + 1;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = i - dim_y + 1;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else {
                n->add_neigh(i);
            }

            // left
            if (x > 0) {
                const uint32_t neigh = i - dim_y;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = network->num_nodes - dim_y + y;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else {
                n->add_neigh(i);
            }

            // right
            if (x < (dim_x - 1)) {
                const uint32_t neigh = i + dim_y;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = y;
                n->add_neigh(neigh);
                network->adj(i,neigh) = 1;
            } else {
                n->add_neigh(i);
            }
        }

    }

    return network;
}



} // namespace stdmMf
