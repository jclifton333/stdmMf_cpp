#include "network.hpp"
#include <glog/logging.h>


namespace stdmMf {

uint32_t Network::size() const {
    return this->num_nodes;
}

const Node & Network::get_node(const uint32_t index) const {
    return this->node_list.nodes(index);
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
