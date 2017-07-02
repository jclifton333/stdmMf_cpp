#include "network.hpp"
#include "ebolaData.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <list>
#include <glog/logging.h>
#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/info/project.hpp>
#include <fstream>

namespace stdmMf {

std::shared_ptr<Network> Network::clone() const {
    return std::shared_ptr<Network>(new Network(*this));
}

std::string Network::kind() const {
    return this->kind_;
}

uint32_t Network::size() const {
    return this->num_nodes_;
}

const Node & Network::get_node(const uint32_t index) const {
    return this->node_list_.nodes(index);
}


std::vector<std::vector<double> > Network::calc_dist() const{
    CHECK_EQ(this->adj_.size1(), this->num_nodes_);
    CHECK_EQ(this->adj_.size2(), this->num_nodes_);

    std::vector<std::vector<double> > dist(this->num_nodes_,
            std::vector<double>(this->num_nodes_,
                    std::numeric_limits<double>::infinity()));
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        dist.at(i).at(i) = 0.0;
        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            if (this->adj_(i,j) == 1 && i != j) {
                dist.at(i).at(j) = 1.0;
            }
        }
    }

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            double & direct(dist.at(i).at(j));
            for (uint32_t k = 0; k < this->num_nodes_; ++k) {
                const double & step_one(dist.at(i).at(k));
                const double & step_two(dist.at(k).at(j));
                if (direct > (step_one + step_two)) {
                    direct = step_one + step_two;
                }
            }
        }
    }

    return dist;
}


const std::vector<std::vector<double> > & Network::dist() const {
    return this->dist_;
}


// Retrieve the adjacency matrix
boost::numeric::ublas::mapped_matrix<int> Network::get_adj() const {
    return this->adj_;
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
            nr.len = 1;
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
    CHECK_LE(curr_len, target_len);
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
        nr.len = target_len;
        runs.push_back(nr);
    } else if (curr_len > 0 && curr_len < target_len) {
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
    } else if (curr_len == 0) {
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


std::vector<std::pair<uint32_t, uint32_t> > Network::edges() const {
    std::vector<std::pair<uint32_t, uint32_t> > edges;
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        for (uint32_t j = (i + 1); j < this->num_nodes_; ++j) {
            if (this->adj_(i,j) != 0) {
                edges.emplace_back(i, j);
            }
        }
    }
    return edges;
}



std::shared_ptr<Network> Network::gen_network(
        const NetworkInit & init) {
    CHECK(init.has_type());

    std::shared_ptr<Network> network(NULL);

    // call appropriate initializer
    switch (init.type()) {
    case NetworkInit_NetType_GRID: {
        CHECK(init.has_dim_x()) << "grid requires x dimension";
        CHECK(init.has_dim_y()) << "grid requires y dimension";
        CHECK(init.has_wrap()) << "grid requires a wrap specification";

        network = Network::gen_grid(init.dim_x(),init.dim_y(),init.wrap());
        break;
    }
    case NetworkInit_NetType_BARABASI: {
        CHECK(init.has_size()) << "barabasi requires a size";

        network = Network::gen_barabasi(init.size());
        break;
    }
    case NetworkInit_NetType_RANDOM: {
        CHECK(init.has_size()) << "random requires a size";

        network = Network::gen_random(init.size());
        break;
    }
    case NetworkInit_NetType_EBOLA: {
        network = Network::gen_ebola();
        break;
    }
    default:
        LOG(FATAL) << "Don't know how to initialize network of type "
                   << init.type() << ".";
        break;
    }

    return network;
}


std::shared_ptr<Network> Network::gen_grid(
        const uint32_t dim_x, const uint32_t dim_y, const bool wrap) {
    std::shared_ptr<Network> network = std::shared_ptr<Network>(new Network());

    // iterate through grid column first
    network->kind_ = "grid-" + std::to_string(dim_x) + "x"
        + std::to_string(dim_y);
    network->num_nodes_ = dim_x * dim_y;
    network->adj_ = boost::numeric::ublas::mapped_matrix<uint32_t>(
            network->num_nodes_,network->num_nodes_);

    for (uint32_t x = 0, i = 0; x < dim_x; ++x) {
        for (uint32_t y = 0; y < dim_y; ++y, ++i) {
            Node * n = network->node_list_.add_nodes();
            n->set_index(i);

            if (dim_x > 1) {
                n->set_x(static_cast<double>(x)/(dim_x - 1));
            } else {
                n->set_x(0.0);
            }

            if (dim_y > 1) {
                n->set_y(static_cast<double>(y)/(dim_y - 1));
            } else {
                n->set_y(0.0);
            }

            // up
            if (y > 0) {
                const uint32_t neigh = i-1;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            } else if(wrap) {
                const uint32_t neigh = i + dim_y - 1;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            }

            // down
            if (y < (dim_y - 1)) {
                const uint32_t neigh = i + 1;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = i - dim_y + 1;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            }

            // left
            if (x > 0) {
                const uint32_t neigh = i - dim_y;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = network->num_nodes_ - dim_y + y;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            }

            // right
            if (x < (dim_x - 1)) {
                const uint32_t neigh = i + dim_y;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            } else if (wrap) {
                const uint32_t neigh = y;
                n->add_neigh(neigh);
                network->adj_(i,neigh) = 1;
            }
        }

    }

    network->dist_ = network->calc_dist();

    return network;
}


std::shared_ptr<Network> Network::gen_barabasi(const uint32_t size) {
    std::shared_ptr<Network> network = std::shared_ptr<Network>(new Network());

    CHECK_GE(size, 2);

    // init adjacency matrix
    network->kind_ = "barabasi-" + std::to_string(size);
    network->num_nodes_ = size;
    network->adj_ = boost::numeric::ublas::mapped_matrix<uint32_t>(
            network->num_nodes_, network->num_nodes_);

    // add the first two nodes
    {
        Node * const n0 = network->node_list_.add_nodes();
        Node * const n1 = network->node_list_.add_nodes();
        n0->set_index(0);
        n1->set_index(1);

        n0->add_neigh(1);
        n1->add_neigh(0);

        network->adj_(0,1) = 1;
        network->adj_(1,0) = 1;
    }

    njm::tools::Rng rng;

    std::vector<uint32_t> edge_deg(2, 1);

    for (uint32_t i = 2; i < size; ++i) {
        Node * const n = network->node_list_.add_nodes();
        n->set_index(i);

        const uint32_t total_deg = std::accumulate(edge_deg.begin(),
                edge_deg.end(), 0);

        uint32_t connect_to = 0;
        uint32_t current_tot = edge_deg.at(0);
        const uint32_t draw = rng.rint(0, total_deg);
        while (draw > current_tot) {
            current_tot += edge_deg.at(++connect_to);
        }
        network->adj_(i, connect_to) = 1;
        network->adj_(connect_to, i) = 1;

        n->add_neigh(connect_to);

        network->node_list_.mutable_nodes(connect_to)->add_neigh(i);

        ++edge_deg.at(connect_to);
        edge_deg.push_back(1);
    }

    network->dist_ = network->calc_dist();

    return network;
}


std::shared_ptr<Network> Network::gen_random(const uint32_t size) {
    std::shared_ptr<Network> network = std::shared_ptr<Network>(new Network());

    network->kind_ = "random-" + std::to_string(size);
    network->num_nodes_ = size;
    network->adj_ = boost::numeric::ublas::mapped_matrix<uint32_t>(
            network->num_nodes_,network->num_nodes_);

    const uint32_t num_neigh(3);

    CHECK_GE(size, std::max(2u, num_neigh));

    njm::tools::Rng rng;

    for (uint32_t i = 0; i < size; ++i) {
        Node * const n = network->node_list_.add_nodes();
        n->set_index(i);

        n->set_x(rng.runif_01());
        n->set_y(rng.runif_01());
    }

    std::vector<std::set<uint32_t> > subnets;
    std::map<uint32_t, uint32_t> subnet_by_node;

    for (uint32_t i = 0; i < size; ++i) {
        Node * const node_i(network->node_list_.mutable_nodes(i));
        std::priority_queue<std::pair<double, uint32_t> > neighs;
        for (uint32_t j = 0; j < size; ++j) {
            if (i == j)
                continue;

            const Node & node_j(network->get_node(j));
            const double dx(node_i->x() - node_j.x());
            const double dy(node_i->y() - node_j.y());
            const double dist(dx * dx + dy * dy);

            neighs.emplace(-dist, j);
        }

        bool linked(false);
        for (uint32_t j = 0; j < num_neigh; ++j) {
            const uint32_t new_neigh(neighs.top().second);
            neighs.pop();

            // adjacency matrix
            network->adj_(i, new_neigh) = 1;
            network->adj_(new_neigh, i) = 1;

            // neighbor list
            node_i->add_neigh(new_neigh);
            network->node_list_.mutable_nodes(new_neigh)->add_neigh(i);

            if (new_neigh < i) {
                const uint32_t new_neigh_subnet(subnet_by_node.at(new_neigh));
                subnets.at(new_neigh_subnet).insert(i);
                if (!linked) {
                    subnet_by_node.emplace(i, new_neigh_subnet);
                }
                linked = true;
            }
        }

        // could not find link, need to add new subnet
        if (!linked) {
            // set index of subnet
            subnet_by_node.emplace(i, subnets.size());

            // create new subnet
            std::set<uint32_t> new_subnet;
            new_subnet.insert(i);
            subnets.push_back(new_subnet);
        }
    }

    // subnets need to be merged if they have common neighbors
    bool merged = false;
    uint32_t num_subnets(subnets.size());
    do {
        merged = false;
        for (uint32_t i = 0; i < num_subnets; ++i) {
            const std::set<uint32_t> & subnet_i(subnets.at(i));
            for (uint32_t j = i + 1; j < num_subnets; ++j) {
                const std::set<uint32_t> & subnet_j(subnets.at(j));
                if (std::any_of(subnet_i.begin(), subnet_i.end(),
                                [&] (const uint32_t & index) {
                                    return subnet_j.count(index) > 0;
                                })) {
                    merged = true;
                    // combine subnet j into subnet i
                    subnets.at(i).insert(subnet_j.begin(), subnet_j.end());
                    // swap subnet j for the last subnet
                    subnets.at(j) = subnets.at(num_subnets - 1);
                    // decrement effective size
                    --num_subnets;

                    break;
                }
            }
            if (merged) {
                break;
            }
        }
    } while(merged);

    // trim vector
    CHECK_GT(num_subnets, 0);
    subnets.resize(num_subnets);

    // combine subnets
    while (subnets.size() > 1) {
        std::set<uint32_t> & curr(subnets.at(0));
        uint32_t curr_index = size;
        uint32_t next_index = size;
        uint32_t subnet_index = subnets.size();
        double smallest_dist = std::numeric_limits<double>::infinity();
        for (uint32_t i = 1; i < subnets.size(); i++) {
            std::set<uint32_t> & next(subnets.at(i));
            for (auto currit = curr.begin(); currit != curr.end(); ++currit) {
                const Node & curr_node(network->get_node(*currit));
                for (auto nextit = next.begin(); nextit != next.end();
                     ++nextit) {
                    const Node & next_node(network->get_node(*nextit));
                    const double diff_x(curr_node.x() - next_node.x());
                    const double diff_y(curr_node.y() - next_node.y());
                    const double dist(diff_x * diff_x + diff_y * diff_y);

                    if (dist < smallest_dist) {
                        smallest_dist = dist;
                        curr_index = *currit;
                        next_index = *nextit;
                        subnet_index = i;
                    }
                }
            }
        }

        // set adjancey
        network->adj_(curr_index, next_index) = 1;
        network->adj_(next_index, curr_index) = 1;

        // add neighbors
        network->node_list_.mutable_nodes(curr_index)->add_neigh(next_index);
        network->node_list_.mutable_nodes(next_index)->add_neigh(curr_index);

        // combine subnets
        curr.insert(subnets.at(subnet_index).begin(),
                subnets.at(subnet_index).end());
        subnets.at(subnet_index) = subnets.at(subnets.size() - 1);
        subnets.resize(subnets.size() - 1);
    }

    network->dist_ = network->calc_dist();

    return network;
}


std::shared_ptr<Network> Network::gen_ebola() {
    std::shared_ptr<Network> network = std::shared_ptr<Network>(new Network());

    network->kind_ = "ebola";
    network->num_nodes_ = EbolaData::loc().size();
    network->adj_ = boost::numeric::ublas::mapped_matrix<uint32_t>(
            network->num_nodes_, network->num_nodes_,
            // allocate memory because this is a fully connected
            // network
            network->num_nodes_ * network->num_nodes_);

    const std::vector<double> & ebola_x(EbolaData::x());
    const std::vector<double> ebola_y(EbolaData::y());

    for (uint32_t i = 0; i < network->num_nodes_; ++i) {
        Node * const n(network->node_list_.add_nodes());
        n->set_x(ebola_x.at(i));
        n->set_y(ebola_y.at(i));

        // index
        n->set_index(i);

        // add distance and neighbors
        std::vector<double> i_dist;
        i_dist.reserve(network->num_nodes_);
        for (uint32_t j = 0; j < network->num_nodes_; ++j) {
            if (j != i) {
                // distance from i to j
                const double diff_x(ebola_x.at(i) - ebola_x.at(j));
                const double diff_y(ebola_y.at(i) - ebola_y.at(j));
                i_dist.push_back(std::sqrt(diff_x * diff_x + diff_y * diff_y));
            } else {
                // distance from i to itself
                i_dist.push_back(0.0);
            }
        }

        // add distance vector to matrix
        network->dist_.push_back(std::move(i_dist));
    }

    // adjacency
    for (uint32_t i = 0; i < EbolaData::edges().size(); ++i) {
        const std::pair<uint32_t, uint32_t> & edge(EbolaData::edges().at(i));

        Node * const a(network->node_list_.mutable_nodes(edge.first));
        Node * const b(network->node_list_.mutable_nodes(edge.second));

        a->add_neigh(edge.second);
        b->add_neigh(edge.first);

        network->adj_(edge.first, edge.second) = 1;
        network->adj_(edge.second, edge.first) = 1;
    }

    // sort neighbors
    for (uint32_t i = 0; i < network->size(); ++i) {
        Node * const node(network->node_list_.mutable_nodes(i));
        std::sort(node->mutable_neigh()->begin(), node->mutable_neigh()->end());
    }

    return network;
}





const NodeList & Network::node_list() const {
    return this->node_list_;
}


} // namespace stdmMf
