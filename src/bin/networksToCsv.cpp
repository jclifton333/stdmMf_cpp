#include "network.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/info/project.hpp>

using namespace stdmMf;


std::string grid_to_csv(const std::shared_ptr<Network> & network) {
    std::stringstream ss;
    ss << "index_a,index_b,x_a,y_a,x_b,y_b\n";
    const auto edges(network->edges());
    for (auto it = edges.begin(); it != edges.end(); ++it) {
        const Node & node_a(network->get_node(it->first));
        const Node & node_b(network->get_node(it->second));

        ss << node_a.index() << ","
           << node_b.index() << ","
           << node_a.x() << ","
           << node_a.y() << ","
           << node_b.x() << ","
           << node_b.y() << "\n";
    }
    return ss.str();
}

std::string barabasi_to_csv(const std::shared_ptr<Network> & network) {
    std::stringstream ss;
    ss << "index_a,index_b,dist\n";
    const auto dist_mat(network->dist());
    for (uint32_t i = 0; i < network->size(); ++i) {
        for (uint32_t j = 0; j < network->size(); ++j) {
            const Node & node_a(network->get_node(i));
            const Node & node_b(network->get_node(j));

            ss << node_a.index() << ","
               << node_b.index() << ","
               << dist_mat.at(node_a.index()).at(node_b.index()) << "\n";
        }
    }

    return ss.str();
}


int main(int argc, char *argv[])
{

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");

    std::vector<NetworkInit> inits;

    {// grid 100
        NetworkInit init;
        init.set_type(NetworkInit_NetType_GRID);
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        inits.push_back(init);
    }

    {// grid 500
        NetworkInit init;
        init.set_type(NetworkInit_NetType_GRID);
        init.set_dim_x(20);
        init.set_dim_y(25);
        init.set_wrap(false);
        inits.push_back(init);
    }

    {// grid 1000
        NetworkInit init;
        init.set_type(NetworkInit_NetType_GRID);
        init.set_dim_x(40);
        init.set_dim_y(25);
        init.set_wrap(false);
        inits.push_back(init);
    }

    {// barabasi 100
        NetworkInit init;
        init.set_type(NetworkInit_NetType_BARABASI);
        init.set_size(100);
        inits.push_back(init);
    }

    {// barabasi 500
        NetworkInit init;
        init.set_type(NetworkInit_NetType_BARABASI);
        init.set_size(500);
        inits.push_back(init);
    }

    {// barabasi 1000
        NetworkInit init;
        init.set_type(NetworkInit_NetType_BARABASI);
        init.set_size(1000);
        inits.push_back(init);
    }

    // write networks to csv
    for (auto it = inits.begin(); it != inits.end(); ++it) {
        const auto network(Network::gen_network(*it));
        if (it->type() == NetworkInit_NetType_GRID) {
            *tk.entry(network->kind() + ".csv") << grid_to_csv(network);
        } else if(it->type() == NetworkInit_NetType_BARABASI) {
            *tk.entry(network->kind() + ".csv") << barabasi_to_csv(network);
        }
        std::cout << "saved " << network->kind() << std::endl;
    }

    return 0;
}
