#include "network.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/info/project.hpp>

using namespace stdmMf;


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

    {// random 100
        NetworkInit init;
        init.set_type(NetworkInit_NetType_RANDOM);
        init.set_size(100);
        inits.push_back(init);
    }

    {// random 500
        NetworkInit init;
        init.set_type(NetworkInit_NetType_RANDOM);
        init.set_size(500);
        inits.push_back(init);
    }

    {// random 1000
        NetworkInit init;
        init.set_type(NetworkInit_NetType_RANDOM);
        init.set_size(1000);
        inits.push_back(init);
    }

    // write networks to csv
    for (auto it = inits.begin(); it != inits.end(); ++it) {
        const auto network(Network::gen_network(*it));
        std::string str;
        network->node_list().SerializeToString(&str);
        *tk.entry(network->kind() + ".pb") << str;
        std::cout << "saved " << network->kind() << std::endl;
    }

    tk.finished();
    tk.print_data_dir();

    return 0;
}
