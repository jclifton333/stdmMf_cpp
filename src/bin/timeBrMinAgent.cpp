#include "system.hpp"
#include "infStateNoSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"

#include "networkRunFeatures.hpp"

#include "objFns.hpp"

#include <thread>

using namespace stdmMf;

int main(int argc, char *argv[]) {

    NetworkInit init;
    init.set_dim_x(25);
    init.set_dim_y(20);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> net(Network::gen_network(init));

    const std::shared_ptr<Model<InfState> > mod(new InfStateNoSoModel(net));
    mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});

    // br min
    System<InfState> s(net->clone(), mod->clone());
    s.seed(0);
    BrMinSimPerturbAgent<InfState> a(net->clone(),
            std::shared_ptr<Features<InfState> >(
                    new NetworkRunFeatures<InfState>(net->clone(), 4)),
            mod->clone(),
            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5, false, false, false, 0, 0, 0, 0);
    a.seed(0);

    s.start();

    runner(&s, &a, 20, 1.0);

    return 0;
}
