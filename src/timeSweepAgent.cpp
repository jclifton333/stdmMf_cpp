#include "sweepAgent.hpp"
#include "networkRunFeatures.hpp"
#include "network.hpp"
#include "random.hpp"
#include <chrono>
#include <fstream>
#include <glog/logging.h>

using namespace stdmMf;

int main(int argc, char *argv[]) {
    // setup network
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Rng> rng(new Rng);
    std::shared_ptr<Network> net = Network::gen_network(init);

    std::shared_ptr<Features> f(new NetworkRunFeatures(net,4));

    std::vector<double> coef(f->num_features());
    std::for_each(coef.begin(), coef.end(),
            [&rng](double & x) {
                x = rng->rnorm_01();
            });

    SweepAgent sa(net, f, coef, 2);
    sa.set_rng(rng);

    boost::dynamic_bitset<> inf_bits(net->size());
    inf_bits.set(0);
    inf_bits.set(1);
    inf_bits.set(3);
    inf_bits.set(10);
    inf_bits.set(11);
    inf_bits.set(30);
    inf_bits.set(20);
    inf_bits.set(23);
    inf_bits.set(19);

    std::vector<boost::dynamic_bitset<> > trt_bits;
    const std::chrono::time_point<std::chrono::high_resolution_clock> tick =
        std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 100; ++i) {
        trt_bits.push_back(sa.apply_trt(inf_bits));
    }
    const std::chrono::time_point<std::chrono::high_resolution_clock> tock =
        std::chrono::high_resolution_clock::now();

    std::cout << "elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                      tock - tick).count()
              << std::endl;

    std::ofstream out;
    out.open("coef.txt");
    if (out.good()) {
        for (uint32_t i = 0; i < coef.size(); ++i) {
            out << coef.at(i) << std::endl;
        }
        out.close();
    } else {
        LOG(FATAL) << "failed to open";
    }

    // out.open("trt_bits.txt");
    // if (out.good()) {
    //     for (uint32_t i = 0; i < trt_bits.size(); ++i) {
    //         out << trt_bits.at(i).to_ulong() << std::endl;
    //     }
    //     out.close();
    // } else {
    //     LOG(FATAL) << "failed to open";
    // }

    return 0;
}
