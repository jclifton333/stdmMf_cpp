#include "adaptTestData.pb.h"

#include "system.hpp"

#include "noCovEdgeModel.hpp"
#include "noCovEdgeOrSoModel.hpp"
#include "noCovEdgeXorSoModel.hpp"
#include "noCovEdgeSepSoModel.hpp"

#include "progress.hpp"

#include "projectInfo.hpp"

#include <regex>

#include <iostream>
#include <fstream>

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>

#include <glog/logging.h>

using namespace stdmMf;


std::vector<double> get_null_distribution(const Observation & obs,
        const std::shared_ptr<Model> & mod) {
}

double get_statistic(const Observation & obs,
        const std::shared_ptr<Model> & mod) {


}


std::shared_ptr<Network> get_network(const std::string & network_kind) {
    std::shared_ptr<Network> net;
    if (network_kind == "grid_10x10") {
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);

        net = Network::gen_network(init);
    } else if(network_kind == "barabasi_100") {
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_BARABASI);

        net = Network::gen_network(init);
    } else {
        LOG(FATAL) << "Don't know how to handle network kind: "
                   << network_kind;
    }

    CHECK_EQ(network_kind, net->kind()) << "Did not recover network";

    return net;
}

std::shared_ptr<Model> get_model(const std::string & model_kind,
        const std::shared_ptr<Network> & net) {

    // latent infections
    const double prob_inf_latent = 0.01;
    const double intcp_inf_latent =
        std::log(1. / (1. - prob_inf_latent) - 1);

    // neighbor infections
    const double prob_inf = 0.5;
    const uint32_t prob_num_neigh = 3;
    const double intcp_inf =
        std::log(std::pow(1. - prob_inf, -1. / prob_num_neigh) - 1.);

    const double trt_act_inf =
        std::log(std::pow(1. - prob_inf * 0.25, -1. / prob_num_neigh) - 1.)
        - intcp_inf;

    const double trt_pre_inf =
        std::log(std::pow(1. - prob_inf * 0.75, -1. / prob_num_neigh) - 1.)
        - intcp_inf;

    // recovery
    const double prob_rec = 0.25;
    const double intcp_rec = std::log(1. / (1. - prob_rec) - 1.);
    const double trt_act_rec =
        std::log(1. / ((1. - prob_rec) * 0.5) - 1.) - intcp_rec;


    std::vector<double> par =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         trt_act_rec,
         trt_pre_inf};

    std::vector<double> par_sep =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         -trt_act_inf,
         trt_act_rec,
         -trt_act_rec,
         trt_pre_inf,
         -trt_pre_inf};

    std::shared_ptr<Model> mod;
    if (model_kind == "no") {
        mod = std::shared_ptr<Model>(new NoCovEdgeModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "or") {
        mod = std::shared_ptr<Model>(new NoCovEdgeOrSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "xor") {
        mod = std::shared_ptr<Model>(new NoCovEdgeXorSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "sep") {
        mod = std::shared_ptr<Model>(new NoCovEdgeSepSoModel(net->clone()));
        mod->par(par_sep);
    } else {
        LOG(FATAL) << "Don't know how to handle model kind: " << model_kind;
    }

    return mod;
}


int main(int argc, char *argv[]) {

    // read in data
    AdaptData ad;
    std::ifstream in(PROJECT_ROOT_DIR +
            "/data/2017-02-08_18-19-53/adapt_data.txt");
    CHECK(in.good());
    std::stringstream in_buf;
    in_buf << in.rdbuf();
    in.close();
    google::protobuf::TextFormat::ParseFromString(in_buf.str(), &ad);

    for (uint32_t sim = 0; sim < ad.sim_size(); ++sim) {
        const SimData & sd = ad.sim(sim);

        std::shared_ptr<Network> net(get_network(sd.network()));

        std::regex model_regex("^model_([a-zA-Z]*)_([a-zA-Z]*)$");
        std::smatch model_match;
        std::regex_search(sd.model(), model_match, model_regex);

        std::shared_ptr<Model> mod_system(get_model(model_match.str(1), net));
        std::shared_ptr<Model> mod_agent(get_model(model_match.str(2), net));


        for (uint32_t rep = 0; rep < sd.rep_size(); ++rep) {
            const Observation & obs = sd.rep(rep);

            double obs_statistic = get_statistic(obs, mod_agent);
            std::vector<double> null_distribution = get_null_distribution(obs,
                    mod_agent);
        }
    }

    return 0;
}
