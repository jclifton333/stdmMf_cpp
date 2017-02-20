#include "adaptTestData.pb.h"

#include "system.hpp"

#include "infStateNoSoModel.hpp"
#include "infStateOrSoModel.hpp"
#include "infStateXorSoModel.hpp"
#include "infStateSepSoModel.hpp"

#include <njm_cpp/tools/progress.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>

#include <iterator>
#include <algorithm>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
namespace ba = boost::accumulators;

#include <regex>

#include <iostream>
#include <fstream>

#include <mutex>
#include <condition_variable>

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>

#include <glog/logging.h>

using namespace stdmMf;


std::vector<Transition<InfState> > obs_to_bitset_vector(
        const Observation & obs) {
    CHECK_GT(obs.transition_size(), 0);

    std::vector<Transition<InfState> > bitset_vector;

    for (uint32_t i = 0; i < obs.transition_size(); ++i) {
        const InfState curr_state(
                boost::dynamic_bitset<>(obs.transition(i).curr_inf_bits()));
        const boost::dynamic_bitset<> curr_trt_bits(
                obs.transition(i).curr_trt_bits());
        const InfState next_state(
                boost::dynamic_bitset<>(obs.transition(i).next_inf_bits()));

        bitset_vector.emplace_back(curr_state, curr_trt_bits, next_state);
    }

    return bitset_vector;
}

std::vector<double> get_null_distribution(
        const std::vector<Transition<InfState> > & eval_history,
        const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfState> > & mod) {
    System<InfState> s(net, mod);

    const uint32_t num_reps = 100;
    std::vector<double> null_dist;
    for (uint32_t rep = 0; rep < num_reps; ++rep) {

        std::vector<Transition<InfState> > sim_eval_history;

        for (uint32_t t = 0; t < eval_history.size(); ++t) {
            s.state(eval_history.at(t).curr_state);
            s.trt_bits(eval_history.at(t).curr_trt_bits);

            // sim under dynamics mod
            s.turn_clock();

            // transition
            sim_eval_history.emplace_back(eval_history.at(t).curr_state,
                    eval_history.at(t).curr_trt_bits, s.state());

        }

        null_dist.push_back(mod->ll(sim_eval_history));
    }

    std::sort(null_dist.begin(), null_dist.end());
    return null_dist;
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

std::shared_ptr<Model<InfState> > get_model(const std::string & model_kind,
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

    std::shared_ptr<Model<InfState> > mod;
    if (model_kind == "no") {
        mod = std::shared_ptr<Model<InfState> >(
                new InfStateNoSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "or") {
        mod = std::shared_ptr<Model<InfState> >(
                new InfStateOrSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "xor") {
        mod = std::shared_ptr<Model<InfState> >(
                new InfStateXorSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "sep") {
        mod = std::shared_ptr<Model<InfState> >(
                new InfStateSepSoModel(net->clone()));
        mod->par(par_sep);
    } else {
        LOG(FATAL) << "Don't know how to handle model kind: " << model_kind;
    }

    return mod;
}


int main(int argc, char *argv[]) {
    // read in data
    AdaptData ad;
    std::ifstream in(njm::info::project::PROJECT_ROOT_DIR +
            "/data/2017-02-08_18-19-53/adapt_data.txt");
    CHECK(in.good());
    std::stringstream in_buf;
    in_buf << in.rdbuf();
    in.close();
    google::protobuf::TextFormat::ParseFromString(in_buf.str(), &ad);

    njm::thread::Pool pool;

    const uint32_t num_points_for_fit = 5;
    const uint32_t num_points_for_eval = 20;

    for (uint32_t sim = 0; sim < ad.sim_size(); ++sim) {
        const SimData & sd = ad.sim(sim);

        std::shared_ptr<Network> net(get_network(sd.network()));

        std::regex model_regex("^model_([a-zA-Z]*)_([a-zA-Z]*)$");
        std::smatch model_match;
        std::regex_search(sd.model(), model_match, model_regex);

        // std::shared_ptr<Model> mod_system(get_model(model_match.str(1), net));
        std::shared_ptr<Model<InfState> > mod_agent(
                get_model(model_match.str(2), net));


        ba::accumulator_set<double, ba::stats<
            ba::tag::mean, ba::tag::variance> > vals;

        uint32_t num_left = sd.rep_size();
        std::mutex mtx;
        std::condition_variable cv;

        for (uint32_t rep = 0; rep < sd.rep_size(); ++rep) {
            // const Observation obs = sd.rep(rep);
            const Observation * obs(&sd.rep(rep));

            auto fn = [=,&vals,&mtx,&num_left,&cv] () {
                const std::vector<Transition<InfState> > history =
                    obs_to_bitset_vector(*obs);
                // const std::vector<InfAndTrt> history =
                //     obs_to_bitset_vector(obs);

                // history for fitting model
                std::vector<Transition<InfState> > fit_history(history.begin(),
                        history.begin() + num_points_for_fit);
                CHECK_EQ(fit_history.size(), num_points_for_fit);

                // fit model
                std::shared_ptr<Model<InfState> > mod_agent_2 =
                mod_agent->clone();
                mod_agent_2->est_par(fit_history);

                // observed data for evaluation
                std::vector<Transition<InfState> > eval_history(
                        history.begin() + num_points_for_fit,
                        history.begin() + num_points_for_fit
                        + num_points_for_eval);
                CHECK_EQ(eval_history.size(), num_points_for_eval);

                // check tail and head of fit and eval
                CHECK_EQ(fit_history.at(
                                num_points_for_fit - 1).next_state.inf_bits,
                        eval_history.at(0).curr_state.inf_bits);

                const double obs_statistic = mod_agent_2->ll(eval_history);

                // history for null
                std::vector<Transition<InfState> > null_history(
                        eval_history.begin(),
                        eval_history.end());
                CHECK_EQ(null_history.size(), num_points_for_eval);

                // get null distribution
                std::vector<double> null_distribution = get_null_distribution(
                        null_history, net, mod_agent_2);

                // position of observed value in null distribution
                const std::vector<double>::iterator lb_it =
                std::lower_bound(null_distribution.begin(),
                        null_distribution.end(), obs_statistic);
                const uint32_t lb = std::distance(
                        null_distribution.begin(), lb_it);

                // convert to a percentile
                std::lock_guard<std::mutex> lock(mtx);
                vals(static_cast<double>(lb) /
                        static_cast<double>(null_distribution.size()));
                --num_left;
                cv.notify_one();
            };

            pool.service()->post(fn);

            // std::cout << "rep: " << rep << std::endl;
            // std::cout << "obs: " << obs_statistic << std::endl;
            // std::cout << "sim:";
            // for (uint32_t i = 0; i < null_distribution.size(); ++i) {
            //     std::cout << " " << null_distribution.at(i);
            // }
            // std::cout << std::endl;
        }

        std::unique_lock<std::mutex> finish_lock(mtx);
        cv.wait(finish_lock, [&num_left]{return num_left == 0;});


        std::cout << sd.model() << ": " << ba::mean(vals)
                  << " (" << ba::variance(vals) << ")"
                  << std::endl;
    }

    return 0;
}
