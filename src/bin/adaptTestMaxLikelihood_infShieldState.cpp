#include "adaptTestData.pb.h"

#include "system.hpp"

#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"

#include <njm_cpp/tools/progress.hpp>

#include <njm_cpp/thread/pool.hpp>

#include <njm_cpp/info/project.hpp>

#include <iterator>
#include <algorithm>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
namespace ba = boost::accumulators;

#include <regex>

#include <condition_variable>

#include <iostream>
#include <fstream>

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>

#include <glog/logging.h>

using namespace stdmMf;


std::vector<Transition<InfShieldState> > obs_to_bitset_vector(
        const Observation & obs) {
    std::vector<Transition<InfShieldState> > bitset_vector;
    for (uint32_t i = 0; i < obs.num_points(); ++i) {
        const InfShieldState curr_state(
                boost::dynamic_bitset<>(
                        obs.transition(i).curr_state().inf_bits()),
                std::vector<double>(
                        obs.transition(i).curr_state().shield().begin(),
                        obs.transition(i).curr_state().shield().end()));

        const boost::dynamic_bitset<> trt_bits(
                obs.transition(i).curr_trt_bits());

        const InfShieldState next_state(
                boost::dynamic_bitset<>(
                        obs.transition(i).next_state().inf_bits()),
                std::vector<double>(
                        obs.transition(i).next_state().shield().begin(),
                        obs.transition(i).next_state().shield().end()));

        bitset_vector.emplace_back(curr_state, trt_bits, next_state);
    }

    return bitset_vector;
}


double get_statistic(
        const std::vector<Transition<InfShieldState> > & history,
        const std::shared_ptr<Network> & net,
        std::shared_ptr<Model<InfShieldState> > & mod) {
    // zero-out parameter values
    mod->par(std::vector<double>(mod->par_size(), 0.0));
    // estimate MLE
    mod->est_par(history);
    // evaluate likelihood at MLE
    return mod->ll(history);
}


std::vector<double> get_null_distribution(
        const std::vector<Transition<InfShieldState> > & history,
        const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod) {
    System<InfShieldState> s(net, mod->clone());

    const uint32_t num_reps = 100;
    std::vector<double> null_dist;
    for (uint32_t rep = 0; rep < num_reps; ++rep) {
        s.seed(rep);

        std::vector<Transition<InfShieldState> > sim_history;
        for (uint32_t t = 0; t < history.size(); ++t) {
            s.state(history.at(t).curr_state);
            s.trt_bits(history.at(t).curr_trt_bits);

            // sim under dynamics mod
            s.turn_clock();

            // record transition
            sim_history.emplace_back(history.at(t).curr_state,
                    history.at(t).curr_trt_bits, s.state());
        }

        std::shared_ptr<Model<InfShieldState> > mod_est(mod->clone());
        mod_est->seed(rep);
        null_dist.push_back(get_statistic(sim_history, net, mod_est));
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

std::shared_ptr<Model<InfShieldState> > get_model(
        const std::string & model_kind,
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

    const double shield_coef = 0.9;

    std::vector<double> par =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         trt_act_rec,
         trt_pre_inf,
         shield_coef};

    std::vector<double> par_sep =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         -trt_act_inf,
         trt_act_rec,
         -trt_act_rec,
         trt_pre_inf,
         -trt_pre_inf,
         shield_coef};

    std::shared_ptr<Model<InfShieldState> > mod;
    if (model_kind == "NoImNoSo") {
        mod = std::shared_ptr<Model<InfShieldState> >(
                new InfShieldStateNoImNoSoModel(net->clone()));
        mod->par(par);
    } else if (model_kind == "PosImNoSo") {
        mod = std::shared_ptr<Model<InfShieldState> >(
                new InfShieldStatePosImNoSoModel(net->clone()));
        mod->par(par);
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

    const uint32_t num_points_for_fit = 20;
    const uint32_t num_skip = 0;

    for (uint32_t sim = 0; sim < ad.sim_size(); ++sim) {
        const SimData & sd = ad.sim(sim);

        ba::accumulator_set<double, ba::stats<
            ba::tag::mean, ba::tag::variance> > vals;

        uint32_t num_left = sd.rep_size();
        std::mutex mtx;
        std::condition_variable cv;

        for (uint32_t rep = 0; rep < sd.rep_size(); ++rep) {
            // const Observation obs = sd.rep(rep);
            const Observation * obs(&sd.rep(rep));

            auto fn = [=,&vals,&mtx,&num_left,&cv] () {
                // get network
                std::shared_ptr<Network> net(get_network(sd.network()));

                // get model
                std::regex model_regex("^model_([a-zA-Z]*)_([a-zA-Z]*)$");
                std::smatch model_match;
                std::regex_search(sd.model(), model_match, model_regex);

                std::shared_ptr<Model<InfShieldState> > mod_agent(
                        get_model(model_match.str(2), net));

                // convert observation to history
                const std::vector<Transition<InfShieldState> > history =
                    obs_to_bitset_vector(*obs);
                // const std::vector<InfAndTrt> history =
                //     obs_to_bitset_vector(obs);

                ////////////////////
                // observed data

                // history for fitting model
                std::vector<Transition<InfShieldState> > fit_history(
                        history.begin() + num_skip,
                        history.begin() + num_points_for_fit + num_skip);
                CHECK_EQ(fit_history.size(), num_points_for_fit);

                // fit model
                mod_agent->est_par(fit_history);
                const double obs_statistic = mod_agent->ll(fit_history);

                ////////////////////
                // simulated data

                // observed data for evaluation
                std::vector<Transition<InfShieldState> > eval_history(
                        fit_history.begin(), fit_history.end() - 1);
                CHECK_EQ(eval_history.size(), num_points_for_fit - 1);

                // get null distribution
                std::vector<double> null_distribution = get_null_distribution(
                        eval_history, net, mod_agent);

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

            pool.service().post(fn);

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
