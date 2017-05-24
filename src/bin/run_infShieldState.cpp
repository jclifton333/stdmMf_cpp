#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "mixtureModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"
#include "brMinIterSimPerturbAgent.hpp"

#include "brMinWtdSimPerturbAgent.hpp"

#include "networkRunSymFeatures.hpp"
#include "finiteQfnFeatures.hpp"

#include "objFns.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/stats.hpp>

#include <njm_cpp/tools/progress.hpp>

#include <future>

#include <thread>

#include <fstream>

#include <chrono>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using namespace stdmMf;

using namespace boost::accumulators;

using njm::tools::mean_and_var;

typedef std::pair<std::shared_ptr<Model<InfShieldState> >,
                  std::shared_ptr<Model<InfShieldState> > > ModelPair;

struct Outcome {
    double value;
    double time;
    std::vector<StateAndTrt<InfShieldState> > history;
};

template <template <typename> class T>
using OutcomeReps = std::vector<T<Outcome> >;

template <template <typename> class T>
using Results = std::map<std::string, OutcomeReps<T> >;

template <template <typename> class T>
struct ModelResults {
    std::string model_kind;
    Results<T> results;
};

template <template <typename> class T>
struct NetworkResults {
    std::string network_kind;
    std::vector<ModelResults<T> > results;
};

template <template <typename> class T>
using AllResults = std::vector<NetworkResults<T> >;


std::string history_to_csv_entry(
        const std::string & agent, const uint32_t & rep,
        const std::vector<StateAndTrt<InfShieldState> > & history) {
    const uint32_t num_points = history.size();
    std::stringstream ss;
    for (uint32_t i = 0; i < num_points; ++i) {

        const InfShieldState & state(history.at(i).state);
        const boost::dynamic_bitset<> & trt_bits(history.at(i).trt_bits);

        const uint32_t num_nodes(trt_bits.size());

        for (uint32_t j = 0; j < num_nodes; ++j) {
            const uint32_t j_inf(static_cast<uint32_t>(state.inf_bits.test(j)));
            const uint32_t j_trt(static_cast<uint32_t>(trt_bits.test(j)));

            ss << agent << ", " << rep << ", " << i << ", " << j << ", "
               << j_inf << ", " << state.shield.at(j) << ", " << j_trt << "\n";
        }
    }
    return ss.str();
}


void queue_sim(
        njm::thread::Pool * const pool,
        njm::tools::Progress<std::ostream> * const progress,
        ModelResults<std::promise> * const results,
        const std::shared_ptr<const Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & time_points) {

    // none
    CHECK_EQ(results->results.count("none"), 1);
    CHECK_EQ(results->results.at("none").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=](){
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);
            NoTrtAgent<InfShieldState> a(net);
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("none").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }

    // random
    CHECK_EQ(results->results.count("random"), 1);
    CHECK_EQ(results->results.at("random").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=](){
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);
            RandomAgent<InfShieldState> a(net);
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("random").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }


    // proximal
    CHECK_EQ(results->results.count("proximal"), 1);
    CHECK_EQ(results->results.at("proximal").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);
            ProximalAgent<InfShieldState> a(net);
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("proximal").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }


    // myopic
    CHECK_EQ(results->results.count("myopic"), 1);
    CHECK_EQ(results->results.at("myopic").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);
            MyopicAgent<InfShieldState> a(net,
                    mod_agents->clone());
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("myopic").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }


    // vfn max finite q
    CHECK_EQ(results->results.count("vfn_finite_q"), 1);
    CHECK_EQ(results->results.at("vfn_finite_q").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<InfShieldState> a(net,
                    std::shared_ptr<Features<InfShieldState> >(
                            new FiniteQfnFeatures<InfShieldState>(
                                    net, {mod_agents->clone()},
                                    std::shared_ptr<Features<InfShieldState> >(
                                            new NetworkRunSymFeatures
                                            <InfShieldState>(
                                                    net, 2)), 1)),
                    mod_agents->clone(),
                    2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("vfn_finite_q").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }


    // br min finite q
    CHECK_EQ(results->results.count("br_finite_q"), 1);
    CHECK_EQ(results->results.at("br_finite_q").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<InfShieldState> s(net, mod_system->clone());
            s.seed(i);

            std::shared_ptr<Model<InfShieldState> > modNoIm(
                    new InfShieldStateNoImNoSoModel(net));
            std::shared_ptr<Model<InfShieldState> > modPosIm(
                    new InfShieldStatePosImNoSoModel(net));

            BrMinSimPerturbAgent<InfShieldState> a(net,
                    std::shared_ptr<Features<InfShieldState> >(
                            new FiniteQfnFeatures<InfShieldState>(
                                    net, {modNoIm, modPosIm},
                                    std::shared_ptr<Features<InfShieldState> >(
                                            new NetworkRunSymFeatures<
                                            InfShieldState>(
                                                    net, 2)), 1)),
                    mod_agents->clone(),
                    0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
                    true, true, false, 0, 0, 0, 0);
            a.seed(i);

            s.start();

            Outcome outcome;

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            outcome.value = runner(&s, &a, time_points, 1.0);

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            outcome.time = std::chrono::duration_cast<
                std::chrono::seconds>(tock - tick).count();

            outcome.history = s.history();
            outcome.history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));

            results->results.at("br_finite_q").at(i).set_value(
                    std::move(outcome));
            progress->update();
        });
    }
}


void queue_all_sims(
        njm::thread::Pool * const pool,
        njm::tools::Progress<std::ostream> * const progress,
        AllResults<std::promise> * const results,
        const std::vector<std::shared_ptr<const Network> > & networks,
        const std::vector<std::pair<std::string,
        std::vector<ModelPair> > > & models,
        const uint32_t & num_reps,
        const uint32_t & time_points) {

    CHECK_EQ(networks.size(), results->size());
    for (uint32_t i = 0; i < networks.size(); ++i) {
        CHECK_EQ(results->at(i).results.size(), models.size());

        for (uint32_t j = 0; j < models.size(); ++j) {
            queue_sim(pool, progress, & results->at(i).results.at(j),
                    networks.at(i),
                    models.at(j).second.at(i).first,
                    models.at(j).second.at(i).second,
                    num_reps, time_points);
        }
    }
}


void process_results(
        njm::data::TrapperKeeper & tk,
        AllResults<std::future> & all_results,
        const std::vector<std::string> & agent_names) {

    njm::data::Entry * const e_read_all = tk.entry("read.txt");
    njm::data::Entry * const e_raw_all = tk.entry("raw.txt");
    *e_raw_all << "network" << ","
               << "model" << ","
               << "agent" << ","
               << "value_mean" << ","
               << "value_ssd" << ","
               << "time_mean" << "\n";

    for (uint32_t i = 0; i < all_results.size(); ++i) {
        NetworkResults<std::future> & nr(all_results.at(i));
        const std::string network_kind(nr.network_kind);

        njm::data::Entry * const e_read_net = tk.entry(
                network_kind + "_read.txt");
        njm::data::Entry * const e_raw_net = tk.entry(
                network_kind + "_raw.txt");
        *e_raw_net << "network" << ","
                   << "model" << ","
                   << "agent" << ","
                   << "value_mean" << ","
                   << "value_ssd" << ","
                   << "time_mean" << "\n";

        for (uint32_t j = 0; j < nr.results.size(); ++j) {
            ModelResults<std::future> & mr(nr.results.at(j));
            const std::string model_kind(mr.model_kind);

            njm::data::Entry * const e_read_mod = tk.entry(
                    network_kind + "_" + model_kind + "_read.txt");
            njm::data::Entry * const e_raw_mod = tk.entry(
                    network_kind + "_" + model_kind + "_raw.txt");
            *e_raw_mod << "network" << ","
                       << "model" << ","
                       << "agent" << ","
                       << "value_mean" << ","
                       << "value_ssd" << ","
                       << "time_mean" << "\n";

            njm::data::Entry * e_history = tk.entry(
                    network_kind + "_" + model_kind + "_history.txt");
            *e_history << "agent, rep, time, node, inf, shield, trt\n";


            *e_read_all << "results for network " << network_kind
                        << " and model pair " << model_kind << "\n";

            *e_read_net << "results for network " << network_kind
                        << " and model pair " << model_kind << "\n";

            *e_read_mod << "results for network " << network_kind
                        << " and model pair " << model_kind << "\n";

            Results<std::future> & r(mr.results);
            std::vector<std::string>::const_iterator it;
            for (it = agent_names.begin(); it != agent_names.end(); ++it) {
                const std::string agent_kind(*it);

                OutcomeReps<std::future> & reps(r.at(agent_kind));

                accumulator_set<double, stats<tag::mean, tag::variance> >
                    values, times;
                for (uint32_t k = 0; k < reps.size(); ++k) {
                    const Outcome & outcome (reps.at(k).get());
                    values(outcome.value);
                    times(outcome.time);

                    const std::string history_str(history_to_csv_entry(
                                    agent_kind, k, outcome.history));
                    *e_history << history_str;
                }
                CHECK_GT(reps.size(), 1);

                const double value_mean(mean(values));
                const double re_scale(static_cast<double>(reps.size())
                        / (reps.size() - 1));
                const double value_sse(std::sqrt(variance(values) * re_scale)
                        / std::sqrt(static_cast<double>(reps.size())));
                const double time_mean(mean(times));

                std::stringstream raw_ss;
                raw_ss << network_kind << ","
                       << model_kind << ","
                       << agent_kind << ","
                       << std::to_string(value_mean) << ","
                       << std::to_string(value_sse) << ","
                       << std::to_string(time_mean) << "\n";

                std::stringstream read_ss;
                read_ss << std::setw(16) << std::right
                        << agent_kind << ": "
                        << std::setw(8) << std::right
                        << std::setprecision(3) << std::fixed
                        << value_mean << " ("
                        << std::setw(8) << std::right
                        << std::setprecision(4) << std::fixed
                        << value_sse << ")  ["
                        << std::setw(8) << std::setprecision(0)
                        << time_mean << "]"
                        << "\n";

                *e_raw_all << raw_ss.str();
                *e_read_all << read_ss.str();

                *e_raw_net << raw_ss.str();
                *e_read_net << read_ss.str();

                *e_raw_mod << raw_ss.str();
                *e_read_mod << read_ss.str();
            }

            // line separators
            *e_read_all << "====================================="
                        << "====================================="
                        << "\n";

            *e_read_net << "====================================="
                        << "====================================="
                        << "\n";

            *e_read_mod << "====================================="
                        << "====================================="
                        << "\n";

            // flush trapper keeper
            tk.flush();
        }
    }
}



int main(int argc, char *argv[]) {
    // gflags::ParseCommandLineFlags(&argc, &argv, true);
    // google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);

    // setup networks
    std::vector<std::shared_ptr<const Network> > networks;
    { // grid network 100
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // barabasi 100
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    { // random 100
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    { // grid network 500
        NetworkInit init;
        init.set_dim_x(20);
        init.set_dim_y(25);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // barabasi 500
        NetworkInit init;
        init.set_size(500);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    { // random 500
        NetworkInit init;
        init.set_size(500);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    { // grid network 1000
        NetworkInit init;
        init.set_dim_x(40);
        init.set_dim_y(25);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // barabasi 1000
        NetworkInit init;
        init.set_size(1000);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    { // random 1000
        NetworkInit init;
        init.set_size(1000);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    // double vector since model depends on network
    std::vector<std::pair<std::string,
                          std::vector<ModelPair> > > models;
    { // models
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

        // shield
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


        { // Correct: NoIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("NoImNoSo-NoImNoSo",
                            models_add));
        }

        { // Correct: PosIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("PosImNoSo-PosImNoSo",
                            models_add));
        }

        { // Correct: PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("PosImNoSo-NoImNoSo",
                            models_add));
        }

        { // Correct: NoIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("NoImNoSo-PosImNoSo",
                            models_add));
        }

        { // Correct: 0.25 NoIm NoSo + 0.75 PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                // set par in advance because mixture model doesn't
                // implement par functionality
                auto mod_one(std::make_shared<InfShieldStateNoImNoSoModel>(
                                networks.at(i)));
                mod_one->par(par);
                auto mod_two(std::make_shared<InfShieldStatePosImNoSoModel>(
                                networks.at(i)));
                mod_two->par(par);
                ModelPair mp (std::shared_ptr<MixtureModel<InfShieldState,
                        InfShieldStateModel> >(
                                new MixtureModel<InfShieldState,
                                InfShieldStateModel>(
                                {mod_one, mod_two}, {0.25, 0.75},
                                networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Mixture-25-75-NoImNoSo-PosImNoSo",
                            models_add));
        }

        { // Correct: 0.50 NoIm NoSo + 0.50 PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                // set par in advance because mixture model doesn't
                // implement par functionality
                auto mod_one(std::make_shared<InfShieldStateNoImNoSoModel>(
                                networks.at(i)));
                mod_one->par(par);
                auto mod_two(std::make_shared<InfShieldStatePosImNoSoModel>(
                                networks.at(i)));
                mod_two->par(par);
                ModelPair mp (std::shared_ptr<MixtureModel<InfShieldState,
                        InfShieldStateModel> >(
                                new MixtureModel<InfShieldState,
                                InfShieldStateModel>(
                                {mod_one, mod_two}, {0.50, 0.50},
                                networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Mixture-50-50-NoImNoSo-PosImNoSo",
                            models_add));
        }

        { // Correct: 0.75 NoIm NoSo + 0.25 PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                // set par in advance because mixture model doesn't
                // implement par functionality
                auto mod_one(std::make_shared<InfShieldStateNoImNoSoModel>(
                                networks.at(i)));
                mod_one->par(par);
                auto mod_two(std::make_shared<InfShieldStatePosImNoSoModel>(
                                networks.at(i)));
                mod_two->par(par);
                ModelPair mp (std::shared_ptr<MixtureModel<InfShieldState,
                        InfShieldStateModel> >(
                                new MixtureModel<InfShieldState,
                                InfShieldStateModel>(
                                {mod_one, mod_two}, {0.75, 0.25},
                                networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Mixture-75-25-NoImNoSo-PosImNoSo",
                            models_add));
        }
    }

    const uint32_t num_reps = 50;
    const uint32_t time_points = 25;

    // set up results containers
    const std::vector<std::string> agent_names({
                "none", "random", "proximal", "myopic",
                "vfn_finite_q", "br_finite_q"});
    AllResults<std::promise> promise_results;
    AllResults<std::future> future_results;

    promise_results.resize(networks.size());
    future_results.resize(networks.size());
    for (uint32_t i = 0; i < networks.size(); ++i) {

        promise_results.at(i).network_kind = networks.at(i)->kind();
        future_results.at(i).network_kind = networks.at(i)->kind();

        promise_results.at(i).results.resize(models.size());
        future_results.at(i).results.resize(models.size());

        for (uint32_t j = 0; j < models.size(); ++j) {

            promise_results.at(i).results.at(j).model_kind = models.at(j).first;
            future_results.at(i).results.at(j).model_kind = models.at(j).first;

            for (uint32_t k = 0; k < agent_names.size(); ++k) {

                // instantiate promises
                promise_results.at(i).results.at(j).results[
                        agent_names.at(k)].resize(num_reps);

                OutcomeReps<std::promise> & promise_reps(
                        promise_results.at(i).results.at(j).results[
                                agent_names.at(k)]);

                // instantiate futures from promises
                OutcomeReps<std::future> & future_reps(
                        future_results.at(i).results.at(j).results[
                        agent_names.at(k)]);
                future_reps.reserve(num_reps);
                for (uint32_t r = 0; r < num_reps; ++r) {
                    future_reps.emplace_back(
                            promise_reps.at(r).get_future());
                }
            }
        }
    }

    // set up tools for sims
    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");
    tk.print_data_dir();

    njm::tools::Progress<std::ostream> progress(
            networks.size() * models.size() * agent_names.size() * num_reps,
            &std::cout);

    njm::thread::Pool pool(std::thread::hardware_concurrency());

    // queue sims
    queue_all_sims(&pool, &progress, &promise_results,
            networks, models, num_reps, time_points);

    // process results
    process_results(tk, future_results, agent_names);

    progress.done();

    tk.finished();
    tk.print_data_dir();

    return 0;
}
