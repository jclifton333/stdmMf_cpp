#include "system.hpp"
#include "ebolaStateGravityModel.hpp"
#include "mixtureModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "allTrtAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"
#include "brMinIterSimPerturbAgent.hpp"

#include "brMinWtdSimPerturbAgent.hpp"

#include "ebolaData.hpp"
// #include "ebolaBinnedFeatures.hpp"
#include "ebolaModelFeatures.hpp"

#include "finiteQfnFeatures.hpp"

#include "objFns.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/stats.hpp>

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

typedef std::pair<std::shared_ptr<Model<EbolaState> >,
                  std::shared_ptr<Model<EbolaState> > > ModelPair;

struct Outcome {
    double value;
    double time;
    std::vector<StateAndTrt<EbolaState> > history;
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


void queue_sim(
        njm::thread::Pool * const pool,
        ModelResults<std::promise> * const results,
        const std::shared_ptr<const Network> & net,
        const std::shared_ptr<Model<EbolaState> > & mod_system,
        const std::shared_ptr<Model<EbolaState> > & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & time_points,
        const double & starting_prop) {

    // none
    CHECK_EQ(results->results.count("none"), 1);
    CHECK_EQ(results->results.at("none").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=](){
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            NoTrtAgent<EbolaState> a(net);
            a.seed(i);

            // sort outbreaks
            std::vector<uint32_t> outbreak_dates;
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0) {
                    outbreak_dates.push_back(EbolaData::outbreaks().at(i));
                }
            }
            std::sort(outbreak_dates.begin(), outbreak_dates.end());
            const uint32_t last_index(
                    std::min(std::max(1u,
                                    static_cast<uint32_t>(
                                            outbreak_dates.size()
                                            * starting_prop)),
                            static_cast<uint32_t>(outbreak_dates.size() - 1u)));
            const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

            EbolaState start_state(EbolaState(net->size()));
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0
                        && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
                    start_state.inf_bits.set(i);
                }
            }
            s.reset();
            s.state(start_state);

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
        });
    }

    // random
    CHECK_EQ(results->results.count("random"), 1);
    CHECK_EQ(results->results.at("random").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=](){
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            RandomAgent<EbolaState> a(net);
            a.seed(i);

            // sort outbreaks
            std::vector<uint32_t> outbreak_dates;
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0) {
                    outbreak_dates.push_back(EbolaData::outbreaks().at(i));
                }
            }
            std::sort(outbreak_dates.begin(), outbreak_dates.end());
            const uint32_t last_index(
                    std::min(std::max(1u,
                                    static_cast<uint32_t>(
                                            outbreak_dates.size()
                                            * starting_prop)),
                            static_cast<uint32_t>(outbreak_dates.size() - 1u)));
            const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

            EbolaState start_state(EbolaState(net->size()));
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0
                        && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
                    start_state.inf_bits.set(i);
                }
            }
            s.reset();
            s.state(start_state);

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
        });
    }


    // proximal
    CHECK_EQ(results->results.count("proximal"), 1);
    CHECK_EQ(results->results.at("proximal").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            ProximalAgent<EbolaState> a(net);
            a.seed(i);

            // sort outbreaks
            std::vector<uint32_t> outbreak_dates;
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0) {
                    outbreak_dates.push_back(EbolaData::outbreaks().at(i));
                }
            }
            std::sort(outbreak_dates.begin(), outbreak_dates.end());
            const uint32_t last_index(
                    std::min(std::max(1u,
                                    static_cast<uint32_t>(
                                            outbreak_dates.size()
                                            * starting_prop)),
                            static_cast<uint32_t>(outbreak_dates.size() - 1u)));
            const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

            EbolaState start_state(EbolaState(net->size()));
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0
                        && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
                    start_state.inf_bits.set(i);
                }
            }
            s.reset();
            s.state(start_state);

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
        });
    }


    // myopic
    CHECK_EQ(results->results.count("myopic"), 1);
    CHECK_EQ(results->results.at("myopic").size(), num_reps);
    for (uint32_t i = 0; i < num_reps; ++i) {
        pool->service().post([=]() {
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            MyopicAgent<EbolaState> a(net,
                    mod_agents->clone());
            a.seed(i);

            // sort outbreaks
            std::vector<uint32_t> outbreak_dates;
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0) {
                    outbreak_dates.push_back(EbolaData::outbreaks().at(i));
                }
            }
            std::sort(outbreak_dates.begin(), outbreak_dates.end());
            const uint32_t last_index(
                    std::min(std::max(1u,
                                    static_cast<uint32_t>(
                                            outbreak_dates.size()
                                            * starting_prop)),
                            static_cast<uint32_t>(outbreak_dates.size() - 1u)));
            const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

            EbolaState start_state(EbolaState(net->size()));
            for (uint32_t i = 0; i < net->size(); ++i) {
                if (EbolaData::outbreaks().at(i) >= 0
                        && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
                    start_state.inf_bits.set(i);
                }
            }
            s.reset();
            s.state(start_state);

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
        });
    }


    // // vfn max finite q
    // CHECK_EQ(results->results.count("vfn_finite_q"), 1);
    // CHECK_EQ(results->results.at("vfn_finite_q").size(), num_reps);
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     pool->service().post([=]() {
    //         System<EbolaState> s(net, mod_system->clone());
    //         s.seed(i);
    //         VfnMaxSimPerturbAgent<EbolaState> a(net,
    //                 std::shared_ptr<Features<EbolaState> >(
    //                         new FiniteQfnFeatures<EbolaState>(
    //                                 net, {mod_agents->clone()},
    //                                 std::shared_ptr<Features<EbolaState> >(
    //                                         new EbolaModelFeatures(
    //                                                 net, mod_agents->clone())),
    //                                 1, false)),
    //                 mod_agents->clone(),
    //                 2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    //         a.seed(i);

    //         // sort outbreaks
    //         std::vector<uint32_t> outbreak_dates;
    //         for (uint32_t i = 0; i < net->size(); ++i) {
    //             if (EbolaData::outbreaks().at(i) >= 0) {
    //                 outbreak_dates.push_back(EbolaData::outbreaks().at(i));
    //             }
    //         }
    //         std::sort(outbreak_dates.begin(), outbreak_dates.end());
    //         const uint32_t outbreaks_cutoff(
    //                 outbreak_dates.at(static_cast<uint32_t>(
    //                                 outbreak_dates.size() * 0.25)));

    //         EbolaState start_state(EbolaState(net->size()));
    //         for (uint32_t i = 0; i < net->size(); ++i) {
    //             if (EbolaData::outbreaks().at(i) >= 0
    //                     && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
    //                 start_state.inf_bits.set(i);
    //             }
    //         }
    //         s.reset();
    //         s.state(start_state);

    //         Outcome outcome;

    //         std::chrono::time_point<
    //             std::chrono::steady_clock> tick =
    //             std::chrono::steady_clock::now();

    //         outcome.value = runner(&s, &a, time_points, 1.0);

    //         std::chrono::time_point<
    //             std::chrono::steady_clock> tock =
    //             std::chrono::steady_clock::now();

    //         outcome.time = std::chrono::duration_cast<
    //             std::chrono::seconds>(tock - tick).count();

    //         outcome.history = s.history();
    //         outcome.history.emplace_back(s.state(),
    //                 boost::dynamic_bitset<>(net->size()));

    //         results->results.at("vfn_finite_q").at(i).set_value(
    //                 std::move(outcome));
    //     });
    // }
}


void queue_all_sims(
        njm::thread::Pool * const pool,
        AllResults<std::promise> * const results,
        const std::vector<std::shared_ptr<const Network> > & networks,
        const std::vector<std::pair<std::string,
        std::vector<ModelPair> > > & models,
        const uint32_t & num_reps,
        const uint32_t & time_points,
        const double & starting_prop) {

    CHECK_EQ(networks.size(), results->size());
    for (uint32_t i = 0; i < networks.size(); ++i) {
        CHECK_EQ(results->at(i).results.size(), models.size());

        for (uint32_t j = 0; j < models.size(); ++j) {
            queue_sim(pool, & results->at(i).results.at(j),
                    networks.at(i),
                    models.at(j).second.at(i).first,
                    models.at(j).second.at(i).second,
                    num_reps, time_points, starting_prop);
        }
    }
}


void process_results(
        AllResults<std::future> & all_results,
        const std::vector<std::string> & agent_names) {

    for (uint32_t i = 0; i < all_results.size(); ++i) {
        NetworkResults<std::future> & nr(all_results.at(i));
        const std::string network_kind(nr.network_kind);

        for (uint32_t j = 0; j < nr.results.size(); ++j) {
            ModelResults<std::future> & mr(nr.results.at(j));
            const std::string model_kind(mr.model_kind);

            std::cout << "results for network " << network_kind
                      << " and model pair " << model_kind << "\n";

            Results<std::future> & r(mr.results);
            std::vector<std::string>::const_iterator it;
            for (it = agent_names.begin(); it != agent_names.end(); ++it) {
                const std::string agent_kind(*it);

                OutcomeReps<std::future> & reps(r.at(agent_kind));

                accumulator_set<double, stats<tag::mean, tag::variance> >
                    values, times, final_inf;
                for (uint32_t k = 0; k < reps.size(); ++k) {
                    const Outcome & outcome (reps.at(k).get());
                    values(outcome.value);
                    times(outcome.time);

                    const uint32_t final_time(outcome.history.size() - 1);
                    const boost::dynamic_bitset<> final_inf_bits(
                            outcome.history.at(final_time).state.inf_bits);
                    final_inf(static_cast<double>(final_inf_bits.count())
                            / final_inf_bits.size());
                }
                CHECK_GT(reps.size(), 1);

                const double value_mean(mean(values));
                const double re_scale(static_cast<double>(reps.size())
                        / (reps.size() - 1));
                const double value_sse(std::sqrt(variance(values) * re_scale)
                        / std::sqrt(static_cast<double>(reps.size())));
                const double time_mean(mean(times));

                const double final_inf_mean(mean(final_inf));
                const double final_inf_sse(std::sqrt(variance(values)
                                * re_scale)
                        / std::sqrt(static_cast<double>(reps.size())));


                std::cout << std::setfill(' ') << std::setw(16) << std::right
                          << agent_kind << ": "
                          << std::setw(8) << std::right
                          << std::setprecision(3) << std::fixed
                          << value_mean << " ("
                          << std::setw(8) << std::right
                          << std::setprecision(4) << std::fixed
                          << value_sse << ")  ["
                          << std::setw(8) << std::setprecision(0)
                          << time_mean << "] -> "
                          << std::setw(8) << std::setprecision(3)
                          << final_inf_mean << " ("
                          << std::setw(8) << std::setprecision(4)
                          << final_inf_sse << ")"
                          << "\n";
            }

            std::cout << "====================================="
                      << "====================================="
                      << "\n";

        }
    }
}



std::vector<double> get_ebola_par(const double starting_prop) {
    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> net(Network::gen_network(init));

    std::vector<StateAndTrt<EbolaState> > obs_history;

    // init model
    const std::shared_ptr<EbolaStateGravityModel> mod(
            new EbolaStateGravityModel(net));

    const std::vector<int> & outbreaks(EbolaData::outbreaks());
    CHECK_EQ(outbreaks.size(), net->size());
    // preallocate history
    const int & max_outbreak(*std::max_element(
                    outbreaks.begin(), outbreaks.end()));
    for (uint32_t i = 0; i < (max_outbreak + 1); ++i) {
        obs_history.emplace_back(EbolaState(net->size()),
                boost::dynamic_bitset<>(net->size()));
    }

    // fill in history
    for (uint32_t i = 0; i < net->size(); ++i) {
        const int outbreak(outbreaks.at(i));
        for (uint32_t j = 0; j < obs_history.size(); ++j) {
            EbolaState & state(obs_history.at(j).state);
            CHECK(!state.inf_bits.test(i));
            if (j >= outbreak && outbreak >= 0 && outbreak <= 157) {
                state.inf_bits.flip(i);
            }
        }
    }

    std::vector<double> par(mod->par_size(), 0.0);
    mod->par(par);

    // mod->est_par(Transition<EbolaState>::from_sequence(obs_history));

    // par = mod->par();

    par = {-4.50446, 1.80575, 0.051293, 0.0, 0.0};

    std::cout << "estimated par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;

    // sort outbreaks
    std::vector<uint32_t> outbreak_dates;
    for (uint32_t i = 0; i < net->size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0) {
            outbreak_dates.push_back(EbolaData::outbreaks().at(i));
        }
    }
    std::sort(outbreak_dates.begin(), outbreak_dates.end());
    const uint32_t last_index(
            std::min(std::max(1u,
                            static_cast<uint32_t>(
                                    outbreak_dates.size() * starting_prop)),
                    static_cast<uint32_t>(outbreak_dates.size() - 1u)));
    const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

    EbolaState start_state(EbolaState(net->size()));
    for (uint32_t i = 0; i < net->size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0
                && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
            start_state.inf_bits.set(i);
        }
    }

    std::cout << "starting infection: "
              << start_state.inf_bits.count()
              << " / " << net->size()
              << " -> "
              << (start_state.inf_bits.count()
                      / static_cast<double>(net->size()))
              << std::endl;

    NoTrtAgent<EbolaState> agent_tune_inf(net);
    AllTrtAgent<EbolaState> agent_tune_trt(net);
    // RandomAgent<EbolaState> agent_tune_trt(net);
    // ProximalAgent<EbolaState> agent_tune_trt(net);
    // MyopicAgent<EbolaState> agent_tune_trt(net, mod);

    const uint32_t time_points(25);
    const uint32_t num_reps(48);

    // tune infection rate
    const double target_tune_inf(0.35);
    bool calibrated(false);
    bool was_above(false);
    double scale = 0.1;
    std::cout << "tuning infection rate" << std::endl;
    std::cout << "target infection: "
              << std::setw(8)
              << std::setfill('0')
              << std::setprecision(6)
              << std::fixed
              << target_tune_inf << std::endl;
    uint32_t iter(0);
    while(!calibrated) {
        mod->par(par);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_inf.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_tune_inf, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;

        std::cout << "\rcurrent: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf << std::flush;

        if (std::abs(avg_inf - target_tune_inf) < 0.01) {
            // done
            calibrated = true;
        } else if (avg_inf > target_tune_inf) {
            // reduce rate of spread
            par.at(0) *= 1.0 + scale;
            par.at(1) = std::log(std::exp(par.at(1)) * (1.0 + scale));
            if (!was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = true;
        } else {
            // increase rate of spread
            par.at(0) /= 1.0 + scale;
            par.at(1) = std::log(std::exp(par.at(1)) / (1.0 + scale));
            if (was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = false;
        }
    }
    std::cout << std::endl;

    // print current par
    std::cout << "par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;


    // tune treatment effect size
    par.at(3) = par.at(4) = -10.0;
    const double start_inf = start_state.inf_bits.count()
        / static_cast<double>(net->size());
    const double target_tune_trt(start_inf
            + 0.05 * (target_tune_inf - start_inf));
    calibrated = false;
    was_above = false;
    scale = 0.1;
    std::cout << "tuning treatment effect size" << std::endl;
    std::cout << "target infection: "
              << std::setw(8)
              << std::setfill('0')
              << std::setprecision(6)
              << std::fixed
              << target_tune_trt << std::endl;
    iter = 0;
    while(!calibrated) {
        mod->par(par);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_trt.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_tune_trt, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;

        std::cout << "\rcurrent: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf
                  << " ("
                  << std::setw(3)
                  << iter
                  << " -> "
                  << std::setw(6)
                  << std::setfill('0')
                  << std::setprecision(3)
                  << std::fixed
                  << par.at(3)
                  << ")" << std::flush;

        if (std::abs(avg_inf - target_tune_trt) < 0.01) {
            // done
            calibrated = true;
        } else if (avg_inf > target_tune_trt) {
            // increase treatment effect size
            par.at(3) *= 1.0 + scale;
            par.at(4) *= 1.0 + scale;
            if (!was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = true;
        } else {
            // decrease treatment effect size
            par.at(3) /= 1.0 + scale;
            par.at(4) /= 1.0 + scale;
            if (was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = false;
        }
        ++iter;
    }
    std::cout << std::endl;

    // print final par
    std::cout << "final par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;

    return par;
}



int main(int argc, char *argv[]) {
    // gflags::ParseCommandLineFlags(&argc, &argv, true);
    // google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);

    EbolaData::init();

    const double starting_prop(0.0);

    // setup networks
    std::vector<std::shared_ptr<const Network> > networks;
    { // random 1000
        NetworkInit init;
        init.set_type(NetworkInit_NetType_EBOLA);
        networks.push_back(Network::gen_network(init));
    }

    // double vector since model depends on network
    std::vector<std::pair<std::string,
                          std::vector<ModelPair> > > models;
    { // models

        // std::vector<double> grav_par{-5.246, std::log(155.8), 0.186,
        //                              -8.0, -8.0};

        // these are obtained from running fit_ebola
        // std::vector<double> grav_par{-2.999819, 1.399234, 0.051293,
        //                              -1.015256*2, -1.015256*2};
        // std::vector<double> grav_par{-3.104833, 1.433642, 0.051293,
        //                              -1.486436, -1.486436};
        std::vector<double> grav_par(get_ebola_par(starting_prop));


        { // Correct: Gravity,  Postulated: Gravity
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<EbolaState> >(
                                new EbolaStateGravityModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<EbolaState> >(
                                new EbolaStateGravityModel(
                                        networks.at(i))));
                mp.first->par(grav_par);
                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Gravity-Gravity",
                            models_add));
        }
    }

    const uint32_t num_reps = 50;
    const uint32_t time_points = 25;

    // set up results containers
    const std::vector<std::string> agent_names({
                "none", "random", "proximal", "myopic"
                // "vfn_finite_q"
                // "br_finite_q",
            });
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

    njm::thread::Pool pool(std::thread::hardware_concurrency());

    // queue sims
    queue_all_sims(&pool, &promise_results, networks,
            models, num_reps, time_points, starting_prop);

    // process results
    process_results(future_results, agent_names);

    return 0;
}
