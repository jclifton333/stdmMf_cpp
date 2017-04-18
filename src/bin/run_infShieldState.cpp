#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"
#include "brMinIterSimPerturbAgent.hpp"

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

using namespace stdmMf;

using njm::tools::mean_and_var;


std::string history_to_csv_entry(const uint32_t & num_nodes,
        const std::string & agent, const uint32_t & rep,
        const std::vector<StateAndTrt<InfShieldState> > & history) {
    const uint32_t num_points = history.size();
    std::stringstream ss;
    for (uint32_t i = 0; i < num_points; ++i) {

        const InfShieldState & state(history.at(i).state);
        const boost::dynamic_bitset<> & trt_bits(history.at(i).trt_bits);

        for (uint32_t j = 0; j < num_nodes; ++j) {
            const uint32_t j_inf(static_cast<uint32_t>(state.inf_bits.test(j)));
            const uint32_t j_trt(static_cast<uint32_t>(trt_bits.test(j)));

            ss << agent << ", " << rep << ", " << i << ", " << j << ", "
               << j_inf << ", " << state.shield.at(j) << ", " << j_trt << "\n";
        }
    }
    return ss.str();
}



std::vector<std::pair<std::string, std::vector<double> > >
run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & time_points,
        njm::data::Entry * const entry) {

    // Pool pool(std::min(num_reps, std::thread::hardware_concurrency()));
    njm::thread::Pool pool(std::thread::hardware_concurrency());


    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(&std::cout));

    uint32_t total_sims = 0;

    // none
    std::vector<std::future<double> > none_val;
    std::vector<std::future<double> > none_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;

        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        none_val.push_back(promise_val->get_future());
        none_time.push_back(promise_time->get_future());

        pool.service().post([=](){
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            NoTrtAgent<InfShieldState> a(net->clone());
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "none", i, history));
            *entry << add_to_entry;

            progress->update();
        });
    }

    // random
    std::vector<std::future<double> > random_val;
    std::vector<std::future<double> > random_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        random_val.push_back(promise_val->get_future());
        random_time.push_back(promise_time->get_future());

        pool.service().post([=](){
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            RandomAgent<InfShieldState> a(net->clone());
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "random", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // proximal
    std::vector<std::future<double> > proximal_val;
    std::vector<std::future<double> > proximal_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        proximal_val.push_back(promise_val->get_future());
        proximal_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            ProximalAgent<InfShieldState> a(net->clone());
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "proximal", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // myopic
    std::vector<std::future<double> > myopic_val;
    std::vector<std::future<double> > myopic_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        myopic_val.push_back(promise_val->get_future());
        myopic_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            MyopicAgent<InfShieldState> a(net->clone(),
                    mod_agents->clone());
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "myopic", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // vfn max length 1
    std::vector<std::future<double> > vfn_len_1_val;
    std::vector<std::future<double> > vfn_len_1_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        vfn_len_1_val.push_back(promise_val->get_future());
        vfn_len_1_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new NetworkRunSymFeatures<InfShieldState>(
                                    net->clone(), 1)),
                    mod_agents->clone(),
                    2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "vfn_len_1", i, history));
            *entry << add_to_entry;

            progress->update();
        });
    }


    // vfn max length 2
    std::vector<std::future<double> > vfn_len_2_val;
    std::vector<std::future<double> > vfn_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        vfn_len_2_val.push_back(promise_val->get_future());
        vfn_len_2_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new NetworkRunSymFeatures<InfShieldState>(
                                    net->clone(), 2)),
                    mod_agents->clone(),
                    2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "vfn_len_2", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // vfn max length 3
    std::vector<std::future<double> > vfn_len_3_val;
    std::vector<std::future<double> > vfn_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        vfn_len_3_val.push_back(promise_val->get_future());
        vfn_len_3_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new NetworkRunSymFeatures<InfShieldState>(
                                    net->clone(), 3)),
                    mod_agents->clone(),
                    2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "vfn_len_3", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // vfn max finite q
    std::vector<std::future<double> > vfn_finite_q_val;
    std::vector<std::future<double> > vfn_finite_q_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        vfn_finite_q_val.push_back(promise_val->get_future());
        vfn_finite_q_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new FiniteQfnFeatures<InfShieldState>(
                                    net->clone(), {mod_agents->clone()},
                                    std::shared_ptr<Features<InfShieldState> >(
                                            new NetworkRunSymFeatures
                                            <InfShieldState>(
                                                    net->clone(), 3)), 3)),
                    mod_agents->clone(),
                    2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "vfn_finite_q", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // br min length 1
    std::vector<std::future<double> > br_len_1_val;
    std::vector<std::future<double> > br_len_1_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        br_len_1_val.push_back(promise_val->get_future());
        br_len_1_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new NetworkRunSymFeatures<InfShieldState>(
                                    net->clone(), 1)),
                    mod_agents->clone(),
                    0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
                    true, true, false, 500, 0, 0, 0);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "br_len_1", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // br min length 2
    std::vector<std::future<double> > br_len_2_val;
    std::vector<std::future<double> > br_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        br_len_2_val.push_back(promise_val->get_future());
        br_len_2_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);
            BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new NetworkRunSymFeatures<InfShieldState>(
                                    net->clone(), 2)),
                    mod_agents->clone(),
                    0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
                    true, true, false, 500, 0, 0, 0);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "br_len_2", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    // // br min length 3
    // std::vector<std::future<double> > br_len_3_val;
    // std::vector<std::future<double> > br_len_3_time;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     ++total_sims;
    //     std::shared_ptr<std::promise<double> > promise_val(
    //             new std::promise<double>);
    //     std::shared_ptr<std::promise<double> > promise_time(
    //             new std::promise<double>);

    //     br_len_3_val.push_back(promise_val->get_future());
    //     br_len_3_time.push_back(promise_time->get_future());

    //     pool.service().post([=]() {
    //         System<InfShieldState> s(net->clone(), mod_system->clone());
    //         s.seed(i);
    //         BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
    //                 std::shared_ptr<Features<InfShieldState> >(
    //                         new NetworkRunSymFeatures<InfShieldState>(
    //                                 net->clone(), 3)),
    //                 mod_agents->clone(),
    //                 0.1, 0.35, 1.41, 1, 0.85, 0.00397,
    //                 true, true, false, 500, 0, 5, 5);
    //         a.seed(i);

    //         s.start();

    //         std::chrono::time_point<
    //             std::chrono::steady_clock> tick =
    //             std::chrono::steady_clock::now();

    //         promise_val->set_value(runner(&s, &a, time_points, 1.0));

    //         std::chrono::time_point<
    //             std::chrono::steady_clock> tock =
    //             std::chrono::steady_clock::now();

    //         promise_time->set_value(std::chrono::duration_cast<
    //                 std::chrono::seconds>(tock - tick).count());

    //         // write history to csv
    //         std::vector<StateAndTrt<InfShieldState> > history(s.history());
    //         history.emplace_back(s.state(),
    //                 boost::dynamic_bitset<>(net->size()));
    //         const std::string add_to_entry(history_to_csv_entry(net->size(),
    //                         "br_len_1", i, history));
    //         *entry << add_to_entry;


    //         progress->update();
    //     });
    // }


    // br min finite q
    std::vector<std::future<double> > br_finite_q_val;
    std::vector<std::future<double> > br_finite_q_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<std::promise<double> > promise_val(
                new std::promise<double>);
        std::shared_ptr<std::promise<double> > promise_time(
                new std::promise<double>);

        br_finite_q_val.push_back(promise_val->get_future());
        br_finite_q_time.push_back(promise_time->get_future());

        pool.service().post([=]() {
            System<InfShieldState> s(net->clone(), mod_system->clone());
            s.seed(i);

            std::shared_ptr<Model<InfShieldState> > modNoIm(
                    new InfShieldStateNoImNoSoModel(net->clone()));
            std::shared_ptr<Model<InfShieldState> > modPosIm(
                    new InfShieldStatePosImNoSoModel(net->clone()));

            BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                    std::shared_ptr<Features<InfShieldState> >(
                            new FiniteQfnFeatures<InfShieldState>(
                                    net->clone(), {modNoIm, modPosIm},
                                    std::shared_ptr<Features<InfShieldState> >(
                                            new NetworkRunSymFeatures<
                                            InfShieldState>(
                                                    net->clone(), 3)), 3)),
                    mod_agents->clone(),
                    0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
                    true, true, false, 500, 0, 0, 0);
            a.seed(i);

            s.start();

            std::chrono::time_point<
                std::chrono::steady_clock> tick =
                std::chrono::steady_clock::now();

            promise_val->set_value(runner(&s, &a, time_points, 1.0));

            std::chrono::time_point<
                std::chrono::steady_clock> tock =
                std::chrono::steady_clock::now();

            promise_time->set_value(std::chrono::duration_cast<
                    std::chrono::seconds>(tock - tick).count());

            // write history to csv
            std::vector<StateAndTrt<InfShieldState> > history(s.history());
            history.emplace_back(s.state(),
                    boost::dynamic_bitset<>(net->size()));
            const std::string add_to_entry(history_to_csv_entry(net->size(),
                            "br_finite_q", i, history));
            *entry << add_to_entry;


            progress->update();
        });
    }


    progress->total(total_sims);

    pool.join();

    progress->done();

    std::vector<std::pair<std::string, std::vector<double> > > all_results;


    {
        const std::string agent_name = "none";
        std::vector<double> val(num_reps);
        std::transform(none_val.begin(), none_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(none_time.begin(), none_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> none_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {none_stats.first,
             std::sqrt(none_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "random";
        std::vector<double> val(num_reps);
        std::transform(random_val.begin(), random_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(random_time.begin(), random_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> random_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {random_stats.first,
             std::sqrt(random_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "proximal";
        std::vector<double> val(num_reps);
        std::transform(proximal_val.begin(), proximal_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(proximal_time.begin(), proximal_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> proximal_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {proximal_stats.first,
             std::sqrt(proximal_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "myopic";
        std::vector<double> val(num_reps);
        std::transform(myopic_val.begin(), myopic_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(myopic_time.begin(), myopic_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> myopic_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {myopic_stats.first,
             std::sqrt(myopic_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_1";
        std::vector<double> val(num_reps);
        std::transform(vfn_len_1_val.begin(), vfn_len_1_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(vfn_len_1_time.begin(), vfn_len_1_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> vfn_len_1_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {vfn_len_1_stats.first,
             std::sqrt(vfn_len_1_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_2";
        std::vector<double> val(num_reps);
        std::transform(vfn_len_2_val.begin(), vfn_len_2_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(vfn_len_2_time.begin(), vfn_len_2_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> vfn_len_2_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {vfn_len_2_stats.first,
             std::sqrt(vfn_len_2_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_3";
        std::vector<double> val(num_reps);
        std::transform(vfn_len_3_val.begin(), vfn_len_3_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(vfn_len_3_time.begin(), vfn_len_3_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> vfn_len_3_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {vfn_len_3_stats.first,
             std::sqrt(vfn_len_3_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_finite_q";
        std::vector<double> val(num_reps);
        std::transform(vfn_finite_q_val.begin(), vfn_finite_q_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(vfn_finite_q_time.begin(), vfn_finite_q_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> vfn_finite_q_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {vfn_finite_q_stats.first,
             std::sqrt(vfn_finite_q_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "br_len_1";
        std::vector<double> val(num_reps);
        std::transform(br_len_1_val.begin(), br_len_1_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(br_len_1_time.begin(), br_len_1_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> br_len_1_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {br_len_1_stats.first,
             std::sqrt(br_len_1_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "br_len_2";
        std::vector<double> val(num_reps);
        std::transform(br_len_2_val.begin(), br_len_2_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(br_len_2_time.begin(), br_len_2_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> br_len_2_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {br_len_2_stats.first,
             std::sqrt(br_len_2_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    // {
    //     const std::string agent_name = "br_len_3";
    //     std::vector<double> val(num_reps);
    //     std::transform(br_len_3_val.begin(), br_len_3_val.end(),
    //             val.begin(), val.begin(),
    //             [](std::future<double> & a, const double & b) {
    //                 return a.get();
    //             });
    //     std::vector<double> time(num_reps);
    //     std::transform(br_len_3_time.begin(), br_len_3_time.end(),
    //             time.begin(), time.begin(),
    //             [](std::future<double> & a, const double & b) {
    //                 return a.get();
    //             });
    //     const std::pair<double, double> br_len_3_stats = mean_and_var(val);
    //     const std::vector<double> agent_res =
    //         {br_len_3_stats.first,
    //          std::sqrt(br_len_3_stats.second / num_reps),
    //          mean_and_var(time).first};
    //     all_results.push_back(std::pair<std::string, std::vector<double> >
    //             (agent_name, agent_res));
    // }

    {
        const std::string agent_name = "br_finite_q";
        std::vector<double> val(num_reps);
        std::transform(br_finite_q_val.begin(), br_finite_q_val.end(),
                val.begin(), val.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        std::vector<double> time(num_reps);
        std::transform(br_finite_q_time.begin(), br_finite_q_time.end(),
                time.begin(), time.begin(),
                [](std::future<double> & a, const double & b) {
                    return a.get();
                });
        const std::pair<double, double> br_finite_q_stats = mean_and_var(val);
        const std::vector<double> agent_res =
            {br_finite_q_stats.first,
             std::sqrt(br_finite_q_stats.second / num_reps),
             mean_and_var(time).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    return all_results;
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);

    // setup networks
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(4);
        init.set_dim_y(4);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    // { // network 2
    //     NetworkInit init;
    //     init.set_size(100);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 3
    //     NetworkInit init;
    //     init.set_dim_x(25);
    //     init.set_dim_y(20);
    //     init.set_wrap(false);
    //     init.set_type(NetworkInit_NetType_GRID);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 4
    //     NetworkInit init;
    //     init.set_size(500);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 5
    //     NetworkInit init;
    //     init.set_dim_x(25);
    //     init.set_dim_y(40);
    //     init.set_wrap(false);
    //     init.set_type(NetworkInit_NetType_GRID);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 6
    //     NetworkInit init;
    //     init.set_size(1000);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // double vector since model depends on network
    typedef std::pair<std::shared_ptr<Model<InfShieldState> >,
                      std::shared_ptr<Model<InfShieldState> > > ModelPair;
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
                    std::vector<ModelPair> >("Model_NoImNoSo_NoImNoSo",
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
                    std::vector<ModelPair> >("Model_PosImNoSo_PosImNoSo",
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
                    std::vector<ModelPair> >("Model_PosImNoSo_NoImNoSo",
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
                    std::vector<ModelPair> >("Model_NoImNoSo_PosImNoSo",
                            models_add));
        }
    }

    const uint32_t num_reps = 48;
    const uint32_t time_points = 100;

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");

    njm::data::Entry * e_read_all = tk.entry("all_read.txt");

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);

        njm::data::Entry * e_read_net = tk.entry(net->kind() + "_read.txt");

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).second.at(i));

            njm::data::Entry * e_raw = tk.entry(
                    net->kind() + "_" + models.at(j).first + "_raw.txt");
            njm::data::Entry * e_read = tk.entry(
                    net->kind() + "_" + models.at(j).first + "_read.txt");

            njm::data::Entry * e_history = tk.entry(
                    "history_" + net->kind() + "_" + models.at(j).first
                    + ".txt");
            *e_history << "agent, rep, time, node, inf, shield, trt\n";

            std::vector<std::pair<std::string, std::vector<double> > >
                results = run(net, mp.first, mp.second, num_reps, time_points,
                        e_history);

            std::cout << "=====================================" << std::endl
                      << "results for network " << net->kind()
                      << " and model pair " << models.at(j).first << std::endl;

            *e_read << "=====================================" << "\n"
                    << "results for network " << net->kind()
                    << " and model pair " << models.at(j).first << "\n";

            *e_read_net << "=====================================" << "\n"
                        << "results for network " << net->kind()
                        << " and model pair " << models.at(j).first << "\n";

            *e_read_all << "=====================================" << "\n"
                        << "results for network " << net->kind()
                        << " and model pair " << models.at(j).first << "\n";

            for (uint32_t k = 0; k < results.size(); ++k) {
                *e_raw << net->kind() << ","
                       << models.at(j).first << ","
                       << results.at(k).first << ","
                       << results.at(k).second.at(0) << ","
                       << results.at(k).second.at(1) << ","
                       << results.at(k).second.at(2) << "\n";

                std::cout << results.at(k).first << ": "
                          << results.at(k).second.at(0) << " ("
                          << results.at(k).second.at(1) << ")  ["
                          << results.at(k).second.at(2) << "]"
                          << std::endl;

                *e_read << results.at(k).first << ": "
                        << results.at(k).second.at(0) << " ("
                        << results.at(k).second.at(1) << ")  ["
                        << results.at(k).second.at(2) << "]"
                        << "\n";

                *e_read_net << results.at(k).first << ": "
                            << results.at(k).second.at(0) << " ("
                            << results.at(k).second.at(1) << ")  ["
                            << results.at(k).second.at(2) << "]"
                            << "\n";
                *e_read_all << results.at(k).first << ": "
                            << results.at(k).second.at(0) << " ("
                            << results.at(k).second.at(1) << ")  ["
                            << results.at(k).second.at(2) << "]"
                            << "\n";
            }
        }
    }

    tk.finished();

    return 0;
}
