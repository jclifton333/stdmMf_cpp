#include "system.hpp"
#include "noCovEdgeModel.hpp"
#include "noCovEdgeOrSoModel.hpp"
#include "noCovEdgeXorSoModel.hpp"
#include "noCovEdgeSepSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"

#include "networkRunSymFeatures.hpp"

#include "objFns.hpp"

#include "result.hpp"

#include "pool.hpp"

#include <thread>

#include <fstream>

using namespace stdmMf;


std::vector<std::pair<std::string, std::vector<double> > >
run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model> & mod_system,
        const std::shared_ptr<Model> & mod_agents,
        const uint32_t & num_reps) {

    // Pool pool(std::min(num_reps, std::thread::hardware_concurrency()));
    Pool pool(std::thread::hardware_concurrency());

    // none
    std::vector<std::shared_ptr<Result<double> > > none_val;
    std::vector<std::shared_ptr<Result<double> > > none_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        none_val.push_back(r_val);
        none_time.push_back(r_time);

        pool.service()->post([=](){
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    NoTrtAgent a(net->clone());

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }

    // random
    std::vector<std::shared_ptr<Result<double> > > random_val;
    std::vector<std::shared_ptr<Result<double> > > random_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        random_val.push_back(r_val);
        random_time.push_back(r_time);

        pool.service()->post([=](){
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    RandomAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // proximal
    std::vector<std::shared_ptr<Result<double> > > proximal_val;
    std::vector<std::shared_ptr<Result<double> > > proximal_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        proximal_val.push_back(r_val);
        proximal_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    ProximalAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // myopic
    std::vector<std::shared_ptr<Result<double> > > myopic_val;
    std::vector<std::shared_ptr<Result<double> > > myopic_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        myopic_val.push_back(r_val);
        myopic_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    MyopicAgent a(net->clone(), mod_agents->clone());
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // vfn max
    std::vector<std::shared_ptr<Result<double> > > vfn_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_val.push_back(r_val);
        vfn_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnMaxSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // br min
    std::vector<std::shared_ptr<Result<double> > > br_val;
    std::vector<std::shared_ptr<Result<double> > > br_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_val.push_back(r_val);
        br_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    BrMinSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }

    // vr max br min adapt step mult 1
    std::vector<std::shared_ptr<Result<double> > > adapt_1_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_1_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_1_val.push_back(r_val);
        adapt_1_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnBrAdaptSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
                            1);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }

    // vr max br min adapt step mult 2
    std::vector<std::shared_ptr<Result<double> > > adapt_2_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_2_val.push_back(r_val);
        adapt_2_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnBrAdaptSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
                            2);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }

    // vr max br min adapt step mult 5
    std::vector<std::shared_ptr<Result<double> > > adapt_5_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_5_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_5_val.push_back(r_val);
        adapt_5_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnBrAdaptSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
                            5);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // vr max br min adapt step mult 10
    std::vector<std::shared_ptr<Result<double> > > adapt_10_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_10_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_10_val.push_back(r_val);
        adapt_10_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnBrAdaptSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunSymFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
                            10);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    pool.join();

    std::vector<std::pair<std::string, std::vector<double> > > all_results;


    {
        const std::string agent_name = "none";
        const std::pair<double, double> none_stats = mean_and_var(
                result_to_vec(none_val));
        const std::vector<double> agent_res =
            {none_stats.first,
             std::sqrt(none_stats.second / num_reps),
             mean_and_var(result_to_vec(none_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "random";
        const std::pair<double, double> random_stats = mean_and_var(
                result_to_vec(random_val));
        const std::vector<double> agent_res =
            {random_stats.first,
             std::sqrt(random_stats.second / num_reps),
             mean_and_var(result_to_vec(random_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "proximal";
        const std::pair<double, double> proximal_stats = mean_and_var(
                result_to_vec(proximal_val));
        const std::vector<double> agent_res =
            {proximal_stats.first,
             std::sqrt(proximal_stats.second / num_reps),
             mean_and_var(result_to_vec(proximal_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "myopic";
        const std::pair<double, double> myopic_stats = mean_and_var(
                result_to_vec(myopic_val));
        const std::vector<double> agent_res =
            {myopic_stats.first,
             std::sqrt(myopic_stats.second / num_reps),
             mean_and_var(result_to_vec(myopic_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn";
        const std::pair<double, double> vfn_stats = mean_and_var(
                result_to_vec(vfn_val));
        const std::vector<double> agent_res =
            {vfn_stats.first,
             std::sqrt(vfn_stats.second / num_reps),
             mean_and_var(result_to_vec(vfn_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "br";
        const std::pair<double, double> br_stats = mean_and_var(
                result_to_vec(br_val));
        const std::vector<double> agent_res =
            {br_stats.first,
             std::sqrt(br_stats.second / num_reps),
             mean_and_var(result_to_vec(br_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_1";
        const std::pair<double, double> adapt_1_stats = mean_and_var(
                result_to_vec(adapt_1_val));
        const std::vector<double> agent_res =
            {adapt_1_stats.first,
             std::sqrt(adapt_1_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_1_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_2";
        const std::pair<double, double> adapt_2_stats = mean_and_var(
                result_to_vec(adapt_2_val));
        const std::vector<double> agent_res =
            {adapt_2_stats.first,
             std::sqrt(adapt_2_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_2_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_5";
        const std::pair<double, double> adapt_5_stats = mean_and_var(
                result_to_vec(adapt_5_val));
        const std::vector<double> agent_res =
            {adapt_5_stats.first,
             std::sqrt(adapt_5_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_5_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_10";
        const std::pair<double, double> adapt_10_stats = mean_and_var(
                result_to_vec(adapt_10_val));
        const std::vector<double> agent_res =
            {adapt_10_stats.first,
             std::sqrt(adapt_10_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_10_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    return all_results;
}


int main(int argc, char *argv[]) {
    // setup networks
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // network 2
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    { // network 3
        NetworkInit init;
        init.set_dim_x(25);
        init.set_dim_y(20);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // network 4
        NetworkInit init;
        init.set_size(500);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

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
    typedef std::pair<std::shared_ptr<Model>,
                      std::shared_ptr<Model> > ModelPair;
    std::vector<std::vector<ModelPair> > models;
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

        { // Correct: No So,  Postulated: No So
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: OrSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: XorSo,  Postulated: XorSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: SepSo,  Postulated: SepSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par_sep);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: SepSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: SepSo,  Postulated: XorSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: SepSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeSepSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par_sep);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: XorSo,  Postulated: OrSo
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: XorSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeXorSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }

        { // Correct: OrSo,  Postulated: No So
            std::vector<ModelPair > models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model>(new NoCovEdgeOrSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model>(new NoCovEdgeModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(models_add);
        }
    }

    std::ofstream ofs_raw;
    ofs_raw.open("run_results_raw.txt", std::ios_base::out);
    CHECK(ofs_raw.good()) << "could not open file";
    ofs_raw << "network,model,agent,mean,se,time" << std::endl;
    ofs_raw.close();

    std::ofstream ofs_read;
    ofs_read.open("run_results_read.txt", std::ios_base::out);
    CHECK(ofs_read.good()) << "could not open file";
    ofs_read.close();

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).at(i));

            std::vector<std::pair<std::string, std::vector<double> > >
                results = run(net, mp.first, mp.second, 50);

            ofs_raw.open("run_results_raw.txt", std::ios_base::app);
            CHECK(ofs_raw.good()) << "could not open file";

            ofs_read.open("run_results.txt", std::ios_base::app);
            CHECK(ofs_read.good()) << "could not open file";

            std::cout << "=====================================" << std::endl
                      << "results for network " << net->kind()
                      << " and model pair " << j << std::endl;

            ofs_read << "=====================================" << std::endl
                << "results for network " << net->kind()
                << " and model pair " << j << std::endl;

            for (uint32_t k = 0; k < results.size(); ++k) {
                ofs_raw << net->kind() << ","
                        << j << ","
                        << results.at(k).first << ","
                        << results.at(k).second.at(0) << ","
                        << results.at(k).second.at(1) << ","
                        << results.at(k).second.at(2) << std::endl;

                std::cout << results.at(k).first << ": "
                          << results.at(k).second.at(0) << " ("
                          << results.at(k).second.at(1) << ")  ["
                          << results.at(k).second.at(2) << "]"
                          << std::endl;

                ofs_read << results.at(k).first << ": "
                         << results.at(k).second.at(0) << " ("
                         << results.at(k).second.at(1) << ")  ["
                         << results.at(k).second.at(2) << "]"
                         << std::endl;
            }

            ofs_raw.close();
            ofs_read.close();
        }
    }

    return 0;
}
