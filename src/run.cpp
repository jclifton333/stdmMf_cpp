#include "system.hpp"
#include "noCovEdgeModel.hpp"
#include "noCovEdgeMaxSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"

#include "networkRunFeatures.hpp"

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

    Pool pool(std::min(num_reps, std::thread::hardware_concurrency()));

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
                                    new NetworkRunFeatures(net->clone(), 3)),
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
                                    new NetworkRunFeatures(net->clone(), 3)),
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

    // vr max br min adapt
    std::vector<std::shared_ptr<Result<double> > > adapt_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_val.push_back(r_val);
        adapt_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod_system->clone());
                    s.set_seed(i);
                    VfnBrAdaptSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunFeatures(net->clone(), 3)),
                            mod_agents->clone(),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
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
        const std::string agent_name = "adapt";
        const std::pair<double, double> adapt_stats = mean_and_var(
                result_to_vec(adapt_val));
        const std::vector<double> agent_res =
            {adapt_stats.first,
             std::sqrt(adapt_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    return all_results;
}


int main(int argc, char *argv[]) {
    std::vector<std::shared_ptr<Network> > networks;
    {
        NetworkInit init;
        init.set_dim_x(25);
        init.set_dim_y(40);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    {
        NetworkInit init;
        init.set_size(1000);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

    std::vector<std::vector<double> > pars;
    {
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

        pars.push_back(par);
    }

    std::ofstream ofs;
    ofs.open("run_results.txt", std::ios_base::out);
    ofs << "network,model,mean,agent,mean,se,time" << std::endl;
    ofs.close();

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);

        for (uint32_t j = 0; j < pars.size(); ++j) {
            const std::shared_ptr<Model> mod(new NoCovEdgeMaxSoModel(net));
            mod->par(pars.at(j));

            std::vector<std::pair<std::string, std::vector<double> > >
                results = run(net, mod, mod, 50);

            ofs.open("run_results.txt", std::ios_base::app);
            if (!ofs.good()) {
                LOG(FATAL) << "could not open file";
            }

            std::cout << "=====================================" << std::endl
                      << "results for network " << net->kind()
                      << " and correct model " << j << std::endl;

            for (uint32_t k = 0; k < results.size(); ++k) {
                ofs << net->kind() << ","
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
            }

            ofs.close();


            const std::shared_ptr<Model> mod_agents(new NoCovEdgeModel(net));
            mod->par(pars.at(j));

            results = run(net, mod, mod_agents, 50);

            ofs.open("run_results.txt", std::ios_base::app);
            if (!ofs.good()) {
                LOG(FATAL) << "could not open file";
            }

            std::cout << "=====================================" << std::endl
                      << "results for network " << net->kind()
                      << " and misspecified model " << j << std::endl;

            for (uint32_t k = 0; k < results.size(); ++k) {
                ofs << net->kind() << ","
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
            }

            ofs.close();
}
    }

    return 0;
}
