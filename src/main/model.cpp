#include "model.hpp"
#include "utilities.hpp"
#include <glog/logging.h>
#include <iostream>
#include <cmath>

namespace stdmMf {

void Model::est_par(const boost::dynamic_bitset<> & inf_bits,
        std::vector<InfAndTrt> history) {
    // this has been copied, so okay to modify
    history.push_back(InfAndTrt(inf_bits,
                    boost::dynamic_bitset<>(inf_bits.size())));
    this->est_par(history);
}

void Model::est_par(const std::vector<InfAndTrt> & history) {
    CHECK_GT(history.size(), 1);

    // create fit object
    ModelFit mf(this, history);

    // set up gsl objects / containers
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer *s;

    gsl_vector * const x = gsl_vector_alloc(this->par_size());

    for(uint32_t pi = 0; pi < this->par_size(); ++pi){
        gsl_vector_set(x, pi, 0.0);
    }

    gsl_multimin_function_fdf my_func;
    my_func.n = this->par_size();
    my_func.f = ModelFit::obj_fn;
    my_func.df = ModelFit::obj_fn_grad;
    my_func.fdf = ModelFit::obj_fn_both;
    my_func.params = &mf;

    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T,this->par_size());

    gsl_multimin_fdfminimizer_set(s,&my_func,x,0.01,0.1);

    // optimization
    int iter = 0;
    int status;
    const int maxIter = 100;
    do{
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);

        if(status)
            break;

        status = gsl_multimin_test_gradient(s->gradient,1e-3);

    }while(status == GSL_CONTINUE && iter < maxIter);

    // check for error
    bool error_occurred = status != GSL_SUCCESS && status != GSL_CONTINUE
        && status != GSL_ENOPROG;

    CHECK(!error_occurred) << "Optimization did not succeed. "
                           << "Exited with code " << status;

    // assign estimated paramter values
    std::vector<double> mle;
    for(uint32_t pi = 0; pi < this->par_size(); ++pi){
        mle.push_back(gsl_vector_get(s->x, pi));
    }

    this->par(mle);

    // clean up
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}


double Model::ll(const std::vector<InfAndTrt> & history) const {
    const uint32_t history_size = history.size();
    CHECK_GE(history_size, 2);
    double ll_value = 0.0;
    for (uint32_t i = 0; i < (history_size - 1); ++i) {
        const InfAndTrt & curr_history = history.at(i);
        const boost::dynamic_bitset<> & curr_inf = curr_history.inf_bits;
        const boost::dynamic_bitset<> & curr_trt = curr_history.trt_bits;
        // infection probabilities
        const std::vector<double> probs = this->probs(curr_inf, curr_trt);

        const boost::dynamic_bitset<> & next_inf = history.at(i + 1).inf_bits;

        // get bits for changes in infection
        const boost::dynamic_bitset<> & change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const auto change_both_sets = both_sets(change_inf);
        const std::vector<uint32_t> changed = change_both_sets.first;
        const uint32_t num_changed = changed.size();
        const std::vector<uint32_t> unchanged = change_both_sets.second;
        const uint32_t num_unchanged = unchanged.size();

        for (uint32_t j = 0; j < num_changed; ++j) {
            const double p = probs.at(changed.at(j));
            ll_value += std::log(std::max(1e-14, p)); // for stability
        }
        for (uint32_t j = 0; j < num_unchanged; ++j) {
            const double p = 1.0 - probs.at(unchanged.at(j));
            ll_value += std::log(std::max(1e-14, p));
        }
    }
    return ll_value / (history_size - 1);
}


double Model::ll(const std::vector<Transition> & history) const {
    const uint32_t history_size = history.size();
    CHECK_GE(history_size, 1);
    double ll_value = 0.0;
    for (uint32_t i = 0; i < history_size; ++i) {
        const boost::dynamic_bitset<> & curr_inf = history.at(i).curr_inf_bits;
        const boost::dynamic_bitset<> & curr_trt = history.at(i).curr_trt_bits;
        // infection probabilities
        const std::vector<double> probs = this->probs(curr_inf, curr_trt);

        const boost::dynamic_bitset<> & next_inf = history.at(i).next_inf_bits;

        // get bits for changes in infection
        const boost::dynamic_bitset<> & change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const auto change_both_sets = both_sets(change_inf);
        const std::vector<uint32_t> changed = change_both_sets.first;
        const uint32_t num_changed = changed.size();
        const std::vector<uint32_t> unchanged = change_both_sets.second;
        const uint32_t num_unchanged = unchanged.size();

        for (uint32_t j = 0; j < num_changed; ++j) {
            const double p = probs.at(changed.at(j));
            ll_value += std::log(std::max(1e-14, p)); // for stability
        }
        for (uint32_t j = 0; j < num_unchanged; ++j) {
            const double p = 1.0 - probs.at(unchanged.at(j));
            ll_value += std::log(std::max(1e-14, p));
        }
    }
    return ll_value / history_size;
}


ModelFit::ModelFit(Model * const model, const std::vector<InfAndTrt> & history)
    : model_(model), history_(history) {
}


double ModelFit::obj_fn(const gsl_vector * x, void * params){
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    // return negative since GSL minimizes the function
    const double ll = mf->model_->ll(mf->history_);
    CHECK(std::isfinite(ll)) << "Likelihood value is not finite";
    return - ll;
}

void ModelFit::obj_fn_grad(const gsl_vector * x, void * params, gsl_vector * g){
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    const std::vector<double> ll_grad = mf->model_->ll_grad(mf->history_);
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        // assign the negative of the gradient value
        // GSL minimizes the function, need to adjust the gradient too
        CHECK(std::isfinite(ll_grad.at(pi)))
            << "Likelihood gradient value is not finite for parameter index "
            << pi;
        gsl_vector_set(g, pi, -ll_grad.at(pi));
    }
}

void ModelFit::obj_fn_both(const gsl_vector * x, void * params, double * f,
        gsl_vector * g){
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    const double ll_value = mf->model_->ll(mf->history_);
    const std::vector<double> ll_grad = mf->model_->ll_grad(mf->history_);

    // log ll
    CHECK(std::isfinite(ll_value));
    *f = -ll_value;

    // log ll grad
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        // assign the negative of the gradient value
        // GSL minimizes the function, need to adjust the gradient too
        CHECK(std::isfinite(ll_grad.at(pi)))
            << "Likelihood gradient value is not finite for parameter index "
            << pi;
        gsl_vector_set(g, pi, -ll_grad.at(pi));
    }
}



} // namespace stdmMf
