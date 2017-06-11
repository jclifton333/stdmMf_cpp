#include "model.hpp"
#include <njm_cpp/tools/bitManip.hpp>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <cmath>

namespace stdmMf {


template <typename State>
Model<State>::Model(const uint32_t & par_size,
        const std::shared_ptr<const Network> & network)
    : RngClass(), par_size_(par_size), network_(network),
      num_nodes_(network->size()) {
}


template <typename State>
Model<State>::Model(const Model<State> & other)
    : RngClass(other), par_size_(other.par_size_),
      network_(other.network_->clone()), num_nodes_(other.num_nodes_) {
}


template <typename State>
uint32_t Model<State>::par_size() const {
    return this->par_size_;
}


template <typename State>
void Model<State>::est_par(const std::vector<StateAndTrt<State> > & history,
        const State & state) {
    this->est_par(Transition<State>::from_sequence(history, state));
}


template <typename State>
void Model<State>::est_par(const std::vector<Transition<State> > & history) {
    CHECK_GE(history.size(), 1);

    // create fit object
    ModelFit<State> mf(this, history);

    // set up gsl objects / containers
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer *s;

    gsl_vector * const x = gsl_vector_alloc(this->par_size());

    const std::vector<double> current_par(this->par());
    for(uint32_t pi = 0; pi < this->par_size(); ++pi){
        gsl_vector_set(x, pi, current_par.at(pi));
    }

    gsl_multimin_function_fdf my_func;
    my_func.n = this->par_size();
    my_func.f = ModelFit<State>::obj_fn;
    my_func.df = ModelFit<State>::obj_fn_grad;
    my_func.fdf = ModelFit<State>::obj_fn_both;
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


template <typename State>
void Model<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}


template class Model<InfState>;
template class Model<InfShieldState>;
template class Model<EbolaState>;


template <typename State>
ModelFit<State>::ModelFit(Model<State> * const model,
        const std::vector<Transition<State> > & history)
    : model_(model), history_(history) {
}


template <typename State>
double ModelFit<State>::obj_fn(const gsl_vector * x, void * params){
    ModelFit<State> * mf = static_cast<ModelFit<State>*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    // return negative since GSL minimizes the function
    const double ll = mf->model_->ll(mf->history_);
    CHECK(std::isfinite(ll)) << "Likelihood value is not finite";


    // penalty
    const double penalty = std::accumulate(par.begin(), par.end(), 0.0,
            [](const double & a, const double & b) {
                if (b > 30.0) {
                    return a + (b - 30) * (b - 30);
                } else if (b < -30.0) {
                    return a + (b + 30) * (b + 30);
                } else {
                    return a;
                }
            });
    return - ll + penalty;
}

template <typename State>
void ModelFit<State>::obj_fn_grad(const gsl_vector * x, void * params,
        gsl_vector * g){
    ModelFit<State> * mf = static_cast<ModelFit<State>*>(params);
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
            << pi << " with value " << par.at(pi)
            << " [seed = " << mf->model_->seed() << "]";


        if (par.at(pi) > 30) {
            gsl_vector_set(g, pi, - ll_grad.at(pi)
                    + 2 * (par.at(pi) - 30));
        } else if (par.at(pi) < -30.0) {
            gsl_vector_set(g, pi, - ll_grad.at(pi)
                    + 2 * (par.at(pi) + 30));
        } else {
            gsl_vector_set(g, pi, - ll_grad.at(pi));
        }
    }
}

template <typename State>
void ModelFit<State>::obj_fn_both(const gsl_vector * x, void * params,
        double * f, gsl_vector * g){
    ModelFit<State> * mf = static_cast<ModelFit<State>*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    const double ll_value = mf->model_->ll(mf->history_);
    const std::vector<double> ll_grad = mf->model_->ll_grad(mf->history_);

    // log ll
    CHECK(std::isfinite(ll_value));

    // penalty
    const double penalty = std::accumulate(par.begin(), par.end(), 0.0,
            [](const double & a, const double & b) {
                if (b > 30.0) {
                    return a + (b - 30) * (b - 30);
                } else if (b < -30.0) {
                    return a + (b + 30) * (b + 30);
                } else {
                    return a;
                }
            });

    *f = - ll_value + penalty;

    // log ll grad
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        // assign the negative of the gradient value
        // GSL minimizes the function, need to adjust the gradient too
        CHECK(std::isfinite(ll_grad.at(pi)))
            << "Likelihood gradient value is not finite for parameter index "
            << pi;
        gsl_vector_set(g, pi, -ll_grad.at(pi));

        if (par.at(pi) > 30) {
            gsl_vector_set(g, pi, - ll_grad.at(pi)
                    + 2 * (par.at(pi) - 30));
        } else if (par.at(pi) < -30.0) {
            gsl_vector_set(g, pi, - ll_grad.at(pi)
                    + 2 * (par.at(pi) + 30));
        } else {
            gsl_vector_set(g, pi, - ll_grad.at(pi));
        }
    }
}


template class ModelFit<InfState>;
template class ModelFit<InfShieldState>;
template class ModelFit<EbolaState>;


} // namespace stdmMf
