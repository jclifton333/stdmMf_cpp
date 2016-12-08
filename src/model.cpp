#include "model.hpp"
#include <glog/logging.h>

namespace stdmMf {


double ModelFit::obj_fn(const gsl_vector * x, void * params){
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    int pi;
    for(pi = 0; pi < mf->model_.par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_.par(par);

    // return negative since GSL minimizes the function
    double ll = mf->model_.ll(mf->history_);
    CHECK(std::isfinite(ll)) << "Likelihood value is not finite";
    return - ll;
}

void ModelFit::obj_fn_grad(const gsl_vector * x, void * params, gsl_vector * g){
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    int pi;
    for(pi = 0; pi < mf->model_.par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_.par(par);

    std::vector<double> ll_grad = mf->model_.ll_grad(mf->history_);
    for(pi = 0; pi < mf->model_.par_size(); ++pi){
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
    // njm::timer.start("objFnBoth");
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    int pi;
    for(pi = 0; pi < mf->model_.par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_.par(par);

    double ll_value = mf->model_.ll(mf->history_);
    std::vector<double> ll_grad = mf->model_.ll_grad(mf->history_);

    // log ll
    CHECK(std::isfinite(ll_value));
    *f = -ll_value;

    // log ll grad
    for(pi = 0; pi < mf->model_.par_size(); ++pi){
        // assign the negative of the gradient value
        // GSL minimizes the function, need to adjust the gradient too
        CHECK(std::isfinite(ll_grad.at(pi)))
            << "Likelihood gradient value is not finite for parameter index "
            << pi;
        gsl_vector_set(g, pi, -ll_grad.at(pi));
    }
    // njm::timer.stop("objFnBoth");
}



} // namespace stdmMf
