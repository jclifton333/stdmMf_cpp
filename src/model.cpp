#include "model.hpp"
#include <glog/logging.h>
#include <iostream>

namespace stdmMf {



void Model::est_par(const std::vector<BitsetPair> & history) {
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

    gsl_multimin_fdfminimizer_set(s,&my_func,x,0.0001,0.1);

    // optimization
    int iter = 0;
    int status;
    const int maxIter = 100;
    do{
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);

        if(status)
            break;

        status = gsl_multimin_test_gradient(s->gradient,1e-4);

        for(uint32_t pi = 0; pi < this->par_size(); ++pi){
            std::cout << gsl_vector_get(s->x, pi) << " ";
        }
        std::cout << std::endl;

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

ModelFit::ModelFit(Model * const model, const std::vector<BitsetPair> & history)
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
    double ll = mf->model_->ll(mf->history_);
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

    std::vector<double> ll_grad = mf->model_->ll_grad(mf->history_);
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
    // njm::timer.start("objFnBoth");
    ModelFit * mf = static_cast<ModelFit*>(params);
    std::vector<double> par;
    for(uint32_t pi = 0; pi < mf->model_->par_size(); ++pi){
        par.push_back(gsl_vector_get(x, pi));
    }

    mf->model_->par(par);

    double ll_value = mf->model_->ll(mf->history_);
    std::vector<double> ll_grad = mf->model_->ll_grad(mf->history_);

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
    // njm::timer.stop("objFnBoth");
}



} // namespace stdmMf
