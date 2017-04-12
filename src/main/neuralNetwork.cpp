#include "neuralNetwork.hpp"

namespace stdmMf {

template <>
const uint32_t NeuralNetwork<InfState>::num_input_per_node_ = 2;


template <>
const uint32_t NeuralNetwork<InfShieldState>::num_input_per_node_ = 3;


template <typename State>
NeuralNetwork<State>::NeuralNetwork(const std::string & model_file,
        const std::string & solver_file,
        const uint32_t & batch_size,
        const uint32_t & num_nodes)
    : num_nodes_(num_nodes), batch_size_(batch_size), max_sweeps_(2) {

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(model_file, &net_param);
    // check input layer for training
    CHECK_EQ(net_param.name(), "q_function");
    CHECK_EQ(net_param.layer(0).name(), "input_data");
    CHECK_EQ(net_param.layer(0).include_size(), 1);
    CHECK_EQ(net_param.layer(0).include(0).phase(), caffe::TRAIN);
    CHECK_EQ(net_param.layer(0).type(), "MemoryData");
    CHECK(net_param.layer(0).has_memory_data_param());
    CHECK_EQ(net_param.layer(0).memory_data_param().channels(), 1);

    // set dimensions for input layer for training
    net_param.mutable_layer(0)->mutable_memory_data_param()->set_batch_size(
            this->batch_size_);
    net_param.mutable_layer(0)->mutable_memory_data_param()->set_height(
            this->num_nodes_);
    net_param.mutable_layer(0)->mutable_memory_data_param()->set_width(
            this->num_input_per_node_);

    // check input layer for evaluating
    CHECK_EQ(net_param.name(), "q_function");
    CHECK_EQ(net_param.layer(1).name(), "input_data");
    CHECK_EQ(net_param.layer(1).include_size(), 1);
    CHECK_EQ(net_param.layer(1).include(0).phase(), caffe::TEST);
    CHECK_EQ(net_param.layer(1).type(), "MemoryData");
    CHECK(net_param.layer(1).has_memory_data_param());
    CHECK_EQ(net_param.layer(1).memory_data_param().channels(), 1);

    // set dimensions for input layer for evaluating
    net_param.mutable_layer(1)->mutable_memory_data_param()->set_batch_size(1);
    net_param.mutable_layer(1)->mutable_memory_data_param()->set_height(
            this->num_nodes_);
    net_param.mutable_layer(1)->mutable_memory_data_param()->set_width(
            this->num_input_per_node_);

    // setup solver
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
    solver_param.mutable_net_param()->CopyFrom(net_param);
    this->solver_.reset(
            caffe::SolverRegistry<double>::CreateSolver(solver_param));
    CHECK_EQ(this->solver_->net()->phase(), caffe::TRAIN);

    // setup eval network
    caffe::NetParameter net_param_eval;
    net_param_eval.CopyFrom(net_param);
    this->eval_net_.reset(
            new caffe::Net<double>(net_param_eval));
    CHECK_EQ(this->eval_net_->phase(), caffe::TEST);

    // share layers from training model with evaluating model
    this->eval_net_->ShareTrainedLayersWith(this->solver_->net().get());
}


template <typename State>
NeuralNetwork<State>::NeuralNetwork(const NeuralNetwork & other)
    : batch_size_(other.batch_size_), num_nodes_(other.num_nodes_),
      state_trt_train_data_(other.state_trt_train_data_),
      outcome_train_data_(other.outcome_train_data_),
      state_trt_eval_data_(other.state_trt_eval_data_),
      outcome_eval_data_(other.outcome_eval_data_),
      max_sweeps_(other.max_sweeps_) {

    // get solver proto from other's solver
    caffe::SolverParameter solver_param(other.solver_->param());
    // copy train proto from other's solver network (this will copy blobs)
    other.solver_->net()->ToProto(solver_param.mutable_net_param());
    // craete solver
    this->solver_.reset(
            caffe::SolverRegistry<double>::CreateSolver(solver_param));

    // copy eval proto from other's eval network (this will copy blobs)
    caffe::NetParameter net_param_eval;
    other.eval_net_->ToProto(&net_param_eval);
    // create eval net
    this->eval_net_.reset(
            new caffe::Net<double>(net_param_eval));

    // share parameters between train and eval network
    this->eval_net_->ShareTrainedLayersWith(this->solver_->net().get());
}



template <typename State>
void NeuralNetwork<State>::fit() {
    this->solver_->Solve();
}




template <typename State>
std::pair<boost::dynamic_bitset<>, double> NeuralNetwork<State>::sweep_max(
        const State & state, const uint32_t & num_trt) {
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);

    // sets of treated and not treated
    std::set<uint32_t> not_trt;
    std::set<uint32_t> has_trt;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        not_trt.insert(i);
    }

    // initialize first treatment bits
    for (uint32_t i = 0; i < num_trt; ++i) {
        this->set_new_treatment(trt_bits, not_trt, has_trt, state);
    }

    double best_val = this->eval(StateAndTrt<State>(state, trt_bits));

    // sweep treatments
    if (this->max_sweeps_ > 0) {
        for (uint32_t i = 0; i < this->max_sweeps_; ++i) {
            const bool changed = this->sweep_treatments(trt_bits, best_val,
                    num_trt, not_trt, has_trt, state);

            if (!changed)
                break;
        }
    } else {
        bool changed = true;
        while (changed) {
            changed = this->sweep_treatments(trt_bits, best_val,
                    num_trt, not_trt, has_trt, state);
        }
    }

    CHECK_EQ(trt_bits.count(), num_trt);

    return std::make_pair(trt_bits, best_val);
}


template <typename State>
void NeuralNetwork<State>::set_new_treatment(
        boost::dynamic_bitset<> & trt_bits,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state) {

    std::set<uint32_t>::const_iterator it, end;
    end = not_trt.end();

    double best_val = std::numeric_limits<double>::lowest();
    std::vector<uint32_t> best_nodes;
    for (it = not_trt.begin(); it != end; ++it) {
        CHECK(!trt_bits.test(*it)) << "bit is already set";
        trt_bits.set(*it); // set new bit

        const double val = this->eval(StateAndTrt<State>(state, trt_bits));

        trt_bits.reset(*it); // reset new bit

        if (val > best_val) {
            best_val = val;
            best_nodes.clear();
            best_nodes.push_back(*it);
        } else if (val == best_val) {
            best_nodes.push_back(*it);
        }
    }

    CHECK_GT(best_nodes.size(), 0);
    if (best_nodes.size() == 1) {
        // unique best node
        const uint32_t best_node = best_nodes.at(0);
        trt_bits.set(best_node);
        // update sets
        not_trt.erase(best_node);
        has_trt.insert(best_node);
    } else {
        // multiple best nodes
        const uint32_t index = this->rng_->rint(0, best_nodes.size());
        const uint32_t best_node = best_nodes.at(index);
        trt_bits.set(best_node);
        // update sets
        not_trt.erase(best_node);
        has_trt.insert(best_node);
    }
}


template <typename State>
bool NeuralNetwork<State>::sweep_treatments(
        boost::dynamic_bitset<> & trt_bits,
        double & best_val,
        const uint32_t & num_trt,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state) {

    std::set<uint32_t>::const_iterator has_it, not_it, has_end, not_end;
    has_end = has_trt.end();

    bool changed = false;

    std::set<uint32_t> new_not_trt;
    std::set<uint32_t> new_has_trt;

    // loop over all current treatments
    for (has_it = has_trt.begin(); has_it != has_end; ++has_it) {
        CHECK_EQ(has_trt.size(), num_trt);
        CHECK(trt_bits.test(*has_it)) << "bit is not set";

        trt_bits.reset(*has_it); // reset

        std::vector<uint32_t> better_nodes;
        better_nodes.push_back(*has_it);

        // see if any non-treated are better
        not_end = not_trt.end();

        bool can_tie_orig = true;

        for (not_it = not_trt.begin(); not_it != not_end; ++not_it) {
            CHECK(!trt_bits.test(*not_it)) << "bit is already set";

            trt_bits.set(*not_it);

            const double val = this->eval(StateAndTrt<State>(state, trt_bits));

            trt_bits.reset(*not_it);

            if (val > best_val) {
                can_tie_orig = false;
                best_val = val;
                better_nodes.clear();
                better_nodes.push_back(*not_it);
            } else if (val == best_val) {
                better_nodes.push_back(*not_it);
            }
        }

        const uint32_t num_better = better_nodes.size();
        if (num_better == 1 && can_tie_orig) {
            // original node was best
            CHECK(!trt_bits.test(*has_it));
            trt_bits.set(*has_it);
        } else if (num_better == 1) {
            // unique better node
            const uint32_t better_node = better_nodes.at(0);
            CHECK(!trt_bits.test(better_node));
            trt_bits.set(better_node);
            not_trt.erase(better_node);
            changed = !can_tie_orig; // only has changed if value improved
            // add *has_it to set of not_treated
            not_trt.insert(*has_it);
            // records for updating has_trt;
            new_has_trt.insert(better_node);
            new_not_trt.insert(*has_it);
        } else {
            // multiple better nodes
            const uint32_t index = this->rng_->rint(0, num_better);
            const uint32_t better_node = better_nodes.at(index);
            CHECK(!trt_bits.test(better_node));
            trt_bits.set(better_node);
            if (better_node != *has_it) { // if it is the original
                not_trt.erase(better_node);
                changed = !can_tie_orig; // only has changed if value improved
                // add *has_it to set of not_treated
                not_trt.insert(*has_it);
                // records for updating has_trt;
                new_has_trt.insert(better_node);
                new_not_trt.insert(*has_it);
            }
        }
    }

    // add new_trt to has_trt
    std::for_each(new_not_trt.begin(), new_not_trt.end(),
            [&has_trt] (const uint32_t & x) {
                has_trt.erase(x);
            });
    has_trt.insert(new_has_trt.begin(), new_has_trt.end());

    return changed;
}



template <>
void NeuralNetwork<InfState>::train_data(
        const std::vector<StateAndTrt<InfState> > & state_trt,
        const std::vector<double> & outcome) {
    const uint32_t num_obs(state_trt.size());
    CHECK_EQ(num_obs % this->batch_size_, 0)
        << "number of observations " << num_obs << " is not a multiple of "
        << this->batch_size_;
    CHECK_EQ(num_obs, outcome.size());


    // setup training input data
    this->state_trt_train_data_.clear();
    this->state_trt_train_data_.reserve(
            this->num_input_per_node_ * this->num_nodes_ * num_obs);
    for (uint32_t i = 0; i < num_obs; ++i) {
        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            this->state_trt_train_data_.push_back(
                    state_trt.at(i).state.inf_bits.test(j) ? 1.0 : 0.0);
            this->state_trt_train_data_.push_back(
                    state_trt.at(i).trt_bits.test(j) ? 1.0 : 0.0);
        }
    }

    CHECK_EQ(this->state_trt_train_data_.size(),
            this->num_input_per_node_ * this->num_nodes_ * num_obs);

    // setup training outcome data
    this->outcome_train_data_ = outcome;

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->solver_->net()->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_train_data_.data(),
            this->outcome_train_data_.data(), num_obs);
}


template <>
void NeuralNetwork<InfShieldState>::train_data(
        const std::vector<StateAndTrt<InfShieldState> > & state_trt,
        const std::vector<double> & outcome) {
    const uint32_t num_obs(state_trt.size());
    CHECK_EQ(num_obs % this->batch_size_, 0)
        << "number of observations " << num_obs << " is not a multiple of "
        << this->batch_size_;
    CHECK_EQ(num_obs, outcome.size());


    // setup training input data
    this->state_trt_train_data_.clear();
    this->state_trt_train_data_.reserve(
            this->num_input_per_node_ * this->num_nodes_ * num_obs);
    for (uint32_t i = 0; i < num_obs; ++i) {
        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            this->state_trt_train_data_.push_back(
                    state_trt.at(i).state.inf_bits.test(j) ? 1.0 : 0.0);
            this->state_trt_train_data_.push_back(
                    state_trt.at(i).state.shield.at(j) ? 1.0 : 0.0);
            this->state_trt_train_data_.push_back(
                    state_trt.at(i).trt_bits.test(j) ? 1.0 : 0.0);
        }
    }

    CHECK_EQ(this->state_trt_train_data_.size(),
            this->num_input_per_node_ * this->num_nodes_ * num_obs);

    // setup training outcome data
    this->outcome_train_data_ = outcome;

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->solver_->net()->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_train_data_.data(),
            this->outcome_train_data_.data(), num_obs);
}


template <>
double NeuralNetwork<InfState>::eval(
        const StateAndTrt<InfState> & state_trt) {
    // setup eval input data
    this->state_trt_eval_data_.clear();
    this->state_trt_eval_data_.reserve(
            this->num_nodes_ * this->num_input_per_node_);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        this->state_trt_eval_data_.push_back(
                state_trt.state.inf_bits.test(i) ? 1.0 : 0.0);
        this->state_trt_eval_data_.push_back(
                state_trt.trt_bits.test(i) ? 1.0 : 0.0);
    }

    CHECK_EQ(this->state_trt_eval_data_.size(),
            this->num_input_per_node_ * this->num_nodes_);

    // setup eval output data (this is dummy data)
    this->outcome_eval_data_.clear();
    this->outcome_eval_data_.push_back(0.0);

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->eval_net_->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_eval_data_.data(),
            this->outcome_eval_data_.data(), 1);

    this->eval_net_->Forward();
    const double outcome(this->eval_net_->blob_by_name("fc3")->cpu_data()[0]);

    CHECK(std::isfinite(outcome));

    return outcome;
}


template <>
double NeuralNetwork<InfShieldState>::eval(
        const StateAndTrt<InfShieldState> & state_trt) {
    // setup eval input data
    this->state_trt_eval_data_.clear();
    this->state_trt_eval_data_.reserve(
            this->num_nodes_ * this->num_input_per_node_);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        this->state_trt_eval_data_.push_back(
                state_trt.state.inf_bits.test(i) ? 1.0 : 0.0);
        this->state_trt_eval_data_.push_back(
                state_trt.state.shield.at(i) ? 1.0 : 0.0);
        this->state_trt_eval_data_.push_back(
                state_trt.trt_bits.test(i) ? 1.0 : 0.0);
    }

    CHECK_EQ(this->state_trt_eval_data_.size(),
            this->num_input_per_node_ * this->num_nodes_);

    // setup eval output data (this is dummy data)
    this->outcome_eval_data_.clear();
    this->outcome_eval_data_.push_back(0.0);

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->eval_net_->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_eval_data_.data(),
            this->outcome_eval_data_.data(), 1);

    this->eval_net_->Forward();
    const double outcome(this->eval_net_->blob_by_name("fc3")->cpu_data()[0]);

    CHECK(std::isfinite(outcome));

    return outcome;
}

template <typename State>
void NeuralNetwork<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
}


template class NeuralNetwork<InfState>;
template class NeuralNetwork<InfShieldState>;




} // namespace stdmMf
