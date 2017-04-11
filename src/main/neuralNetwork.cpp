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
    : num_nodes_(num_nodes), batch_size_(batch_size) {

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(model_file, &net_param);
    // check input layer for training
    CHECK_EQ(net_param.name(), "q_learning");
    CHECK_EQ(net_param.layer(0).name(), "input_data");
    CHECK_EQ(net_param.layer(0).phase(), "TRAIN");
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
    CHECK_EQ(net_param.name(), "q_learning");
    CHECK_EQ(net_param.layer(1).name(), "input_data");
    CHECK_EQ(net_param.layer(1).phase(), "TEST");
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
void NeuralNetwork<State>::fit() {
    this->solver_->Solve();
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

    CHECK_EQ(this->state_trt_train_data_.size(),
            this->num_input_per_node_ * this->num_nodes_);

    // setup eval output data (this is dummy data)
    this->outcome_eval_data_.clear();
    this->outcome_eval_data_.push_back(0.0);

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->eval_net_->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_train_data_.data(),
            this->outcome_eval_data_.data(), 1);

    this->eval_net_->Forward();
    return this->eval_net_->blob_by_name("fc3")->cpu_data()[0];
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

    CHECK_EQ(this->state_trt_train_data_.size(),
            this->num_input_per_node_ * this->num_nodes_);

    // setup eval output data (this is dummy data)
    this->outcome_eval_data_.clear();
    this->outcome_eval_data_.push_back(0.0);

    caffe::MemoryDataLayer<double> * input_data_layer(
            static_cast<caffe::MemoryDataLayer<double> *>(
                    this->eval_net_->layer_by_name("input_data").get()));
    input_data_layer->Reset(this->state_trt_train_data_.data(),
            this->outcome_eval_data_.data(), 1);

    this->eval_net_->Forward();
    return this->eval_net_->blob_by_name("fc3")->cpu_data()[0];
}





} // namespace stdmMf
