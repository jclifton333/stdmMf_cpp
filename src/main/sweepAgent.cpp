#include <glog/logging.h>
#include <set>
#include <limits>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <future>
#include <numeric>
#include "sweepAgent.hpp"

namespace stdmMf {

template <typename State>
SweepAgent<State>::SweepAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features<State> > & features,
        const std::vector<double> & coef,
        const std::function<double(const std::vector<double> &,
                const std::vector<double> &)> & eval_fn,
        const uint32_t & max_sweeps,
        const bool & do_sweep)
    : Agent<State>(network), features_(features), coef_(coef),
      eval_fn_(eval_fn), max_sweeps_(max_sweeps), do_sweep_(do_sweep),
      do_parallel_(false) {

    CHECK_EQ(this->coef_.size(), this->features_->num_features());
}


template <typename State>
SweepAgent<State>::SweepAgent(const SweepAgent & other)
    : Agent<State>(other),
    features_(other.features_->clone()), coef_(other.coef_),
      eval_fn_(other.eval_fn_), max_sweeps_(other.max_sweeps_),
      do_sweep_(other.do_sweep_), do_parallel_(other.do_parallel_),
    pool_(new njm::thread::Pool(*other.pool_)) {
}


template <typename State>
std::shared_ptr<Agent<State> > SweepAgent<State>::clone() const{
    return std::shared_ptr<Agent<State> >(new SweepAgent<State>(*this));
}


template <typename State>
void SweepAgent<State>::set_parallel(const bool & do_parallel,
        const uint32_t & num_threads) {
    this->do_parallel_ = do_parallel;
    if (do_parallel) {
        this->pool_ = std::shared_ptr<njm::thread::Pool>(
                new njm::thread::Pool(num_threads));
    } else {
        this->pool_.reset();
    }
}


template <typename State>
boost::dynamic_bitset<> SweepAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    return this->apply_trt(state);
}


template <typename State>
boost::dynamic_bitset<> SweepAgent<State>::apply_trt(
        const State & state) {
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);

    // sets of treated and not treated
    std::set<uint32_t> not_trt;
    std::set<uint32_t> has_trt;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        not_trt.insert(i);
    }

    std::vector<double> feat = this->features_->get_features(state,
            trt_bits);

    // initialize first treatment bits
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        this->set_new_treatment(trt_bits, not_trt, has_trt, state, feat);
    }

    double best_val = this->eval_fn_(this->coef_, feat);

    // sweep treatments
    if (this->do_sweep_) {
        for (uint32_t i = 0; i < this->max_sweeps_; ++i) {
            const bool changed = this->sweep_treatments(trt_bits, best_val,
                    not_trt, has_trt, state, feat);

            if (!changed)
                break;
        }
    }

    CHECK_EQ(trt_bits.count(), this->num_trt_);

    return trt_bits;
}


template <typename State>
void SweepAgent<State>::set_new_treatment(
        boost::dynamic_bitset<> & trt_bits,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state,
        std::vector<double> & feat) const {
    if (this->do_parallel_) {
        set_new_treatment_parallel(trt_bits, not_trt, has_trt, state, feat);
    } else {
        set_new_treatment_serial(trt_bits, not_trt, has_trt, state, feat);
    }
}


template <typename State>
void SweepAgent<State>::set_new_treatment_serial(
        boost::dynamic_bitset<> & trt_bits,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state,
        std::vector<double> & feat) const {

    std::set<uint32_t>::const_iterator it, end;
    end = not_trt.end();

    const boost::dynamic_bitset<> trt_bits_old(trt_bits);

    double best_val = std::numeric_limits<double>::lowest();
    std::vector<uint32_t> best_nodes;

    CHECK_GT(not_trt.size(), 0);
    for (it = not_trt.begin(); it != end; ++it) {
        CHECK(!trt_bits.test(*it)) << "bit is already set";
        trt_bits.set(*it); // set new bit

        // update features for treating *it
        this->features_->update_features(*it, state, trt_bits,
                state, trt_bits_old, feat);

        const double val = this->eval_fn_(this->coef_, feat);

        // update features for removing treatment on *it
        this->features_->update_features(*it, state, trt_bits_old,
                state, trt_bits, feat);

        trt_bits.reset(*it);

        if (val > best_val) {
            best_val = val;
            best_nodes.clear();
            best_nodes.push_back(*it);
        } else if (val == best_val) {
            best_nodes.push_back(*it);
        }
    }

    CHECK_GT(best_nodes.size(), 0)
        << "seed: " << this->rng()->seed() << std::endl
        << "coef:" << std::accumulate(this->coef_.begin(), this->coef_.end(),
                std::string(), [] (const std::string & a, const double & x) {
                    return a + " " + std::to_string(x);
                }) << std::endl
        << "feat: " << std::accumulate(feat.begin(), feat.end(),
                std::string(), [] (const std::string & a, const double & x) {
                    return a + " " + std::to_string(x);
                }) << std::endl;

    uint32_t best_node;
    if (best_nodes.size() == 1) {
        // unique best node
        best_node = best_nodes.at(0);
    } else {
        // multiple best nodes
        const uint32_t index = this->rng_->rint(0, best_nodes.size());
        best_node = best_nodes.at(index);
        trt_bits.set(best_node);
    }

    // set bit for bet node
    trt_bits.set(best_node);

    // update sets
    not_trt.erase(best_node);
    has_trt.insert(best_node);

    // update sets
    not_trt.erase(best_node);
    has_trt.insert(best_node);

    // update features for treating best_node
    this->features_->update_features(best_node, state, trt_bits,
            state, trt_bits_old, feat);
}


template <typename State>
void SweepAgent<State>::set_new_treatment_parallel(
        boost::dynamic_bitset<> & trt_bits,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state,
        std::vector<double> & feat) const {

    std::set<uint32_t>::const_iterator it, end;
    end = not_trt.end();

    const boost::dynamic_bitset<> trt_bits_old(trt_bits);

    double best_val = std::numeric_limits<double>::lowest();
    std::vector<uint32_t> best_nodes;

    std::atomic<uint32_t> num_left(not_trt.size());
    std::mutex finish_mtx;
    std::condition_variable cv;

    auto fn = [this, &feat, &state, &trt_bits, &trt_bits_old, &best_val,
            &best_nodes, &num_left, &finish_mtx, &cv]
        (const uint32_t & new_trt) -> std::pair<double, uint32_t> {

        // copy as part of each job
        auto trt_bits_cpy(trt_bits);
        auto feat_cpy(feat);

        trt_bits_cpy.set(new_trt); // set new bit

        // update features for treating *it
        this->features_->update_features_async(new_trt, state, trt_bits_cpy,
                state, trt_bits_old, feat_cpy);

        const double val = this->eval_fn_(this->coef_, feat_cpy);

        // notify main thread
        std::lock_guard<std::mutex> lk(finish_mtx);
        --num_left;
        cv.notify_one();

        return std::pair<double, uint32_t>(val, new_trt);
    };

    std::vector<std::future<std::pair<double, uint32_t> > > all_res;

    typedef std::packaged_task<std::pair<double, uint32_t>()> package_type;

    CHECK_GT(not_trt.size(), 0);
    for (it = not_trt.begin(); it != end; ++it) {
        CHECK(!trt_bits.test(*it)) << "bit is already set";

        std::shared_ptr<package_type> task(
                new package_type(std::bind(fn, *it)));

        all_res.push_back(task->get_future());

        this->pool_->service().post(std::bind(&package_type::operator(), task));
    }


    // wait for jobs to finish
    std::unique_lock<std::mutex> finish_lock(finish_mtx);
    cv.wait(finish_lock, [&num_left]{return num_left == 0;});

    for (uint32_t i = 0; i < all_res.size(); ++i) {
        const std::pair<double, uint32_t> & p = all_res.at(i).get();
        if (p.first > best_val) {
            best_val = p.first;
            best_nodes.clear();
            best_nodes.push_back(p.second);
        } else if (p.first == best_val) {
            best_nodes.push_back(p.second);
        }
    }


    CHECK_GT(best_nodes.size(), 0);
    uint32_t best_node;
    if (best_nodes.size() == 1) {
        // unique best node
        best_node = best_nodes.at(0);
    } else {
        // multiple best nodes
        const uint32_t index = this->rng_->rint(0, best_nodes.size());
        best_node = best_nodes.at(index);
        trt_bits.set(best_node);
    }

    // set bit for bet node
    trt_bits.set(best_node);

    // update sets
    not_trt.erase(best_node);
    has_trt.insert(best_node);

    // update sets
    not_trt.erase(best_node);
    has_trt.insert(best_node);

    // update features for treating best_node
    this->features_->update_features(best_node, state, trt_bits,
            state, trt_bits_old, feat);
}


template <typename State>
bool SweepAgent<State>::sweep_treatments(
        boost::dynamic_bitset<> & trt_bits,
        double & best_val,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const State & state,
        std::vector<double> & feat) const {


    std::set<uint32_t>::const_iterator has_it, not_it, has_end, not_end;
    has_end = has_trt.end();

    bool changed = false;

    std::set<uint32_t> new_not_trt;
    std::set<uint32_t> new_has_trt;

    // loop over all current treatments
    for (has_it = has_trt.begin(); has_it != has_end; ++has_it) {
        CHECK_EQ(has_trt.size(), this->num_trt());
        CHECK(trt_bits.test(*has_it)) << "bit is not set";

        boost::dynamic_bitset<> trt_bits_old = trt_bits;

        trt_bits.reset(*has_it); // reset

        // update features
        this->features_->update_features(*has_it, state, trt_bits,
                state, trt_bits_old, feat);

        // set up old to match trt_bits
        trt_bits_old.reset(*has_it);

        std::vector<uint32_t> better_nodes;
        better_nodes.push_back(*has_it);

        // see if any non-treated are better
        not_end = not_trt.end();

        bool can_tie_orig = true;

        for (not_it = not_trt.begin(); not_it != not_end; ++not_it) {
            CHECK(!trt_bits.test(*not_it)) << "bit is already set";

            trt_bits.set(*not_it);

            // update features for setting *not_it
            this->features_->update_features(*not_it, state, trt_bits,
                    state, trt_bits_old, feat);

            const double val = this->eval_fn_(this->coef_, feat);

            // update features for resetting *not_it
            this->features_->update_features(*not_it, state, trt_bits_old,
                    state, trt_bits, feat);

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
        uint32_t better_node;
        if (num_better == 1 && can_tie_orig) {
            // original node was best
            CHECK(!trt_bits.test(*has_it));
            trt_bits.set(*has_it);
            better_node = *has_it;
        } else if (num_better == 1) {
            // unique better node
            better_node = better_nodes.at(0);
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
            better_node = better_nodes.at(index);
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

        // update features for new treatment
        this->features_->update_features(better_node, state, trt_bits,
                state, trt_bits_old, feat);
    }

    // add new_trt to has_trt
    std::for_each(new_not_trt.begin(), new_not_trt.end(),
            [&has_trt] (const uint32_t & x) {
                has_trt.erase(x);
            });
    has_trt.insert(new_has_trt.begin(), new_has_trt.end());

    return changed;
}


template<typename State>
void SweepAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->features_->rng(rng);
}


template class SweepAgent<InfState>;
template class SweepAgent<InfShieldState>;



} // namespace stdmMf
