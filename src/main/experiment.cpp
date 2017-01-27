#include "experiment.hpp"

#include <glog/logging.h>
#include <algorithm>

#include <iostream>

namespace stdmMf {


Experiment::Experiment()
    : num_groups_(0) {
}

Experiment::FactorGroup * Experiment::add_group() {
    this->groups_.push_back(Experiment::FactorGroup());
    return &this->groups_.at(this->num_groups_++);
}


void Experiment::start() {
    this->group_num_ = 0;
    std::for_each(this->groups_.begin(), this->groups_.end(),
            [](FactorGroup & g) {g.start();});
}


bool Experiment::next() {
    if(!this->groups_.at(this->group_num_).next()) {
        ++this->group_num_;
    }

    return this->group_num_ < this->num_groups_;
}


Experiment::Factor Experiment::get() const {
    return this->groups_.at(this->group_num_).get();
}


Experiment::FactorGroup::FactorGroup()
    : n_factors_(0) {
}

void Experiment::FactorGroup::start() {
    this->levels_ = std::vector<uint32_t>(this->n_factors_, 0);
}

bool Experiment::FactorGroup::next() {
    bool has_next = true;
    for (uint32_t i = 0; i < this->n_factors_; ++i) {
        if (++this->levels_.at(i) < this->n_levels_.at(i)) {
            break;
        } else {
            if (i + 1 == this->n_factors_) {
                has_next = false;
            }
            this->levels_.at(i) = 0;
        }
    }
    return has_next;
}


Experiment::Factor Experiment::FactorGroup::get() const {
    Factor f;
    for (uint32_t i = 0; i < this->n_factors_; i++) {
        f.push_back(this->factors_.at(i).at(levels_.at(i)));
    }
    return f;
}


void Experiment::FactorGroup::add_factor(const std::vector<int> & vals) {
    const uint32_t size = vals.size();
    CHECK_GT(size, 0);
    Factor new_f;
    for (uint32_t i = 0; i < size; ++i) {
        FactorLevel fl;
        fl.type = FactorLevel::Type::is_int;
        fl.val.int_val = vals.at(i);

        new_f.push_back(fl);
    }
    this->factors_.push_back(new_f);
    ++this->n_factors_;
    this->n_levels_.push_back(size);
}


void Experiment::FactorGroup::add_factor(const std::vector<double> & vals) {
    const uint32_t size = vals.size();
    CHECK_GT(size, 0);
    Factor new_f;
    for (uint32_t i = 0; i < size; ++i) {
        FactorLevel fl;
        fl.type = FactorLevel::Type::is_double;
        fl.val.double_val = vals.at(i);

        new_f.push_back(fl);
    }
    this->factors_.push_back(new_f);
    ++this->n_factors_;
    this->n_levels_.push_back(size);
}


} // namespace stdmMf
