#ifndef RESULT_HPP
#define RESULT_HPP

#include <glog/logging.h>

#include <algorithm>

namespace stdmMf {


template<class T>
class Result {
protected:
    T value_;
    bool has_value_;

public:
    Result() : has_value_(false) {};

    void set(const T & value) { this->value_ = value; this->has_value_ = true;};

    const T & get() const {CHECK(this->has_value_); return this->value_;};
};

template<class T>
std::vector<T> result_to_vec(const std::vector<Result<T> > & result) {
    std::vector<T> vec;
    std::for_each(result.begin(), result.end(),
            [&](const Result<T> & r){vec.push_back(r.get());});
    return vec;
}


} // namespace stdmMf


#endif // RESULT_HPP
