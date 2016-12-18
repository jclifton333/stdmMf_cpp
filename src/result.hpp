#ifndef RESULT_HPP
#define RESULT_HPP

#include <glog/logging.h>

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


} // namespace stdmMf


#endif // RESULT_HPP
