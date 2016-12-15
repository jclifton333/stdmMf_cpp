#ifndef OPTIM_HPP
#define OPTIM_HPP

#include <functional>
#include <cstdint>
#include "random.hpp"
#include "utilities.hpp"

namespace stdmMf {


class Optim : public RngClass {
protected:
    const std::function<double(std::vector<double>, void *)> f_;

    std::vector<double> par_;

    const uint32_t par_size_;

    void * const data_;

    uint32_t completed_steps_;


public:
    enum ErrorCode {SUCCESS, CONTINUE, ERROR};

    Optim(const std::function<double(std::vector<double>, void*)> & f,
            const std::vector<double> & par,
            void * const data);

    std::vector<double> par() const;


    virtual Optim::ErrorCode step() = 0;
};

}


#endif // OPTIM_HPP