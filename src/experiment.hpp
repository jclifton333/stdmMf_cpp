#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <vector>
#include <cstdint>
#include <glog/logging.h>

namespace stdmMf {



class Experiment {
public:
    struct FactorLevel {
        enum Type {is_int, is_double};
        Type type;
        union {
            int int_val;
            double double_val;
        } val;

        friend std::ostream& operator<<(std::ostream& os,
                const FactorLevel & c) {
            if (c.type == is_int) {
                os << c.val.int_val;
            } else if (c.type == is_double) {
                os << c.val.double_val;
            } else {
                LOG(FATAL) << "unhandled FactorLevel::Type " << c.type;
            }
            return os;
        }
    };

    typedef std::vector<FactorLevel> Factor;

protected:
    std::vector<Factor> factors_;
    std::vector<uint32_t> n_levels_;
    uint32_t n_factors_;

    std::vector<uint32_t> levels_;
public:
    Experiment();

    void start();

    bool next();

    void add_factor(const std::vector<int> & factor);

    void add_factor(const std::vector<double> & factor);

    Factor get() const;
};


} // namespace stdmMf


#endif // EXPERIMENT_HPP
