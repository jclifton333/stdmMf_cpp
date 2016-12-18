#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <vector>
#include <cstdint>

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
