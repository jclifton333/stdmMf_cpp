#ifndef EBOLA_DATA_HPP
#define EBOLA_DATA_HPP


#include <vector>
#include <cstdint>
#include <string>

#include <glog/logging.h>

namespace stdmMf {


class EbolaData {
protected:
    static std::vector<std::string> country_;
    static std::vector<std::string> county_;
    static std::vector<std::string> loc_;
    static std::vector<uint32_t> region_;
    static std::vector<int> outbreaks_;
    static std::vector<double> population_;
    static std::vector<double> x_;
    static std::vector<double> y_;

    static bool init_;

    EbolaData();

public:
    inline static const std::vector<std::string> & country() {
        CHECK(EbolaData::init_);
        return EbolaData::country_;
    }

    inline static const std::vector<std::string> & county() {
        CHECK(EbolaData::init_);
        return EbolaData::county_;
    }

    inline static const std::vector<std::string> & loc() {
        CHECK(EbolaData::init_);
        return EbolaData::loc_;
    }

    inline static const std::vector<uint32_t> & region() {
        CHECK(EbolaData::init_);
        return EbolaData::region_;
    }

    inline static const std::vector<int> & outbreaks() {
        CHECK(EbolaData::init_);
        return EbolaData::outbreaks_;
    }

    inline static const std::vector<double> & population() {
        CHECK(EbolaData::init_);
        return EbolaData::population_;
    }

    inline static const std::vector<double> & x() {
        CHECK(EbolaData::init_);
        return EbolaData::x_;
    }

    inline static const std::vector<double> & y() {
        CHECK(EbolaData::init_);
        return EbolaData::y_;
    }

    static void init ();
};


} // namespace stdmMF


#endif // EBOLA_DATA_HPP
