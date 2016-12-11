#ifndef FEATURES_HPP
#define FEATURES_HPP


namespace stdmMf {


class Features {
public:
    virtual std::vector<double> get_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) = 0;

    virtual std::vector<double> update_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits,
            const std::vector<double> & prev_feat) = 0;

    virtual uint32_t num_features() = 0;
};


} // namespace stdmMf


#endif // FEATURES_HPP
