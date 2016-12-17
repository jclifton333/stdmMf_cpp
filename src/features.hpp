#ifndef FEATURES_HPP
#define FEATURES_HPP


namespace stdmMf {


class Features {
public:
    virtual std::shared_ptr<Features> clone() const = 0;

    virtual std::vector<double> get_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits) = 0;

    virtual void update_features(
            const uint32_t & changed_node,
            const boost::dynamic_bitset<> & inf_bits_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const boost::dynamic_bitset<> & inf_bits_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) = 0;

    virtual uint32_t num_features() const = 0;
};


} // namespace stdmMf


#endif // FEATURES_HPP
