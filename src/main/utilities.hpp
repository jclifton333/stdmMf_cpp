#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <memory>

namespace stdmMf {


template <typename T>
std::vector<std::shared_ptr<T> > clone_vec(
        const std::vector<std::shared_ptr<T> > & vec) {
    std::vector<std::shared_ptr<T> > vec_2(vec.size());
    std::transform(vec.begin(), vec.end(), vec_2.begin(),
            [] (const std::shared_ptr<T> & t_) {
                return std::static_pointer_cast<T>(t_->clone());
            });
    return vec_2;
}


} // namespace stdmMf


#endif // UTILITIES_HPP
