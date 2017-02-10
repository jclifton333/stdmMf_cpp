#include "pool.hpp"

#include <thread>

namespace stdmMf {

void Pool::worker_job() {
    this->service_->run();
}


Pool::Pool()
    : Pool(std::max(std::thread::hardware_concurrency(), 1u)) {
}

Pool::Pool(const uint32_t & num_threads)
    : num_threads_(num_threads), service_(new boost::asio::io_service),
      work_(new boost::asio::io_service::work(*this->service_)),
      workers_(), joined_(false) {
    for (uint32_t i = 0; i < num_threads_; ++i) {
        this->workers_.create_thread(boost::bind(&Pool::worker_job, this));
    }
}


Pool::Pool(const Pool & other)
    : Pool(other.num_threads_) {
}


Pool::~Pool() {
    if (!this->joined_) {
        this->join();
    }
}


const boost::shared_ptr<boost::asio::io_service> Pool::service() const {
    return this->service_;
}


void Pool::join() {
    this->joined_ = true;
    this->work_.reset();
    this->workers_.join_all();
}



} // namespace stdmMf
