#include "pool.hpp"

namespace stdmMf {

void Pool::worker_job() {
    this->service_->run();
}


Pool::Pool(const uint32_t & num_threads)
    : num_threads_(num_threads), service_(new boost::asio::io_service),
      work_(new boost::asio::io_service::work(*this->service_)),
      workers_() {
    for (uint32_t i = 0; i < num_threads_; ++i) {
        this->workers_.create_thread(boost::bind(&Pool::worker_job, this));
    }
}


const boost::shared_ptr<boost::asio::io_service> Pool::service() const {
    return this->service_;
}


void Pool::join() {
    this->work_.reset();
    this->workers_.join_all();
}



} // namespace stdmMf
