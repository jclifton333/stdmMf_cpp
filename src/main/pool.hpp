#ifndef POOL_HPP
#define POOL_HPP

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "result.hpp"

namespace stdmMf {


class Pool {
protected:
    const uint32_t & num_threads_;

    const boost::shared_ptr<boost::asio::io_service> service_;
    boost::shared_ptr<boost::asio::io_service::work> work_;

    boost::thread_group workers_;

    void worker_job();

    bool joined_;

public:
    Pool(const uint32_t & num_threads);

    Pool(const Pool & other);

    ~Pool();

    const boost::shared_ptr<boost::asio::io_service> service() const;

    void join();
};

} // namespace stdmMf


#endif // POOL_HPP
