#ifndef TRAPPER_KEEPER_HPP
#define TRAPPER_KEEPER_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <boost/filesystem.hpp>
#include <mutex>


namespace stdmMf {


std::string time_stamp();


class TrapperKeeper;

class Entry {
protected:
    friend class TrapperKeeper;

    std::stringstream content_;

    friend std::ostream & operator<<(std::ostream & os, const Entry & r);

    void wipe();

    std::mutex stream_mutex_;

public:
    Entry();
    Entry(const Entry & other);

    template<class T>
    Entry& operator<<(const T & new_content) {
        this->content_ << new_content;
        return *this;
    }
};


class TrapperKeeper {
    std::map<boost::filesystem::path, Entry> entries_;
    const boost::filesystem::path root_;
    const boost::filesystem::path temp_;
    const boost::filesystem::path date_;

    bool wiped_;
    bool finished_;

    std::mutex filesystem_mutex_;

public:
    TrapperKeeper(const std::string & name,
            const boost::filesystem::path & root);
    ~TrapperKeeper();

    void finished();

    const boost::filesystem::path & root() const;

    const boost::filesystem::path & temp() const;

    const boost::filesystem::path & date() const;

    void wipe();

    Entry& entry(const boost::filesystem::path & entry_path);

    void flush();
};


} // namespace stdmMf


#endif // TRAPPER_KEEPER_HPP
