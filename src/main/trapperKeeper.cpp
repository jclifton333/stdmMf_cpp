#include "trapperKeeper.hpp"

#include "projectInfo.hpp"

#include <glog/logging.h>
#include <unistd.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>

namespace stdmMf {

std::string time_stamp() {
    const boost::posix_time::ptime now =
        boost::posix_time::second_clock::universal_time();

    std::stringstream ss;
    ss << now.date().year()
       << "-" << std::setfill('0') << std::setw(2)
       << static_cast<int>(now.date().month())
       << "-" << std::setfill('0') << std::setw(2) << now.date().day()
       << "_" << std::setfill('0') << std::setw(2) << now.time_of_day().hours()
       << "-" << std::setfill('0') << std::setw(2)
       << now.time_of_day().minutes()
       << "-" << std::setfill('0') << std::setw(2)
       << now.time_of_day().seconds();

    return ss.str();
}

std::ostream & operator<<(std::ostream & os, const Entry & r) {
    os << r.content_.str();
}


void Entry::wipe() {
    this->content_.str("");
    this->content_.clear();
}


Entry::Entry() {
}

Entry::Entry(const Entry & other) {
    // protect writing data
    std::lock_guard<std::mutex> lock(this->stream_mutex_);
    *this << other.content_.str();
}



TrapperKeeper::TrapperKeeper(const std::string & name,
        const boost::filesystem::path & root)
    : root_(root),
      temp_(boost::filesystem::temp_directory_path()
              / boost::filesystem::unique_path()),
      date_(time_stamp()),
      wiped_(false),
      finished_(false) {

    char hostname[HOST_NAME_MAX];
    int host_not_found = gethostname(hostname, HOST_NAME_MAX);

    // create readme entry
    Entry& readme = this->stream("README.txt");
    readme << "date: " << date_ << "\n";
    if (host_not_found) {
        readme << "host: " << "anonymous" << "\n";
    } else {
        readme << "host: " << hostname << "\n";
    }
    readme << "host: " << hostname << "\n"
           << "git-SHA-1: " << git_sha1 << "\n"
           << "name: " << name << "\n";
}

TrapperKeeper::~TrapperKeeper() {
    this->finished();
}

void TrapperKeeper::finished() {
    std::lock_guard<std::mutex> lock(this->filesystem_mutex_);
    if (!this->wiped_ && !this->finished_) {
        this->flush();

        // create root if doesn't exists
        if (!boost::filesystem::is_directory(this->root_)) {
            boost::filesystem::create_directories(this->root_);
        }

        if (!boost::filesystem::is_directory(this->root_ / this->date_)) {
            // use date as directory name if it doesn't already exists
            boost::filesystem::rename(this->temp_, this->root_ / this->date_);
        } else {
            // add a counter to date for a new directory name
            uint32_t counter = 1;
            std::stringstream ss;
            ss << "_" << std::setw(3) << std::setfill('0') << counter++;
            boost::filesystem::path new_path = this->root_ / this->date_;
            new_path += ss.str();
            while(boost::filesystem::is_directory(new_path)) {
                ss.str("");
                ss.clear();
                ss << "_" << std::setw(3) << std::setfill('0') << counter++;
                new_path = this->root_ / this->date_;
                new_path += ss.str();
            }

            boost::filesystem::rename(this->temp_, new_path);
        }

        this->finished_ = true;
    }
}

const boost::filesystem::path & TrapperKeeper::root() const {
    return this->root_;
}

const boost::filesystem::path & TrapperKeeper::temp() const {
    return this->temp_;
}

const boost::filesystem::path & TrapperKeeper::date() const {
    return this->date_;
}


void TrapperKeeper::wipe() {
    this->wiped_ = true;
    // remove temporary directory
    boost::filesystem::remove_all(this->temp_);
}


Entry & TrapperKeeper::stream(const boost::filesystem::path & entry_path) {
    CHECK(!this->wiped_);
    // return reference if exists, if not then create and return
    // reference
    return this->entries_[this->temp_ / entry_path];
}


void TrapperKeeper::flush() {
    // check to make sure temp directory exists
    if (!boost::filesystem::is_directory(this->temp_)) {
        boost::filesystem::create_directory(this->temp_);
    }

    CHECK(!this->wiped_);
    for (auto & pair : this->entries_) {
        // write to file and wipe the record
        std::ofstream ofs;
        ofs.open(pair.first.string().c_str(), std::ios_base::app);
        CHECK(ofs.good()) << pair.first;
        ofs << pair.second;
        ofs.close();
        pair.second.wipe();
    }
}



} // namespace stdmMf
