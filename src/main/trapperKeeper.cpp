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
    std::lock_guard<std::mutex> lock(r.mutex_);

    os << r.content_.str();
}


void Entry::wipe() {
    std::lock_guard<std::mutex> lock(this->mutex_);

    this->content_.str("");
    this->content_.clear();
}

std::string Entry::retrieve_and_wipe() {
    std::lock_guard<std::mutex> lock(this->mutex_);

    std::string str = this->content_.str();
    this->content_.str("");
    this->content_.clear();

    return str;
}


Entry::Entry(const boost::filesystem::path & path)
    : path_(path) {
}


Entry::Entry(const Entry & other)
    : path_(other.path_) {
    // protect writing data
    std::lock(this->mutex_, other.mutex_);
    std::lock_guard<std::mutex> this_lock(this->mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> other_lock(other.mutex_, std::adopt_lock);
    this->content_ << other.content_.str();
}

boost::filesystem::path Entry::path() {
    return this->path_;
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
    Entry& readme = this->entry("README.txt");
    readme << "date: " << date_ << "\n";
    if (host_not_found) {
        readme << "host: " << "anonymous" << "\n";
    } else {
        readme << "host: " << hostname << "\n";
    }
    readme << "host: " << hostname << "\n"
           << "git-SHA-1: " << GIT_SHA1 << "\n"
           << "name: " << name << "\n";
}

TrapperKeeper::~TrapperKeeper() {
    this->finished();
}

void TrapperKeeper::finished() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (!this->wiped_ && !this->finished_) {
        this->flush_no_lock();

        // create root if doesn't exists
        if (!boost::filesystem::is_directory(this->root_)) {
            boost::filesystem::create_directories(this->root_);
        }

        if (!boost::filesystem::is_directory(this->root_ / this->date_)) {
            // use date as directory name if it doesn't already exists
            this->copy_contents(this->temp_, this->root_ / this->date_);
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

            this->copy_contents(this->temp_, new_path);
        }

        this->finished_ = true;
        this->wipe_no_lock();
    }
}


void TrapperKeeper::copy_contents(const boost::filesystem::path & source,
        const boost::filesystem::path & dest) {
    if (!boost::filesystem::is_directory(dest)) {
        boost::filesystem::create_directory(dest);
    }

    boost::filesystem::directory_iterator it(source), end;
    for (; it != end; ++it) {
        boost::filesystem::path current(it->path());
        // copy directory
        if (boost::filesystem::is_directory(current)) {
            this->copy_contents(current, dest / current.filename());
        } else {
            boost::filesystem::copy_file(current, dest / current.filename());
        }
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
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->wipe_no_lock();
}


void TrapperKeeper::wipe_no_lock() {
    this->wiped_ = true;
    // remove temporary directory
    boost::filesystem::remove_all(this->temp_);
}


Entry & TrapperKeeper::entry(const boost::filesystem::path & entry_path) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    CHECK(!this->wiped_);
    // return reference if exists, if not then create and return
    // reference
    this->entries_.emplace_back(Entry(this->temp_ / entry_path));
    return this->entries_.back();
}

void TrapperKeeper::flush() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->flush_no_lock();
}

void TrapperKeeper::flush_no_lock() {
    // check to make sure temp directory exists
    if (!boost::filesystem::is_directory(this->temp_)) {
        boost::filesystem::create_directory(this->temp_);
    }

    CHECK(!this->wiped_);
    for (auto & entry : this->entries_) {
        // create directories
        if (!boost::filesystem::is_directory(entry.path().parent_path())) {
            boost::filesystem::create_directories(entry.path().parent_path());
        }

        // write to file and wipe the record
        std::ofstream ofs;
        ofs.open(entry.path().string().c_str(), std::ios_base::app);
        CHECK(ofs.good()) << entry.path().string();
        ofs << entry.retrieve_and_wipe();
        ofs.close();
    }
    this->entries_.clear();
}



} // namespace stdmMf
