#include "ebolaData.hpp"

#include <njm_cpp/info/project.hpp>
#include <iterator>
#include <fstream>

#include <iostream>

namespace stdmMf {

bool EbolaData::init_ = false;

std::vector<std::string> EbolaData::country_;
std::vector<std::string> EbolaData::county_;
std::vector<std::string> EbolaData::loc_;
std::vector<uint32_t> EbolaData::region_;
std::vector<int> EbolaData::outbreaks_;
std::vector<double> EbolaData::population_;
std::vector<double> EbolaData::x_;
std::vector<double> EbolaData::y_;

void EbolaData::init() {
    if (EbolaData::init_) {
        return;
    }

    const std::string ebola_root_dir(
            njm::info::project::PROJECT_ROOT_DIR + "/src/data/");
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_country.txt"};
        CHECK(inputFile.good()) << "ebola_country";
        std::istream_iterator<std::string> input(inputFile);
        std::copy(input, std::istream_iterator<std::string>(),
                std::back_inserter(EbolaData::country_));
        CHECK_EQ(EbolaData::country_.size(), 290) << "ebola_country";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_county.txt"};
        CHECK(inputFile.good()) << "ebola_county";
        std::istream_iterator<std::string> input(inputFile);
        std::copy(input, std::istream_iterator<std::string>(),
                std::back_inserter(EbolaData::county_));
        CHECK_EQ(EbolaData::county_.size(), 290) << "ebola_county";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_loc.txt"};
        CHECK(inputFile.good()) << "ebola_loc";
        std::istream_iterator<std::string> input(inputFile);
        std::copy(input, std::istream_iterator<std::string>(),
                std::back_inserter(EbolaData::loc_));
        CHECK_EQ(EbolaData::loc_.size(), 290) << "ebola_loc";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_region.txt"};
        CHECK(inputFile.good()) << "ebola_region";
        std::istream_iterator<uint32_t> input(inputFile);
        std::copy(input, std::istream_iterator<uint32_t>(),
                std::back_inserter(EbolaData::region_));
        CHECK_EQ(EbolaData::region_.size(), 290) << "ebola_region";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_outbreaks.txt"};
        CHECK(inputFile.good()) << "ebola_outbreaks";
        std::istream_iterator<int> input(inputFile);
        std::copy(input, std::istream_iterator<int>(),
                std::back_inserter(EbolaData::outbreaks_));
        CHECK_EQ(EbolaData::outbreaks_.size(), 290) << "ebola_outbreaks";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_population.txt"};
        CHECK(inputFile.good()) << "ebola_population";
        std::istream_iterator<double> input(inputFile);
        std::copy(input, std::istream_iterator<double>(),
                std::back_inserter(EbolaData::population_));
        CHECK_EQ(EbolaData::population_.size(), 290) << "ebola_population";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_x.txt"};
        CHECK(inputFile.good()) << "ebola_x";
        std::istream_iterator<double> input(inputFile);
        std::copy(input, std::istream_iterator<double>(),
                std::back_inserter(EbolaData::x_));
        CHECK_EQ(EbolaData::x_.size(), 290) << "ebola_x";
        inputFile.close();
    }
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_y.txt"};
        CHECK(inputFile.good()) << "ebola_y";
        std::istream_iterator<double> input(inputFile);
        std::copy(input, std::istream_iterator<double>(),
                std::back_inserter(EbolaData::y_));
        CHECK_EQ(EbolaData::y_.size(), 290) << "ebola_y";
        inputFile.close();
    }
    EbolaData::init_ = true;
}



} // namespace stdmMf
