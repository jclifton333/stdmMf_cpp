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
std::vector<std::pair<uint32_t, uint32_t> > EbolaData::edges_;

void EbolaData::deinit() {
    EbolaData::country_.clear();
    EbolaData::county_.clear();
    EbolaData::loc_.clear();
    EbolaData::region_.clear();
    EbolaData::outbreaks_.clear();
    EbolaData::population_.clear();
    EbolaData::x_.clear();
    EbolaData::y_.clear();
    EbolaData::edges_.clear();
    EbolaData::init_ = false;
}


void EbolaData::init(
        const std::vector<std::string> & country,
        const std::vector<std::string> & county,
        const std::vector<std::string> & loc,
        const std::vector<uint32_t> & region,
        const std::vector<int> & outbreaks,
        const std::vector<double> & population,
        const std::vector<double> & x,
        const std::vector<double> & y,
        const std::vector<std::pair<uint32_t, uint32_t> > & edges) {
    if (EbolaData::init_) {
        return;
    }

    EbolaData::country_ = country;
    EbolaData::county_ = county;
    EbolaData::loc_ = loc;
    EbolaData::region_ = region;
    EbolaData::outbreaks_ = outbreaks;
    EbolaData::population_ = population;
    EbolaData::x_ = x;
    EbolaData::y_ = y;
    EbolaData::edges_ = edges;

    EbolaData::init_ = true;
}


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
    {
        std::ifstream inputFile{ebola_root_dir + "ebola_edges.txt"};
        CHECK(inputFile.good()) << "ebola_edges";
        std::istream_iterator<uint32_t> input(inputFile);
        std::vector<uint32_t> edge_vec;
        std::copy(input, std::istream_iterator<uint32_t>(),
                std::back_inserter(edge_vec));
        CHECK_EQ(edge_vec.size() % 2, 0);
        for (uint32_t i = 0; i < edge_vec.size(); i += 2) {
            EbolaData::edges_.emplace_back(edge_vec.at(i), edge_vec.at(i + 1));
        }
        inputFile.close();
    }

    EbolaData::init_ = true;
}



} // namespace stdmMf
