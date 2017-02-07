#include <gtest/gtest.h>
#include <glog/logging.h>
#include "progress.hpp"

#include <sstream>
#include <regex>

namespace stdmMf {

TEST(TestProgress, TestNoTotal) {
    std::stringstream ss;
    Progress<std::stringstream> p(&ss);
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex("^(\r)    0 [(][ 0-9]{3}[.][0-9]{2} hours[)]$")));

    p.update();
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex("^[^]*\\r    1"
                            " [(][ 0-9]{3}[.][0-9]{2} hours[)]$"
                            )));

    p.update();
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex("^[^]*\\r    2"
                            " [(][ 0-9]{3}[.][0-9]{2} hours[)]$")));
}


} // namespace stdmMf


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
