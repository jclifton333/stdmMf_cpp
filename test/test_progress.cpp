#include <gtest/gtest.h>
#include <glog/logging.h>
#include "progress.hpp"

#include <sstream>
#include <regex>

namespace stdmMf {

TEST(TestProgress, TestNoTotal) {
    std::stringstream ss;
    Progress<std::stringstream> p(&ss);
    std::string match = "\\r    0 \\([ 0-9]{3}(.)[0-9]{2} hours\\)";
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));

    p.update();
    match += "\\r    1 \\([ 0-9]{3}(.)[0-9]{2} hours\\)";
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));

    p.update();
    match += "\\r    2 \\([ 0-9]{3}(.)[0-9]{2} hours\\)";
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));
}


TEST(TestProgress, TestTotal2) {
    std::stringstream ss;
    Progress<std::stringstream> p(2, &ss);

    const std::string hours = " \\([ 0-9]{3}(.)[0-9]{2} hours\\)";
    std::string match = "\\r    0 /     2 \\[ {20}\\]" + hours;
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));

    match += "\\r    1 /     2 \\[#{10} {10}\\]" + hours;
    p.update();
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));

    match += "\\r    2 /     2 \\[#{20}\\]" + hours;
    p.update();
    EXPECT_TRUE(std::regex_match(ss.str(),
                    std::regex(match)));
}



} // namespace stdmMf


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
