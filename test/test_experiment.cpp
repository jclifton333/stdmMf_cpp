#include <gtest/gtest.h>
#include <glog/logging.h>
#include "experiment.hpp"

namespace stdmMf {


TEST(TestExperiment, IntAndDouble) {

    Experiment e;
    e.add_factor(std::vector<int>({-1, 0, 1}));
    e.add_factor(std::vector<double>({-0.1, 0.0, 0.1}));

    e.start();
    Experiment::Factor f;
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, -1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, -0.1);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 0);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, -0.1);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, -0.1);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, -1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.0);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 0);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.0);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.0);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, -1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.1);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 0);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.1);

    CHECK(e.next());
    f = e.get();
    CHECK_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    CHECK_EQ(f.at(0).val.int_val, 1);
    CHECK_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    CHECK_EQ(f.at(1).val.double_val, 0.1);

    CHECK(!e.next());
}


} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
