#include <gtest/gtest.h>
#include <glog/logging.h>
#include "experiment.hpp"

namespace stdmMf {


TEST(TestExperiment, IntAndDouble) {

    Experiment e;
    Experiment::FactorGroup * g = e.add_group();
    g->add_factor(std::vector<int>({-1, 0, 1}));
    g->add_factor(std::vector<double>({-0.1, 0.0, 0.1}));

    e.start();
    Experiment::Factor f;
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 0);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.0);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 0);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.0);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.0);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 0);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.1);

    ASSERT_TRUE(!e.next());
}


TEST(TestExperiment, MultiGroupIntAndDouble) {
    Experiment e;
    Experiment::FactorGroup * g0 = e.add_group();
    g0->add_factor(std::vector<int>({-1, 1}));
    g0->add_factor(std::vector<double>({-0.1, 0.1}));

    Experiment::FactorGroup * g1 = e.add_group();
    g1->add_factor(std::vector<int>({-2, 2}));
    g1->add_factor(std::vector<double>({-0.2, 0.2}));

    e.start();

    Experiment::Factor f;
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 1);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.1);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -2);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.2);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 2);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, -0.2);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, -2);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.2);

    ASSERT_TRUE(e.next());
    f = e.get();
    EXPECT_EQ(f.at(0).type, Experiment::FactorLevel::Type::is_int);
    EXPECT_EQ(f.at(0).val.int_val, 2);
    EXPECT_EQ(f.at(1).type, Experiment::FactorLevel::Type::is_double);
    EXPECT_EQ(f.at(1).val.double_val, 0.2);

    ASSERT_TRUE(!e.next());
}


} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
