#include <gtest/gtest.h>
#include <glog/logging.h>
#include "utilities.hpp"

namespace stdmMf {


TEST(TestUtilities, ActiveSet) {
    boost::dynamic_bitset<> bs(4);

    bs.set(1);
    bs.set(3);

    std::vector<uint32_t> active = active_set(bs);
    EXPECT_EQ(active.size(), 2);
    EXPECT_EQ(active.at(0), 1);
    EXPECT_EQ(active.at(1), 3);

    bs.reset();
    bs.set(0);
    bs.set(3);

    active = active_set(bs);
    EXPECT_EQ(active.size(), 2);
    EXPECT_EQ(active.at(0), 0);
    EXPECT_EQ(active.at(1), 3);

    bs.set();

    active = active_set(bs);
    EXPECT_EQ(active.size(), 4);
    EXPECT_EQ(active.at(0), 0);
    EXPECT_EQ(active.at(1), 1);
    EXPECT_EQ(active.at(2), 2);
    EXPECT_EQ(active.at(3), 3);


    bs.reset();

    active = active_set(bs);
    EXPECT_EQ(active.size(), 0);
}

TEST(TestUtilities, InactiveSet) {
    boost::dynamic_bitset<> bs(4);

    bs.set(1);
    bs.set(3);

    std::vector<uint32_t> inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 2);
    EXPECT_EQ(inactive.at(0), 0);
    EXPECT_EQ(inactive.at(1), 2);

    bs.reset();
    bs.set(0);
    bs.set(3);

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 2);
    EXPECT_EQ(inactive.at(0), 1);
    EXPECT_EQ(inactive.at(1), 2);

    bs.set();

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 0);

    bs.reset();

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 4);
    EXPECT_EQ(inactive.at(0), 0);
    EXPECT_EQ(inactive.at(1), 1);
    EXPECT_EQ(inactive.at(2), 2);
    EXPECT_EQ(inactive.at(3), 3);
}

TEST(TestUtilities, BothSets) {
    boost::dynamic_bitset<> bs(4);

    bs.set(1);
    bs.set(3);

    std::vector<uint32_t> active = active_set(bs);
    EXPECT_EQ(active.size(), 2);
    EXPECT_EQ(active.at(0), 1);
    EXPECT_EQ(active.at(1), 3);
    std::vector<uint32_t> inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 2);
    EXPECT_EQ(inactive.at(0), 0);
    EXPECT_EQ(inactive.at(1), 2);

    bs.reset();
    bs.set(0);
    bs.set(3);

    active = active_set(bs);
    EXPECT_EQ(active.size(), 2);
    EXPECT_EQ(active.at(0), 0);
    EXPECT_EQ(active.at(1), 3);

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 2);
    EXPECT_EQ(inactive.at(0), 1);
    EXPECT_EQ(inactive.at(1), 2);

    bs.set();

    active = active_set(bs);
    EXPECT_EQ(active.size(), 4);
    EXPECT_EQ(active.at(0), 0);
    EXPECT_EQ(active.at(1), 1);
    EXPECT_EQ(active.at(2), 2);
    EXPECT_EQ(active.at(3), 3);

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 0);

    bs.reset();

    active = active_set(bs);
    EXPECT_EQ(active.size(), 0);

    inactive = inactive_set(bs);
    EXPECT_EQ(inactive.size(), 4);
    EXPECT_EQ(inactive.at(0), 0);
    EXPECT_EQ(inactive.at(1), 1);
    EXPECT_EQ(inactive.at(2), 2);
    EXPECT_EQ(inactive.at(3), 3);
}

TEST(TestUtilities, CombineSets) {
    boost::dynamic_bitset<> one(4);
    boost::dynamic_bitset<> two(4);

    one.set(2);
    one.set(3);

    two.set(1);
    two.set(3);

    std::vector<uint32_t> combined = combine_sets(one, two);
    EXPECT_EQ(combined.size(), 4);
    EXPECT_EQ(combined.at(0), 0);
    EXPECT_EQ(combined.at(1), 1);
    EXPECT_EQ(combined.at(2), 2);
    EXPECT_EQ(combined.at(3), 3);

    combined = combine_sets(two, one);
    EXPECT_EQ(combined.size(), 4);
    EXPECT_EQ(combined.at(0), 0);
    EXPECT_EQ(combined.at(1), 2);
    EXPECT_EQ(combined.at(2), 1);
    EXPECT_EQ(combined.at(3), 3);
}

TEST(TestUtilities, VectorAddAAndB) {
    const std::vector<double> a = {0.0, 1.0, 1.5, -1.0, -1.5};
    const std::vector<double> b = {0.1, 0.2, -0.1, 0.3, -0.2};

    const std::vector<double> c = {0.1, 1.2, 1.4, -0.7, -1.7};
    const std::vector<double> res = add_a_and_b(a,b);

    ASSERT_EQ(a.size(), b.size());
    ASSERT_EQ(a.size(), c.size());

    EXPECT_EQ(a.size(), res.size());
    for (uint32_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
            << "Failed for index i, a = " << a.at(i) << " and b = " << b.at(i);
    }
}

TEST(TestUtilities, VectorAddBToA) {
    std::vector<double> a = {0.0, 1.0, 1.5, -1.0, -1.5};
    const std::vector<double> b = {0.1, 0.2, -0.1, 0.3, -0.2};

    const std::vector<double> c = {0.1, 1.2, 1.4, -0.7, -1.7};
    add_b_to_a(a,b);

    ASSERT_EQ(a.size(), b.size());
    ASSERT_EQ(a.size(), c.size());
    for (uint32_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a.at(i), c.at(i), 1e-14)
            << "Failed for index i, a = " << a.at(i) << " and b = " << b.at(i);
    }
}

TEST(TestUtilities, ScalarAddAAndB) {
    const std::vector<double> a = {0.0, 1.0, 1.5, -1.0, -1.5};
    const std::vector<double> b = {0.1, 0.2, -0.1, 0.3, -0.2};

    uint32_t index = 0;
    {
        const std::vector<double> c = {0.1, 1.1, 1.6, -0.9, -1.4};
        const std::vector<double> res = add_a_and_b(a,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {0.2, 1.2, 1.7, -0.8, -1.3};
        const std::vector<double> res = add_a_and_b(a,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {-0.1, 0.9, 1.4, -1.1, -1.6};
        const std::vector<double> res = add_a_and_b(a,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {0.3, 1.3, 1.8, -0.7, -1.2};
        const std::vector<double> res = add_a_and_b(a,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {-0.2, 0.8, 1.3, -1.2, -1.7};
        const std::vector<double> res = add_a_and_b(a,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;
}

TEST(TestUtilities, ScalarAddBToA) {
    const std::vector<double> a = {0.0, 1.0, 1.5, -1.0, -1.5};
    const std::vector<double> b = {0.1, 0.2, -0.1, 0.3, -0.2};

    uint32_t index = 0;
    {
        const std::vector<double> c = {0.1, 1.1, 1.6, -0.9, -1.4};
        std::vector<double> res = a;
        add_b_to_a(res,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {0.2, 1.2, 1.7, -0.8, -1.3};
        std::vector<double> res = a;
        add_b_to_a(res,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {-0.1, 0.9, 1.4, -1.1, -1.6};
        std::vector<double> res = a;
        add_b_to_a(res,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {0.3, 1.3, 1.8, -0.7, -1.2};
        std::vector<double> res = a;
        add_b_to_a(res,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;

    {
        const std::vector<double> c = {-0.2, 0.8, 1.3, -1.2, -1.7};
        std::vector<double> res = a;
        add_b_to_a(res,b.at(index));

        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), c.size());

        EXPECT_EQ(a.size(), res.size());
        for (uint32_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(res.at(i), c.at(i), 1e-14)
                << "Failed for element i = " << i << " with a = " << a.at(i)
                << " and b = " << b.at(index);
        }
    }
    ++index;
}

TEST(TestUtilities, MultAAndB) {
}


} // namespace stdmMf


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
