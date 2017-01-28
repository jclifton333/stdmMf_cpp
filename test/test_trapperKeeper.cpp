#include <gtest/gtest.h>
#include <glog/logging.h>
#include "trapperKeeper.hpp"


namespace stdmMf {


TEST(TestTrapperKeeper, FlushAndFinish) {
    // create root in temp directory
    const boost::filesystem::path root =
        boost::filesystem::temp_directory_path()
        / boost::filesystem::unique_path();

    TrapperKeeper tk("test", root);
    const boost::filesystem::path & temp = tk.temp();

    // check directory is created
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
    tk.flush();

    EXPECT_TRUE(boost::filesystem::exists(temp)) << temp;
    EXPECT_TRUE(boost::filesystem::exists(temp / "README.txt"))
        << temp / "README.txt";

    // test two files
    tk.entry("file1.txt") << "hello" << "\n" << "world";
    tk.entry("file2.txt") << "goodbye" << "\n" << "world";
    tk.flush();
    EXPECT_TRUE(boost::filesystem::exists(temp / "file1.txt"))
        << temp / "file1.txt";
    EXPECT_TRUE(boost::filesystem::exists(temp / "file2.txt"))
        << temp / "file2.txt";

    // test finished
    tk.finished();
    EXPECT_TRUE(boost::filesystem::exists(root)) << root;
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / "README.txt"))
        << root / tk.date() / "README.txt";
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / "file1.txt"))
        << root / tk.date() / "file1.txt";
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / "file2.txt"))
        << root / tk.date() / "file2.txt";
}


TEST(TestTrapperKeeper, Existing001) {
    // create root in temp directory
    const boost::filesystem::path root =
        boost::filesystem::temp_directory_path()
        / boost::filesystem::unique_path();

    TrapperKeeper tk("test", root);
    const boost::filesystem::path & temp = tk.temp();

    // check directory is created
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
    tk.flush();

    EXPECT_TRUE(boost::filesystem::exists(temp)) << temp;
    EXPECT_TRUE(boost::filesystem::exists(temp / "README.txt"))
        << temp / "README.txt";

    // create date directory
    boost::filesystem::create_directory(root);
    boost::filesystem::create_directory(root / tk.date());
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date()));

    tk.finished();
    boost::filesystem::path date001 = tk.date();
    date001 += "_001";
    EXPECT_TRUE(boost::filesystem::exists(root / date001));
    EXPECT_TRUE(boost::filesystem::exists(root / date001 / "README.txt"));
    EXPECT_FALSE(boost::filesystem::exists(root / tk.date() / "README.txt"));

}



TEST(TestTrapperKeeper, Existing003) {
    // create root in temp directory
    const boost::filesystem::path root =
        boost::filesystem::temp_directory_path()
        / boost::filesystem::unique_path();

    TrapperKeeper tk("test", root);
    const boost::filesystem::path & temp = tk.temp();

    // check directory is created
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
    tk.flush();

    EXPECT_TRUE(boost::filesystem::exists(temp)) << temp;
    EXPECT_TRUE(boost::filesystem::exists(temp / "README.txt"))
        << temp / "README.txt";

    // create date directories
    boost::filesystem::path date001 = tk.date();
    date001 += "_001";
    boost::filesystem::path date002 = tk.date();
    date002 += "_002";
    boost::filesystem::create_directories(root / tk.date());
    boost::filesystem::create_directories(root / date001);
    boost::filesystem::create_directories(root / date002);
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date()));
    EXPECT_TRUE(boost::filesystem::exists(root / date001));
    EXPECT_TRUE(boost::filesystem::exists(root / date002));

    tk.finished();
    boost::filesystem::path date003 = tk.date();
    date003 += "_003";
    EXPECT_TRUE(boost::filesystem::exists(root / date003));
    EXPECT_TRUE(boost::filesystem::exists(root / date003 / "README.txt"));
    EXPECT_FALSE(boost::filesystem::exists(root / tk.date() / "README.txt"));
    EXPECT_FALSE(boost::filesystem::exists(root / date001 / "README.txt"));
    EXPECT_FALSE(boost::filesystem::exists(root / date002 / "README.txt"));
}



} // namespace stdmMf


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
