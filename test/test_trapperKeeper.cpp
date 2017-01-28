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
    const boost::filesystem::path file1 = "file1.txt";
    const boost::filesystem::path file2 = "dir2/file2.txt";
    const boost::filesystem::path file3 = "dir2/dir3/file3.txt";

    tk.entry(file1) << "hello" << "\n" << "world";
    tk.entry(file2) << "goodbye" << "\n" << "world";
    tk.entry(file3) << "hello" << "\n" << "again";
    tk.flush();

    EXPECT_TRUE(boost::filesystem::exists(temp / file1))
        << temp / file1;
    EXPECT_TRUE(boost::filesystem::exists(temp / file2))
        << temp / file2;
    EXPECT_TRUE(boost::filesystem::exists(temp / file3))
        << temp / file3;

    // test finished
    tk.finished();
    EXPECT_TRUE(boost::filesystem::exists(root)) << root;
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / "README.txt"))
        << root / tk.date() / "README.txt";
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / file1))
        << root / tk.date() / file1;
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / file2))
        << root / tk.date() / file2;
    EXPECT_TRUE(boost::filesystem::exists(root / tk.date() / file3))
        << root / tk.date() / file3;
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
    EXPECT_TRUE(boost::filesystem::exists(root / date001)) << root / date001;
    EXPECT_TRUE(boost::filesystem::exists(root / date001 / "README.txt"))
        << root / date001 / "README.txt";
    EXPECT_FALSE(boost::filesystem::exists(root / tk.date() / "README.txt"))
        << root / tk.date() / "README.txt";
    EXPECT_FALSE(boost::filesystem::exists(temp)) << temp;
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
