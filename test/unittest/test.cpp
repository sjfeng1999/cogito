//
//
//
//

#include <gtest/gtest.h>


TEST(General, MAIN){
    EXPECT_TRUE(2 / 2 == 1);
}

int main(){
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}