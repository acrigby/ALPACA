#include <iostream>

int main(){
    for( int ii=0; ii<10; ++ii ){
        int input;
        std::cin >> input;
        std::cout << input*2 << std::endl;
        std::cout.flush();
    }
}