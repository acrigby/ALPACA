#include <iostream>

int main(){
    for( int ii=0; ii<10; ++ii ){
        char* input;
        char *output;
        std::cin >> input;
        long result = strtol(input,&output,);
        std::cout << result*2 << std::endl;
        std::cout.flush();
    }
}