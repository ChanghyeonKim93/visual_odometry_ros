
#include "timer.h"
namespace timer{
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    auto gap = finish - start;

    void tic(){
        start = std::chrono::high_resolution_clock::now();
    };
    
    double toc(bool flag_verbose){
        finish = std::chrono::high_resolution_clock::now();
        gap = finish - start;
        if(flag_verbose){
            std::cout << " exec time: " << (double)(gap/std::chrono::microseconds(1)) / 1000.0 << "[ms]\n";
        }
        return (double)(gap/std::chrono::microseconds(1))/1000.0;
    };

    // Get current data/time, format is yyyy-mm-dd.hh:mm:ss
    const std::string currentDateTime(){
        time_t now = time(0);
        struct tm tstruct;
        char buf[80];
        tstruct = *localtime(&now);
        // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
        // for more information about data/time format
        strftime(buf, sizeof(buf), "%Y-%m-%d_%H_%M_%S", &tstruct);

        return buf;
    };
};