#ifndef _TIMER_H_
#define _TIMER_H_
#include <iostream>
#include <chrono>

namespace timer{
    void tic();
    double toc(bool flag_verbose);
    const std::string currentDateTime();
};
#endif