#ifndef _STOP_WATCH_TIMER_H_
#define _STOP_WATCH_TIMER_H_

#include <iostream>
#include <string>
#include <chrono>

class StopWatchTimer
{
private:
    std::string timer_name_;

    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point intermediate_;
    std::chrono::high_resolution_clock::time_point end_;


public:
    inline static std::chrono::high_resolution_clock::time_point ref_time_ = std::chrono::high_resolution_clock::now();

public:
    StopWatchTimer(const std::string& timer_name);
    ~StopWatchTimer();

public:
    double start(bool flag_verbose = false);
    double lap(bool flag_verbose = false);
    double stop(bool flag_verbose = false);   

};
#endif