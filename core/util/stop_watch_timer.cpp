#include "core/util/stop_watch_timer.h"

StopWatchTimer::StopWatchTimer(const std::string& timer_name)
: timer_name_(timer_name)
{

};

StopWatchTimer::~StopWatchTimer()
{

};


double StopWatchTimer::start(bool flag_verbose)
{
    start_ = std::chrono::high_resolution_clock::now();
    
    std::chrono::high_resolution_clock::duration gap = start_ - ref_time_;
    double gap_in_msec = (double)(gap/std::chrono::microseconds(1))*0.001;
    if(flag_verbose)
    {
        std::cout << "[" << timer_name_ << "]    start at: " << gap_in_msec << " [ms]\n";
    }

    return gap_in_msec;
};

double StopWatchTimer::lap(bool flag_verbose)
{
    intermediate_ = std::chrono::high_resolution_clock::now();
    
    std::chrono::high_resolution_clock::duration gap = intermediate_ - start_;
    double gap_in_msec = (double)(gap/std::chrono::microseconds(1))*0.001;
    if(flag_verbose)
    {
        std::cout << "[" << timer_name_ << "] lap time at: " << gap_in_msec << " [ms]\n";
    }

    return gap_in_msec;
};

double StopWatchTimer::stop(bool flag_verbose)
{
    end_ = std::chrono::high_resolution_clock::now();
    
    std::chrono::high_resolution_clock::duration gap = end_ - start_;
    double gap_in_msec = (double)(gap/std::chrono::microseconds(1))*0.001;
    if(flag_verbose)
    {
        std::cout << "[" << timer_name_ << "]      end at: " << gap_in_msec << " [ms]\n";
    }

    return gap_in_msec;
};