#include "core/util/timer.h"

namespace timer{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point finish  = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration gap = finish - start;

    void tic(){
        start = std::chrono::high_resolution_clock::now();
    };

    // return elapsed time from "tic()" in milliseconds.
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

    StopWatch::StopWatch(const std::string& timer_name)
    : timer_name_(timer_name)
    {

    };

    StopWatch::~StopWatch()
    {

    };

    double StopWatch::start(bool flag_verbose)
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

    double StopWatch::lapTimeFromStart(bool flag_verbose)
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

    double StopWatch::stop(bool flag_verbose)
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
};