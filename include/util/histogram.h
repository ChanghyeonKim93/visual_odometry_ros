#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>

namespace histogram {
    template <typename T>
    void makeHistogram(const std::vector<T> vals, T hist_min, T hist_max, int nbins,
        std::vector<float>& hist_centers, std::vector<int>& hist_counts){

            int n_data = vals.size();

            float step = (float)(hist_max-hist_min)/(float)nbins;

            hist_centers.resize(nbins);
            hist_centers[0] = (float)hist_min;
            hist_centers[nbins-1] = (float)hist_max;
            for(int i = 1; i < nbins-1;++i){
                hist_centers[i] = hist_centers[i-1] + step;
                // std::cout << hist_centers[i] <<std::endl;
            }

            hist_counts.resize(nbins,0);
            for(int i = 0; i < n_data; ++i){
                float val = (float)vals[i];
                val -= (float)hist_min;
                int idx = (int)floor(val/step);
                if(idx >= 0 && idx < nbins){
                    ++hist_counts[idx];
                }            
            }
        };
        
    float medianHistogram(const std::vector<float>& hist_centers, std::vector<int>& hist_counts);
};


#endif