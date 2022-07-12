#include "util/histogram.h"

namespace histogram {
  

    float medianHistogram(const std::vector<float>& hist_centers, std::vector<int>& hist_counts){
        float res = -1.0f;

        int nbins = hist_counts.size();
        int sum = 0;
        int sum_mid = 0;
        for(int i = 0; i < nbins; ++i){
            sum += hist_counts[i];
            // std::cout << hist_counts[i] <<std::endl;
        }
        
        std::vector<std::pair<int,float>> data(nbins);
        for(int i = 0; i < nbins; ++i){
            data[i] = {hist_counts[i], hist_centers[i]};
        }

        std::sort(data.begin(),data.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) { return a.first > b.first; });

        // std::cout << " largest : " << data.front().first <<"," <<data.front().second << std::endl;
        // std::cout << " smallest : " << data.back().first <<"," <<data.back().second << std::endl;

        // res = hist_centers[idx_median];
        return data.front().second ;
    };
};
