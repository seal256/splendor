# include "random_util.h"

// Has lienar time in weights len, but this doesn't matter for our purposes
size_t weighted_random_choice(const std::vector<int>& weights) {
    int total = std::accumulate(weights.begin(), weights.end(), 0);
    int threshold = rand() % total;
    
    int cumulative = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (cumulative >= threshold) {
            return i;
        }
    }
    
    return weights.size() - 1;
}
