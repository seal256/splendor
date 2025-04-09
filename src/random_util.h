#include <algorithm>
#include <random>

// This solution is far from perfect
// TODO: properly migrate random generators to C++17 standards
template<typename RandomIt>
void random_shuffle(RandomIt first, RandomIt last) {
    thread_local std::random_device rd;
    thread_local std::mt19937 g(std::rand());
    std::shuffle(first, last, g);
}

template<typename T>
size_t weighted_random_choice(const std::vector<T>& weights) {
    const T total = std::accumulate(weights.begin(), weights.end(), T());
    const double random_float = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    const T threshold = static_cast<T>(random_float * total);
    
    T cumulative = 0;
    for (size_t n = 0; n < weights.size(); ++n) {
        cumulative += weights[n];
        if (cumulative > threshold) {
            return n;
        }
    }
  
    return weights.size() - 1;
}