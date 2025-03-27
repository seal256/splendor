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

size_t weighted_random_choice(const std::vector<int>& weights);
