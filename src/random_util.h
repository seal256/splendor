#include <algorithm>
#include <random>

// TODO: get rid of std::rand()

template<typename RandomIt>
void random_shuffle(RandomIt first, RandomIt last) {
    // thread_local std::random_device rd;
    thread_local std::mt19937 g(std::rand());
    std::shuffle(first, last, g);
}

template<typename T>
size_t weighted_random_choice(const std::vector<T>& weights) {
    const T total = std::accumulate(weights.begin(), weights.end(), T());
    if (total <= T()) {
        throw std::invalid_argument("Negative or zero weights in weighted_random_choice");
    }
    thread_local std::mt19937 gen(std::rand());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    const T threshold = static_cast<T>(dis(gen) * total);
 
    T cumulative = 0;
    for (size_t n = 0; n < weights.size(); ++n) {
        cumulative += weights[n];
        if (cumulative > threshold) {
            return n;
        }
    }
  
    return weights.size() - 1;
}

std::vector<double> sample_dirichlet(double alpha, size_t len);