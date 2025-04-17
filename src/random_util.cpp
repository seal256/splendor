#include "random_util.h"

// Dirichlet-distributed random vector
std::vector<double> sample_dirichlet(double alpha, size_t len) {
    thread_local std::mt19937 gen(std::rand());
    std::gamma_distribution<double> gamma(alpha, 1.0);

    std::vector<double> samples;
    samples.reserve(len);
    double sum = 0;
    for (size_t n = 0; n < len; ++n) {
        double sample = gamma(gen);
        sum += sample;
        samples.push_back(sample);
    }

    for (double& sample : samples) {
        sample /= sum;
    }

    return samples;
}