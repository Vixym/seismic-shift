#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <future>
#include <getopt.h>

// Configuration struct to hold program settings
struct ProgramConfig
{
    int n = 10000;              // Number of vectors
    int d = 100000;             // Original dimension
    int k = 100;                // Target dimension
    bool compare_dense = false; // Whether to compare with dense JLT
    bool verbose = true;        // Verbose output
};

// Utility functions for analysis
namespace utils
{
    // Calculate Euclidean distance between two vectors
    double euclidean_distance(const std::vector<double> &a, const std::vector<double> &b)
    {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Generate random data for testing
    std::vector<std::vector<double>> generate_random_data(int n, int d, unsigned int seed = 42)
    {
        // Create output vector with pre-allocated space
        std::vector<std::vector<double>> X(n, std::vector<double>(d));

        // Determine number of threads
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4; // Fallback if hardware_concurrency returns 0

        // Parallel generation of random data
        std::vector<std::future<void>> futures;
        for (int thread = 0; thread < num_threads; ++thread)
        {
            futures.push_back(std::async(std::launch::async, [&](int thread_id)
                                         {
            // Create thread-local random generator
            std::mt19937 gen(seed + thread_id);
            std::normal_distribution<> dist(0, 1);

            // Divide work across threads
            int start = thread_id * (n / num_threads);
            int end = (thread_id == num_threads - 1) 
                      ? n 
                      : (thread_id + 1) * (n / num_threads);

            // Generate data for this thread's subset of vectors
            for (int i = start; i < end; ++i)
            {
                for (int j = 0; j < d; ++j)
                {
                    X[i][j] = dist(gen);
                }
            } }, thread));
        }

        // Wait for all threads to complete
        for (auto &future : futures)
        {
            future.wait();
        }

        return X;
    }

    // Calculate relative errors in distance preservation
    double calculate_relative_error(const std::vector<std::vector<double>> &X,
                                    const std::vector<std::vector<double>> &Y,
                                    int num_samples)
    {
        int n = X.size();

        // Sample pairs for comparison
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n - 1);

        double total_relative_error = 0.0;
        int count = 0;

        for (int s = 0; s < num_samples; ++s)
        {
            int i = dist(gen);
            int j = dist(gen);
            if (i != j)
            {
                double original_dist = euclidean_distance(X[i], X[j]);
                double projected_dist = euclidean_distance(Y[i], Y[j]);

                // Avoid division by zero
                if (original_dist > 1e-10)
                {
                    double relative_error = std::abs(projected_dist - original_dist) / original_dist;
                    total_relative_error += relative_error;
                    count++;
                }
            }
        }

        return count > 0 ? total_relative_error / count : 0.0;
    }
}

// Dense JLT implementation
class DenseJLT
{
private:
    std::vector<std::vector<double>> R_dense;
    double scaling_factor;
    std::mutex gen_mutex;

public:
    // Parallelized dense matrix generation
    DenseJLT(int d, int k) : R_dense(d, std::vector<double>(k))
    {
        // Calculate scaling factor
        scaling_factor = 1.0 / std::sqrt(k);

        // Parallel matrix generation
        std::vector<std::future<void>> futures;
        int num_threads = std::thread::hardware_concurrency();

        for (int thread = 0; thread < num_threads; ++thread)
        {
            futures.push_back(std::async(std::launch::async, [this, d, k, thread, num_threads]()
                                         {
                // Create thread-local random generator
                std::random_device rd;
                std::mt19937 gen(rd() + thread);
                std::normal_distribution<> dist(0, 1);

                // Divide work across threads
                int start_col = thread * (k / num_threads);
                int end_col = (thread == num_threads - 1) ? k : (thread + 1) * (k / num_threads);

                for (int i = 0; i < d; ++i) {
                    for (int j = start_col; j < end_col; ++j) {
                        // Generate random values with proper scaling
                        this->R_dense[i][j] = dist(gen) * scaling_factor;
                    }
                } }));
        }

        // Wait for all threads to complete
        for (auto &future : futures)
        {
            future.wait();
        }
    }

    // Parallel dense matrix transformation
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &X)
    {
        int n = X.size();
        int k = R_dense[0].size();
        std::vector<std::vector<double>> Y(n, std::vector<double>(k, 0.0));

        // Parallel transformation
        std::vector<std::future<void>> futures;
        int num_threads = std::thread::hardware_concurrency();

        for (int thread = 0; thread < num_threads; ++thread)
        {
            futures.push_back(std::async(std::launch::async, [&](int thread_id)
                                         {
                int start = thread_id * (n / num_threads);
                int end = (thread_id == num_threads - 1) 
                          ? n 
                          : (thread_id + 1) * (n / num_threads);

                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < k; ++j) {
                        double sum = 0.0;
                        for (int l = 0; l < int(R_dense.size()); ++l) {
                            sum += X[i][l] * R_dense[l][j];
                        }
                        Y[i][j] = sum;
                    }
                } }, thread));
        }

        // Wait for all threads to complete
        for (auto &future : futures)
        {
            future.wait();
        }

        return Y;
    }
};

// Sparse JLT implementation
class SparseJLT
{
private:
    struct SparseMatrix
    {
        int rows;
        int cols;
        std::vector<std::vector<std::pair<int, double>>> data;
    };

    SparseMatrix R; // Projection matrix
    double scaling_factor;
    std::mutex gen_mutex;

public:
    // Initialize a sparse random projection matrix
    SparseJLT(int d, int k, int s = -1)
    {
        // If sparsity parameter not provided, set based on dimension
        if (s == -1)
        {
            // Use log(d) as recommended by literature for good balance of speed and accuracy
            s = std::max(static_cast<int>(std::log(d)), 3);
        }

        if (s > d)
        {
            s = d;
            std::cout << "Warning: Sparsity parameter s was larger than d, setting s=d" << std::endl;
        }

        std::cout << "Using sparsity parameter s = " << s << std::endl;

        // Initialize the sparse matrix
        R.rows = d;
        R.cols = k;
        R.data.resize(k);

        // Calculate proper scaling factor for sparse JLT
        scaling_factor = std::sqrt(static_cast<double>(d) / (s * k));

        std::cout << "Using scaling factor: " << scaling_factor << std::endl;

        // Parallel column generation
        std::vector<std::future<std::vector<std::pair<int, double>>>> futures;
        for (int j = 0; j < k; ++j)
        {
            futures.push_back(std::async(std::launch::async, [this, d, s]()
                                         {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> row_dist(0, d - 1);
                std::uniform_int_distribution<> sign_dist(0, 1);

                // Generate s distinct random row indices
                std::unordered_set<int> row_indices;
                std::vector<std::pair<int, double>> column_data;

                {
                    std::lock_guard<std::mutex> lock(gen_mutex);
                    while (row_indices.size() < static_cast<size_t>(s))
                    {
                        row_indices.insert(row_dist(gen));
                    }
                }

                // Assign random +/- scaling_factor values
                for (int row_idx : row_indices)
                {
                    int sign;
                    {
                        std::lock_guard<std::mutex> lock(gen_mutex);
                        sign = sign_dist(gen);
                    }
                    double value = (sign == 0 ? -1.0 : 1.0) * scaling_factor;
                    column_data.push_back(std::make_pair(row_idx, value));
                }

                // Sort by row index for faster multiplication
                std::sort(column_data.begin(), column_data.end(),
                          [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                          {
                              return a.first < b.first;
                          });

                return column_data; }));
        }

        // Collect results
        for (size_t j = 0; j < futures.size(); ++j)
        {
            R.data[j] = futures[j].get();
        }
    }

    // Parallel matrix transformation
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &X)
    {
        int n = X.size();
        int k = R.cols;

        // Initialize output matrix Y
        std::vector<std::vector<double>> Y(n, std::vector<double>(k, 0.0));

        // Parallel transformation
        std::vector<std::future<void>> futures;
        int num_threads = std::thread::hardware_concurrency();

        for (int thread = 0; thread < num_threads; ++thread)
        {
            futures.push_back(std::async(std::launch::async, [&](int thread_id)
                                         {
                int start = thread_id * (n / num_threads);
                int end = (thread_id == num_threads - 1) 
                          ? n 
                          : (thread_id + 1) * (n / num_threads);

                for (int i = start; i < end; ++i)
                {
                    for (int j = 0; j < k; ++j)
                    {
                        double sum = 0.0;
                        for (const auto &entry : R.data[j])
                        {
                            int row_idx = entry.first;
                            double value = entry.second;
                            sum += X[i][row_idx] * value;
                        }
                        Y[i][j] = sum;
                    }
                } }, thread));
        }

        // Wait for all threads to complete
        for (auto &future : futures)
        {
            future.wait();
        }

        return Y;
    }
};

// Function to demonstrate the theoretical guarantees
void theoretical_guarantees(int d, double epsilon = 0.1, double delta = 0.05)
{
    std::cout << "\nTheoretical Guarantees of JL Transform:" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // Calculate required dimension based on JL lemma
    int k_jl = static_cast<int>(std::ceil(4 * std::log(100) / (epsilon * epsilon * 0.5)));

    std::cout << "For preserving distances between 100 points with:" << std::endl;
    std::cout << "- Original dimension d = " << d << std::endl;
    std::cout << "- Distortion bound ε = " << epsilon << std::endl;
    std::cout << "- Success probability = " << (1 - delta) << std::endl;
    std::cout << "Required target dimension k ≥ " << k_jl << std::endl;
}

// Parse command-line arguments
ProgramConfig parse_arguments(int argc, char *argv[])
{
    ProgramConfig config;
    int opt;

    struct option long_options[] = {
        {"vectors", required_argument, 0, 'n'},
        {"original-dim", required_argument, 0, 'd'},
        {"target-dim", required_argument, 0, 'k'},
        {"no-dense-compare", no_argument, 0, 'N'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "n:d:k:Nvh", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'n':
            config.n = std::stoi(optarg);
            break;
        case 'd':
            config.d = std::stoi(optarg);
            break;
        case 'k':
            config.k = std::stoi(optarg);
            break;
        case 'N':
            config.compare_dense = false;
            break;
        case 'v':
            config.verbose = true;
            break;
        case 'h':
            std::cout << "Usage: " << argv[0]
                      << " [-n vectors] [-d original-dim] [-k target-dim] [-N] [-v] [-h]" << std::endl;
            std::cout << "  -n, --vectors       Number of vectors (default: " << config.n << ")" << std::endl;
            std::cout << "  -d, --original-dim  Original dimension (default: " << config.d << ")" << std::endl;
            std::cout << "  -k, --target-dim    Target dimension (default: " << config.k << ")" << std::endl;
            std::cout << "  -N, --no-dense-compare  Skip dense JLT comparison" << std::endl;
            std::cout << "  -v, --verbose       Verbose output" << std::endl;
            std::cout << "  -h, --help          Display this help message" << std::endl;
            exit(0);
        default:
            std::cerr << "Usage: " << argv[0]
                      << " [-n vectors] [-d original-dim] [-k target-dim] [-N] [-v] [-h]" << std::endl;
            exit(1);
        }
    }

    return config;
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    ProgramConfig config = parse_arguments(argc, argv);

    if (config.verbose)
    {
        std::cout << "Configuration:" << std::endl;
        std::cout << "- Vectors: " << config.n << std::endl;
        std::cout << "- Original dimension: " << config.d << std::endl;
        std::cout << "- Target dimension: " << config.k << std::endl;
        std::cout << "- Compare dense JLT: " << (config.compare_dense ? "Yes" : "No") << std::endl;
    }

    std::cout << "Number of concurrent threads supported: "
              << std::thread::hardware_concurrency() << std::endl;

    std::cout << "Generating " << config.n << " random points in " << config.d << " dimensions..." << std::endl;
    auto X = utils::generate_random_data(config.n, config.d);

    // Dense JLT comparison (optional)
    if (config.compare_dense)
    {
        std::cout << "Starting dense JLT, converting to target dimension k = " << config.k << "..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Create and transform using dense JLT
        DenseJLT dense_jlt(config.d, config.k);
        auto Y_dense = dense_jlt.transform(X);

        auto dense_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count() /
                              1000.0;

        std::cout << "Dense JLT transformation time: " << dense_duration << " seconds" << std::endl;

        // Accuracy analysis for dense JLT
        int num_samples = config.n;
        double dense_error = utils::calculate_relative_error(X, Y_dense, num_samples);
        std::cout << "Dense JLT avg relative error: " << dense_error << std::endl;
    }

    // Sparse JLT
    std::cout << "Starting sparse JLT, converting to target dimension k = " << config.k << "..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create sparse JLT with automatic sparsity parameter
    SparseJLT sjlt(config.d, config.k);
    auto Y_sparse = sjlt.transform(X);

    auto sparse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::high_resolution_clock::now() - start_time)
                               .count() /
                           1000.0;

    std::cout << "Sparse JLT transformation time: " << sparse_duration << " seconds" << std::endl;

    // Accuracy analysis for sparse JLT
    int num_samples = config.n;
    double sparse_error = utils::calculate_relative_error(X, Y_sparse, num_samples);
    std::cout << "Sparse JLT avg relative error: " << sparse_error << std::endl;

    // 2. Detailed error analysis
    std::vector<double> error_ratios;
    std::mt19937 gen(42); // Use fixed seed for reproducibility
    std::uniform_int_distribution<> dist(0, config.n - 1);
    std::vector<std::vector<double>> examples;
    int num_detail_samples = config.n;
    int num_examples = 10;

    for (int s = 0; s < num_detail_samples; ++s)
    {
        int i = dist(gen);
        int j = dist(gen);
        if (i != j)
        {
            double original_dist = utils::euclidean_distance(X[i], X[j]);
            double projected_dist = utils::euclidean_distance(Y_sparse[i], Y_sparse[j]);

            if (original_dist > 1e-10)
            {
                double ratio = projected_dist / original_dist;
                error_ratios.push_back(ratio);

                if (s < num_examples)
                {
                    examples.push_back({original_dist, projected_dist, ratio});
                }
            }
        }
    }

    // Calculate mean and standard deviation of ratios
    if (!error_ratios.empty())
    {
        double sum = 0.0;
        for (double ratio : error_ratios)
        {
            sum += ratio;
        }
        double mean = sum / error_ratios.size();

        double sum_sq_diff = 0.0;
        for (double ratio : error_ratios)
        {
            double diff = ratio - mean;
            sum_sq_diff += diff * diff;
        }
        double std_dev = std::sqrt(sum_sq_diff / error_ratios.size());

        std::cout << "\nDistance ratio statistics:" << std::endl;
        std::cout << "Mean ratio: " << mean << " (ideally close to 1.0)" << std::endl;
        std::cout << "Standard deviation: " << std_dev << std::endl;
    }

    // Print first few examples
    std::cout << "\nExamples (" << num_examples << "):" << std::endl;
    std::cout << "----------------------" << std::endl;
    for (int s = 0; s < num_examples; ++s)
    {
        std::cout << "Sample " << s << ": Original dist = " << examples[s][0]
                  << ", Sparse projected dist = " << examples[s][1]
                  << ", Ratio = " << examples[s][2] << std::endl;
    }

    return 0;
}