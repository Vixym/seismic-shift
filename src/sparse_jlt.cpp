#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>

/*
    Sparse JLT for dimensionality reduction
    Implementation references this paper: https://arxiv.org/pdf/1012.1577
*/
class SparseJLT
{
private:
    struct SparseMatrix
    {
        int rows;
        int cols;
        std::vector<std::vector<std::pair<int, double>>> data; // For each column, store (row_idx, value) pairs
    };

    SparseMatrix R; // Projection matrix
    double scaling_factor;

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

        // Random number generation with fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> row_dist(0, d - 1);
        std::uniform_int_distribution<> sign_dist(0, 1);

        // Calculate proper scaling factor for sparse JLT
        // This ensures distance preservation (JL lemma)
        scaling_factor = std::sqrt(static_cast<double>(d) / (s * k));

        std::cout << "Using scaling factor: " << scaling_factor << std::endl;

        for (int j = 0; j < k; ++j)
        {
            // Generate s distinct random row indices
            std::unordered_set<int> row_indices;
            while (row_indices.size() < static_cast<size_t>(s))
            {
                row_indices.insert(row_dist(gen));
            }

            // Assign random +/- scaling_factor values
            for (int row_idx : row_indices)
            {
                double value = (sign_dist(gen) == 0 ? -1.0 : 1.0) * scaling_factor;
                R.data[j].push_back(std::make_pair(row_idx, value));
            }

            // Sort by row index for faster multiplication
            std::sort(R.data[j].begin(), R.data[j].end(),
                      [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                      {
                          return a.first < b.first;
                      });
        }
    }

    // Apply the sparse JLT to input data X
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &X)
    {
        int n = X.size();
        int k = R.cols;

        // Initialize output matrix Y
        std::vector<std::vector<double>> Y(n, std::vector<double>(k, 0.0));

        // Perform sparse matrix multiplication more efficiently
        for (int i = 0; i < n; ++i)
        {
            // For each data point
            for (int j = 0; j < k; ++j)
            {
                // For each output dimension
                double sum = 0.0;
                for (const auto &entry : R.data[j])
                {
                    int row_idx = entry.first;
                    double value = entry.second;
                    sum += X[i][row_idx] * value;
                }
                Y[i][j] = sum;
            }
        }

        return Y;
    }

    // Variant that allows in-place transformation to save memory
    void transform_inplace(const std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &Y)
    {
        int n = X.size();
        int k = R.cols;

        // Resize Y if needed
        if (Y.size() != n || Y[0].size() != k)
        {
            Y.resize(n, std::vector<double>(k, 0.0));
        }
        else
        {
            // Clear existing values
            for (int i = 0; i < n; ++i)
            {
                std::fill(Y[i].begin(), Y[i].end(), 0.0);
            }
        }

        // Perform sparse matrix multiplication
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                for (const auto &entry : R.data[j])
                {
                    int row_idx = entry.first;
                    double value = entry.second;
                    Y[i][j] += X[i][row_idx] * value;
                }
            }
        }
    }
};

// Utility functions for testing
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
        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0, 1);

        std::vector<std::vector<double>> X(n, std::vector<double>(d));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < d; ++j)
            {
                X[i][j] = dist(gen);
            }
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

// Example function to demonstrate usage
void demo_sparse_jlt()
{
    int n = 1000;    // number of data points
    int d = 1000000; // original dimensionality
    int k = 100;     // target dimensionality

    std::cout << "Generating " << n << " random points in " << d << " dimensions..." << std::endl;
    auto X = utils::generate_random_data(n, d);

    // Standard random projection (dense)
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create dense random matrix with proper scaling
    std::vector<std::vector<double>> R_dense(d, std::vector<double>(k));
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, 1);
        double scaling = 1.0 / std::sqrt(k);

        for (int i = 0; i < d; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                R_dense[i][j] = dist(gen) * scaling;
            }
        }
    }

    // Dense matrix multiplication
    std::vector<std::vector<double>> Y_dense(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            double sum = 0.0;
            for (int l = 0; l < d; ++l)
            {
                sum += X[i][l] * R_dense[l][j];
            }
            Y_dense[i][j] = sum;
        }
    }

    auto dense_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count() /
                          1000.0;

    // Sparse JLT
    start_time = std::chrono::high_resolution_clock::now();

    // Create sparse JLT with automatic sparsity parameter
    SparseJLT sjlt(d, k);
    auto Y_sparse = sjlt.transform(X);

    auto sparse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::high_resolution_clock::now() - start_time)
                               .count() /
                           1000.0;

    // Calculate and report metrics
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Standard JLT time: " << dense_duration << " seconds" << std::endl;
    std::cout << "Sparse JLT time: " << sparse_duration << " seconds" << std::endl;
    std::cout << "Speedup: " << dense_duration / sparse_duration << "x" << std::endl;

    int num_samples = 1000; // Increase sample size for better accuracy
    double dense_error = utils::calculate_relative_error(X, Y_dense, num_samples);
    double sparse_error = utils::calculate_relative_error(X, Y_sparse, num_samples);

    std::cout << "\nAccuracy comparison:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Standard JLT avg relative error: " << dense_error << std::endl;
    std::cout << "Sparse JLT avg relative error: " << sparse_error << std::endl;
    std::cout << "Error ratio (sparse/dense): " << sparse_error / (dense_error > 0 ? dense_error : 1.0) << std::endl;

    // Error analysis
    std::cout << "\nDetailed error analysis:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::vector<double> error_ratios;
    int num_detail_samples = 50;
    std::mt19937 gen(42); // Use fixed seed for reproducibility
    std::uniform_int_distribution<> dist(0, n - 1);

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

                if (s < 10)
                { // Print first 10 samples
                    std::cout << "Sample " << s << ": Original dist = " << original_dist
                              << ", Projected dist = " << projected_dist
                              << ", Ratio = " << ratio << std::endl;
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
}

// Function to demonstrate the theoretical guarantees
void theoretical_guarantees()
{
    std::cout << "\nTheoretical Guarantees of JL Transform:" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    int d = 10000;        // original dimension
    double epsilon = 0.1; // desired accuracy (distortion factor)
    double delta = 0.05;  // failure probability

    // Calculate required dimension based on JL lemma
    int k_jl = static_cast<int>(std::ceil(4 * std::log(100) / (epsilon * epsilon * 0.5)));

    std::cout << "For preserving distances between 100 points with:" << std::endl;
    std::cout << "- Original dimension d = " << d << std::endl;
    std::cout << "- Distortion bound ε = " << epsilon << std::endl;
    std::cout << "- Success probability = " << (1 - delta) << std::endl;
    std::cout << "Required target dimension k ≥ " << k_jl << std::endl;
}

int main()
{
    // Run the demo
    demo_sparse_jlt();

    // Show theoretical guarantees
    theoretical_guarantees();

    return 0;
}