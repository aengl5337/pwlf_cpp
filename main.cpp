#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <Eigen/Dense>

/*
Piecewise Linear Fit using Differential Evolution and Variable Projection
Authors: Alec Engl, Google Gemini
Description:
This code fits a piecewise linear function to given data points (x, y)
using Differential Evolution (DE) for optimizing the breakpoints and
Variable Projection for efficiently solving the linear parameters.
The piecewise linear function is represented in a truncated power basis,
which allows for straightforward construction of the design matrix.

It takes as inspiration the methodology used in the Python package `pwlf` by Charles Jekel.
(Source: https://github.com/cjekel/piecewise_linear_fit_py)


Usage:
1. Prepare your data points in vectors `x` and `y`.
2. Create an instance of `PiecewiseLinearFit` with the data.
3. Call the `fit` method with the desired number of segments.
4. Retrieve the breakpoints and coefficients after fitting.

*/


// --- Configuration ---
// Dependencies: Eigen 3 (Header only)
// Compilation: g++ -O3 -I /usr/include/eigen3 pwlf.cpp -o pwlf

using namespace Eigen;
using std::vector;

class PiecewiseLinearFit {
public:
    PiecewiseLinearFit(const vector<double>& x, const vector<double>& y) 
        : x_data(x), y_data(y) {
        
        // Map std::vector to Eigen Vectors for math operations
        n_data = x.size();
        eig_x = Map<const VectorXd>(x_data.data(), n_data);
        eig_y = Map<const VectorXd>(y_data.data(), n_data);
        
        x_min = eig_x.minCoeff();
        x_max = eig_x.maxCoeff();
    }

    // Main fitting function
    // n_segments: Number of linear segments to fit
    // pop_size: Population size for Differential Evolution (default 50)
    // max_iter: Maximum generations (default 1000)
    vector<double> fit(int n_segments, int pop_size = 50, int max_iter = 1000) {
        if (n_segments < 1) return {};
        
        // We optimize n_segments - 1 internal breakpoints.
        // The first and last breakpoints are fixed at x_min and x_max.
        int n_internal_breaks = n_segments - 1;
        
        if (n_internal_breaks == 0) {
            // Simple linear regression case
            best_ssr = cost_function({}); 
            return {x_min, x_max};
        }

        // --- Differential Evolution Setup ---
        // Bounds: All breakpoints must be between [x_min, x_max]
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(x_min, x_max);
        std::uniform_real_distribution<> rand01(0.0, 1.0);

        // Initialize Population (pop_size x n_internal_breaks)
        vector<VectorXd> population(pop_size);
        vector<double> fitness(pop_size);

        for (int i = 0; i < pop_size; ++i) {
            population[i].resize(n_internal_breaks);
            for (int j = 0; j < n_internal_breaks; ++j) {
                population[i][j] = dis(gen);
            }
            // Sort initial genes to make them valid breakpoints
            std::sort(population[i].data(), population[i].data() + population[i].size());
            fitness[i] = cost_function(population[i]);
        }

        // DE Parameters
        double F = 0.5; // Mutation factor
        double CR = 0.7; // Crossover probability
        
        // Best solution tracking
        int best_idx = 0;
        double best_score = fitness[0];
        for(int i=1; i<pop_size; ++i) {
            if(fitness[i] < best_score) {
                best_score = fitness[i];
                best_idx = i;
            }
        }

        // --- DE Optimization Loop ---
        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < pop_size; ++i) {
                // 1. Mutation: Select 3 random distinct vectors (a, b, c)
                // Using "rand/1/bin" strategy (classic DE)
                int a, b, c;
                do { a = std::rand() % pop_size; } while (a == i);
                do { b = std::rand() % pop_size; } while (b == i || b == a);
                do { c = std::rand() % pop_size; } while (c == i || c == a || c == b);

                // Create Mutant Vector: v = a + F * (b - c)
                VectorXd mutant = population[a] + F * (population[b] - population[c]);

                // 2. Crossover: Mix target(i) and mutant
                VectorXd trial = population[i];
                int j_rand = std::rand() % n_internal_breaks; // Ensure at least one parameter changes
                for (int j = 0; j < n_internal_breaks; ++j) {
                    if (rand01(gen) < CR || j == j_rand) {
                        // Clamp to bounds
                        trial[j] = std::max(x_min, std::min(x_max, mutant[j]));
                    }
                }
                
                // Sort trial breakpoints (pwlf requirement)
                std::sort(trial.data(), trial.data() + trial.size());

                // 3. Selection: Evaluate Fitness
                double trial_score = cost_function(trial);

                if (trial_score <= fitness[i]) {
                    population[i] = trial;
                    fitness[i] = trial_score;
                    if (trial_score < best_score) {
                        best_score = trial_score;
                        best_idx = i;
                    }
                }
            }
        }

        // Store final best parameters
        best_ssr = best_score;
        final_internal_breaks = population[best_idx];

        // Construct full breakpoint list
        vector<double> result;
        result.push_back(x_min);
        for(auto v : final_internal_breaks) result.push_back(v);
        result.push_back(x_max);
        
        // Solve one last time to store coefficients (slopes/intercepts)
        solve_coefficients(final_internal_breaks);
        
        return result;
    }

    // Accessors for results
    double get_ssr() const { return best_ssr; }
    VectorXd get_coefficients() const { return beta; }

private:
    vector<double> x_data, y_data;
    VectorXd eig_x, eig_y;
    double x_min, x_max;
    int n_data;

    double best_ssr;
    VectorXd final_internal_breaks;
    VectorXd beta; // Fitted parameters (intercept, slopes...)

    // The cost function: SSE for a given set of breakpoints
    double cost_function(const VectorXd& internal_breaks) {
        MatrixXd A = assemble_matrix(internal_breaks);
        
        // Solve Ax = b using Least Squares (Pivoted QR is robust)
        // This is the "Variable Projection" step
        VectorXd beta_local = A.colPivHouseholderQr().solve(eig_y);
        
        VectorXd y_pred = A * beta_local;
        double ssr = (eig_y - y_pred).squaredNorm();
        return ssr;
    }

    // Stores the final Beta parameters
    void solve_coefficients(const VectorXd& internal_breaks) {
         MatrixXd A = assemble_matrix(internal_breaks);
         beta = A.colPivHouseholderQr().solve(eig_y);
    }

    // Build the Truncated Power Basis Matrix
    MatrixXd assemble_matrix(const VectorXd& internal_breaks) {
        int n_breaks = internal_breaks.size();
        int n_params = n_breaks + 2; // Intercept + (n_breaks+1) segments
        
        MatrixXd A(n_data, n_params);

        // Column 0: Intercept (Ones)
        A.col(0) = VectorXd::Ones(n_data);

        // Column 1: First Line (x - t0)
        A.col(1) = eig_x.array() - x_min;

        // Columns 2..N: Truncated Power Basis (x - t_i)_+
        for (int i = 0; i < n_breaks; ++i) {
            double break_pt = internal_breaks[i];
            // element-wise max(0, x - break_pt)
            A.col(i + 2) = (eig_x.array() - break_pt).cwiseMax(0.0);
        }
        return A;
    }
};

int main() {
    // --- Example Usage ---
    
    // 1. Generate Dummy Data (Sin Wave)
    int N = 100;
    vector<double> x(N), y(N);
    for(int i=0; i<N; ++i) {
        x[i] = (double)i / (N-1) * 10.0; // x from 0 to 10
        y[i] = std::sin(x[i] * 0.5) * x[i]; // Non-linear shape
    }

    // 2. Initialize Fitter
    PiecewiseLinearFit fitter(x, y);

    // 3. Fit 3 Segments
    std::cout << "Running Differential Evolution..." << std::endl;
    vector<double> breaks = fitter.fit(3);

    // 4. Output Results
    std::cout << "Found Breakpoints: ";
    for (double b : breaks) std::cout << b << " ";
    std::cout << "\nSum of Squared Residuals: " << fitter.get_ssr() << std::endl;
    
    // Coefficients corresponds to [Beta1, Beta2, Beta3...]
    // f(x) = Beta1 + Beta2*(x-t0) + Beta3*(x-t1)_+ ...
    std::cout << "Coefficients: \n" << fitter.get_coefficients() << std::endl;

    return 0;
}