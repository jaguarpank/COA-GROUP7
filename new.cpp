#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

using namespace std;

const int MAX_CITIES = 10; // Maximum number of cities
const int MAX_THREADS = 4; // Maximum number of threads

// Function to calculate total distance of a tour using a distance matrix
double tourDistance(const vector<int>& tour, const vector<vector<double>>& distMatrix) {
    double dist = 0.0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        int city1 = tour[i], city2 = tour[i + 1];
        dist += distMatrix[city1][city2];
    }
    // Return to the starting city
    int lastCity = tour.back();
    dist += distMatrix[lastCity][tour.front()];
    return dist;
}

// Particle structure representing a potential solution to the TSP
struct Particle {
    vector<int> tour; // Permutation of cities
    double fitness;   // Total distance traveled in the tour

    Particle(int numCities) : tour(numCities), fitness(0.0) {
        // Initialize the tour with a random permutation of cities
        for (int i = 0; i < numCities; ++i)
            tour[i] = i;
        random_shuffle(tour.begin(), tour.end());
    }
};

// Parallel PSO algorithm to solve the TSP with distance matrix
vector<int> parallelPSO_TSP(int numCities, int numParticles, int numIterations, const vector<vector<double>>& distMatrix) {
    vector<Particle> particles(numParticles, Particle(numCities));
    vector<int> globalBestTour;
    double globalBestFitness = numeric_limits<double>::max();

    // Parallel PSO execution
    #pragma omp parallel for num_threads(MAX_THREADS)
    for (int i = 0; i < numParticles; ++i) {
        for (int iter = 0; iter < numIterations; ++iter) {
            // Evaluate fitness
            particles[i].fitness = tourDistance(particles[i].tour, distMatrix);

            // Update global best
            #pragma omp critical
            {
                if (particles[i].fitness < globalBestFitness) {
                    globalBestFitness = particles[i].fitness;
                    globalBestTour = particles[i].tour;
                }
            }

            // Update velocity and position (not shown for brevity)
        }
    }

    return globalBestTour;
}

int main() {
    // Example setup with distance matrix
    int numCities = 5;      
    int numParticles = 25;
    int numIterations = 1000;
    vector<vector<double>> distanceMatrix = {
        {0.0, 3.0, 4.0, 2.0, 7.0},
        {3.0, 0.0, 4.0, 6.0, 3.0},
        {4.0, 4.0, 0.0, 5.0, 8.0},
        {2.0, 6.0, 5.0, 0.0, 6.0},
        {7.0, 3.0, 8.0, 6.0, 0.0}
    };

    // Solve the TSP using parallel PSO
    vector<int> bestTour = parallelPSO_TSP(numCities, numParticles, numIterations, distanceMatrix);

    // Output the best tour found
    for (int i = 0; i < bestTour.size(); ++i) {
        if (bestTour[i] == 1) {
            for (int j = i; j < bestTour.size(); ++j) {
                cout << bestTour[j] << endl;
            }
            for (int j = 0; j < i; ++j) {
                cout << bestTour[j] << endl;
            }
            break;
        }
    }

    return 0;
}
