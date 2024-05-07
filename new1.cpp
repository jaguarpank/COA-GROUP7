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
    int numCities = 15;      
    int numParticles = 225;
    int numIterations = 1000;
    vector<vector<double>> distanceMatrix = {
                {0.0, 29.0, 82.0, 46.0, 68.0, 52.0, 72.0, 42.0, 51.0, 55.0, 29.0, 74.0, 23.0, 72.0, 46.0},
        {29.0, 0.0, 55.0, 46.0, 42.0, 43.0, 43.0, 23.0, 23.0, 31.0, 41.0, 51.0, 11.0, 52.0, 21.0},
        {82.0, 55.0, 0.0, 68.0, 46.0, 55.0, 23.0, 43.0, 41.0, 29.0, 79.0, 21.0, 64.0, 31.0, 51.0},
        {46.0, 46.0, 68.0, 0.0, 82.0, 15.0, 72.0, 31.0, 62.0, 42.0, 21.0, 51.0, 51.0, 43.0, 64.0},
        {68.0, 42.0, 46.0, 82.0, 0.0, 74.0, 23.0, 52.0, 21.0, 46.0, 82.0, 58.0, 46.0, 65.0, 23.0},
        {52.0, 43.0, 55.0, 15.0, 74.0, 0.0, 61.0, 23.0, 55.0, 31.0, 33.0, 37.0, 51.0, 29.0, 59.0},
        {72.0, 43.0, 23.0, 72.0, 23.0, 61.0, 0.0, 42.0, 23.0, 31.0, 77.0, 37.0, 51.0, 46.0, 33.0},
        {42.0, 23.0, 43.0, 31.0, 52.0, 23.0, 42.0, 0.0, 33.0, 15.0, 37.0, 33.0, 33.0, 31.0, 37.0},
        {51.0, 23.0, 41.0, 62.0, 21.0, 55.0, 23.0, 33.0, 0.0, 29.0, 62.0, 46.0, 29.0, 51.0, 11.0},
        {55.0, 31.0, 29.0, 42.0, 46.0, 31.0, 31.0, 15.0, 29.0, 0.0, 51.0, 21.0, 41.0, 23.0, 37.0},
        {29.0, 41.0, 79.0, 21.0, 82.0, 33.0, 77.0, 37.0, 62.0, 51.0, 0.0, 65.0, 42.0, 59.0, 61.0},
        {74.0, 51.0, 21.0, 51.0, 58.0, 37.0, 37.0, 33.0, 46.0, 21.0, 65.0, 0.0, 61.0, 11.0, 55.0},
        {23.0, 11.0, 64.0, 51.0, 46.0, 51.0, 51.0, 33.0, 29.0, 41.0, 42.0, 61.0, 0.0, 62.0, 23.0},
        {72.0, 52.0, 31.0, 43.0, 65.0, 29.0, 46.0, 31.0, 51.0, 23.0, 59.0, 11.0, 62.0, 0.0, 59.0},
        {46.0, 21.0, 51.0, 64.0, 23.0, 59.0, 33.0, 37.0, 11.0, 37.0, 61.0, 55.0, 23.0, 59.0, 0.0}
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
