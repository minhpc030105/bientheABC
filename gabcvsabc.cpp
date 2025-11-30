#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // rand, srand
#include <ctime>
#include <limits>
#include <iomanip>
using namespace std;

// ============ Simple random functions (no mt19937) ============
double rand01() {
    return (double)rand() / (double)RAND_MAX;          // [0,1)
}

double randUniform(double a, double b) {
    return a + (b - a) * rand01();                     // [a,b]
}

int randInt(int l, int r) {                            // [l,r]
    return l + rand() % (r - l + 1);
}

// ============ Problem: f(x,y) = (x-2)^2 + y^2 ============
double objective(const vector<double>& x) {
    double x1 = x[0];
    double x2 = x[1];
    return (x1 - 2.0) * (x1 - 2.0) + x2 * x2;
}

// ============ Fitness ============
double calcFitness(double cost) {
    if (cost >= 0.0) return 1.0 / (1.0 + cost);
    return 1.0 + fabs(cost);
}

// ============ Food source ============
struct FoodSource {
    vector<double> x;   // position
    double cost;        // objective value
    double fit;         // fitness
    int trial;          // number of consecutive failures

    FoodSource() {}
    FoodSource(int D) {
        x.assign(D, 0.0);
        cost = numeric_limits<double>::infinity();
        fit = 0.0;
        trial = 0;
    }
};

// ============ Standard ABC ============
FoodSource runABC(int D, double lower, double upper,
                  int SN, int limit, int maxCycle,
                  long long& evalCount)
{
    vector<FoodSource> foods(SN, FoodSource(D));

    // --- initialization ---
    for (int i = 0; i < SN; ++i) {
        for (int j = 0; j < D; ++j)
            foods[i].x[j] = randUniform(lower, upper);
        foods[i].cost = objective(foods[i].x);
        evalCount++;
        foods[i].fit  = calcFitness(foods[i].cost);
        foods[i].trial = 0;
    }

    FoodSource best = foods[0];
    for (int i = 1; i < SN; ++i)
        if (foods[i].cost < best.cost) best = foods[i];

    // --- main loop ---
    for (int cycle = 0; cycle < maxCycle; ++cycle) {

        // 1. Employed bees
        for (int i = 0; i < SN; ++i) {
            int k;
            do { k = randInt(0, SN - 1); } while (k == i);
            int j = randInt(0, D - 1);

            vector<double> v = foods[i].x;
            double phi = randUniform(-1.0, 1.0);

            v[j] = foods[i].x[j] + phi * (foods[i].x[j] - foods[k].x[j]);

            if (v[j] < lower) v[j] = lower;
            if (v[j] > upper) v[j] = upper;

            double newCost = objective(v);
            evalCount++;

            if (newCost < foods[i].cost) {
                foods[i].x = v;
                foods[i].cost = newCost;
                foods[i].fit = calcFitness(newCost);
                foods[i].trial = 0;
            } else {
                foods[i].trial++;
            }
        }

        // 2. Onlooker bees
        vector<double> prob(SN);
        double sumFit = 0.0;
        for (int i = 0; i < SN; ++i) {
            prob[i] = foods[i].fit;
            sumFit += foods[i].fit;
        }
        if (sumFit == 0.0) sumFit = 1.0;
        for (int i = 0; i < SN; ++i)
            prob[i] /= sumFit;

        int t = 0;
        int i = 0;
        while (t < SN) {
            double r = rand01();
            if (r < prob[i]) {
                t++;

                int k;
                do { k = randInt(0, SN - 1); } while (k == i);
                int j = randInt(0, D - 1);

                vector<double> v = foods[i].x;
                double phi = randUniform(-1.0, 1.0);

                v[j] = foods[i].x[j] + phi * (foods[i].x[j] - foods[k].x[j]);

                if (v[j] < lower) v[j] = lower;
                if (v[j] > upper) v[j] = upper;

                double newCost = objective(v);
                evalCount++;

                if (newCost < foods[i].cost) {
                    foods[i].x = v;
                    foods[i].cost = newCost;
                    foods[i].fit = calcFitness(newCost);
                    foods[i].trial = 0;
                } else {
                    foods[i].trial++;
                }
            }
            i++;
            if (i == SN) i = 0;
        }

        // 3. Scout bee
        int worst = 0;
        for (int idx = 1; idx < SN; ++idx)
            if (foods[idx].trial > foods[worst].trial)
                worst = idx;

        if (foods[worst].trial > limit) {
            for (int j = 0; j < D; ++j)
                foods[worst].x[j] = randUniform(lower, upper);
            foods[worst].cost = objective(foods[worst].x);
            evalCount++;
            foods[worst].fit = calcFitness(foods[worst].cost);
            foods[worst].trial = 0;
        }

        // update best
        for (int idx = 0; idx < SN; ++idx)
            if (foods[idx].cost < best.cost)
                best = foods[idx];
    }

    return best;
}

// ============ Gbest-guided ABC (GABC) ============
FoodSource runGABC(int D, double lower, double upper,
                   int SN, int limit, int maxCycle,
                   long long& evalCount,
                   double c_gbest = 1.5)
{
    vector<FoodSource> foods(SN, FoodSource(D));

    // --- initialization ---
    for (int i = 0; i < SN; ++i) {
        for (int j = 0; j < D; ++j)
            foods[i].x[j] = randUniform(lower, upper);
        foods[i].cost = objective(foods[i].x);
        evalCount++;
        foods[i].fit  = calcFitness(foods[i].cost);
        foods[i].trial = 0;
    }

    FoodSource gbest = foods[0];
    for (int i = 1; i < SN; ++i)
        if (foods[i].cost < gbest.cost) gbest = foods[i];

    // --- main loop ---
    for (int cycle = 0; cycle < maxCycle; ++cycle) {

        // 1. Employed bees (with gbest term)
        for (int i = 0; i < SN; ++i) {
            int k;
            do { k = randInt(0, SN - 1); } while (k == i);
            int j = randInt(0, D - 1);

            vector<double> v = foods[i].x;
            double phi = randUniform(-1.0, 1.0);

            v[j] = foods[i].x[j]
                 + phi * (foods[i].x[j] - foods[k].x[j])
                 + c_gbest * (gbest.x[j] - foods[i].x[j]);

            if (v[j] < lower) v[j] = lower;
            if (v[j] > upper) v[j] = upper;

            double newCost = objective(v);
            evalCount++;

            if (newCost < foods[i].cost) {
                foods[i].x = v;
                foods[i].cost = newCost;
                foods[i].fit = calcFitness(newCost);
                foods[i].trial = 0;
            } else {
                foods[i].trial++;
            }
        }

        // 2. Onlooker bees (with gbest term)
        vector<double> prob(SN);
        double sumFit = 0.0;
        for (int i = 0; i < SN; ++i) {
            prob[i] = foods[i].fit;
            sumFit += foods[i].fit;
        }
        if (sumFit == 0.0) sumFit = 1.0;
        for (int i = 0; i < SN; ++i)
            prob[i] /= sumFit;

        int t = 0;
        int i = 0;
        while (t < SN) {
            double r = rand01();
            if (r < prob[i]) {
                t++;

                int k;
                do { k = randInt(0, SN - 1); } while (k == i);
                int j = randInt(0, D - 1);

                vector<double> v = foods[i].x;
                double phi = randUniform(-1.0, 1.0);

                v[j] = foods[i].x[j]
                     + phi * (foods[i].x[j] - foods[k].x[j])
                     + c_gbest * (gbest.x[j] - foods[i].x[j]);

                if (v[j] < lower) v[j] = lower;
                if (v[j] > upper) v[j] = upper;

                double newCost = objective(v);
                evalCount++;

                if (newCost < foods[i].cost) {
                    foods[i].x = v;
                    foods[i].cost = newCost;
                    foods[i].fit = calcFitness(newCost);
                    foods[i].trial = 0;
                } else {
                    foods[i].trial++;
                }
            }
            i++;
            if (i == SN) i = 0;
        }

        // 3. Scout bee
        int worst = 0;
        for (int idx = 1; idx < SN; ++idx)
            if (foods[idx].trial > foods[worst].trial)
                worst = idx;

        if (foods[worst].trial > limit) {
            for (int j = 0; j < D; ++j)
                foods[worst].x[j] = randUniform(lower, upper);
            foods[worst].cost = objective(foods[worst].x);
            evalCount++;
            foods[worst].fit = calcFitness(foods[worst].cost);
            foods[worst].trial = 0;
        }

        // update global best
        for (int idx = 0; idx < SN; ++idx)
            if (foods[idx].cost < gbest.cost)
                gbest = foods[idx];
    }

    return gbest;
}

// ============ Printing ============
void printResult(const string& name, const FoodSource& best,
                 double lower, double upper, long long evalCount) {
    cout << "****** " << name << " ******\n";
    cout << "Best cost f(x*) = " << scientific << setprecision(10)
         << best.cost << "\n";
    cout << "Best position [x, y] = [" << fixed << setprecision(8)
         << best.x[0] << ", " << best.x[1] << "]\n";
    cout << "Number of evaluations = " << evalCount << "\n";
    cout << "Bounds for [x, y] = [(" << lower << ", " << upper
         << "), (" << lower << ", " << upper << ")]\n\n";
}

// ============ main ============
int main() {
    // Fix seed for reproducibility
    srand(42);

    int D = 2;
    double lower = -5.0, upper = 5.0;
    int SN = 20, limit = 50, maxCycle = 500;

    long long evalABC = 0, evalGABC = 0;

    FoodSource bestABC  = runABC (D, lower, upper, SN, limit, maxCycle, evalABC);
    FoodSource bestGABC = runGABC(D, lower, upper, SN, limit, maxCycle, evalGABC);

    printResult("ABC goc", bestABC, lower, upper, evalABC);
    printResult("GABC",    bestGABC, lower, upper, evalGABC);

    return 0;
}
