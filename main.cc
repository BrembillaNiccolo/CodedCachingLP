#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include "gurobi_c++.h"
#include <sstream>
#include <cmath>
#include <omp.h>
#include <list>
#include "parser.h"
#include <set>
#include <ctime>
#include "setup.h"
#include <sys/stat.h>
#include <iomanip>  // add this include for std::setprecision
#include <unordered_map>

using namespace std;

int Debug = 0;
std::ofstream outFile;
std::ofstream solutionFile;
std::ofstream outputSolutionFile=std::ofstream("output_solution.txt");
bool printSolutions = false;

// Custom hash function for std::vector<int>
struct VectorHash
{
    template <typename T>
    std::size_t operator()(const std::vector<T> &v) const
    {
        std::size_t seed = 0;
        for (const auto &i : v)
        {
            seed ^= std::hash<T>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/*
    Print the value of the variable
    If value < setup.numFile print W
    If value < setup.numUser + setup.numFile + setup.commonInformations print Z
    If value < setup.numUser + setup.numFile + setup.commonInformations + numDemands print X
    Print the demand
*/

void printValue(int value, Setup &setup)
{
    if (value < setup.numFile)
    {
        outFile << "W" << value + 1 << " ";
    }
    else if (value < setup.numUser + setup.numFile)
    {
        outFile << "Z" << value - setup.numFile + 1 << " ";
    }
    else
    {
        // TODO print correct demand
        outFile << "X";
        for (auto &d : setup.demands[value - setup.numFile - setup.numUser - setup.commonInformations])
        {
            outFile << d + 1;
        }
        outFile << " ";
    }
}

/*
    Check if distance between two points is less than 1e-5
    Used to check if points are collinear
*/
int ColinearCheck(double x1, double y1, double x2, double y2, double x3, double y3)
{
    // Calculate the area of the triangle formed by the three points
    //     double area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);

    //     // If the area is zero, the points are collinear
    //     if (std::fabs(area) < 1e-8) {
    //         return 1; // Points are on the same line
    //     } else {
    //         return 0; // Points are not on the same line
    //     }
    if (fabs((y1 + y2) / 2 - y3) <= 1e-5)
        return 1;
    return 0;
}

/*
    Create all possible setup.demands
    Each user can request any file
    Generate all possible combinations of requests
    Store them in the setup.demands vector
*/

void createAllDemands(Setup &setup)
{
    int totalCombinations = 1;

    // Compute the total number of combinations
    for (int i = 0; i < setup.numUser; i++)
        totalCombinations *= setup.numFile;
    try
    {

        // Reserve space for all combinations
        setup.demands.reserve(totalCombinations);
    }
    catch (const bad_alloc &e)
    {
        cerr << "Memory allocation failedduring creation of all setup.demands: " << e.what() << endl;
        exit(1);
    }
    vector<int> current(setup.numUser, 0);

    // Generate all combinations
    for (int idx = 0; idx < totalCombinations; idx++)
    {
        setup.demands.push_back(current);
        for (int k = setup.numUser - 1; k >= 0; k--)
        {
            if (++current[k] < setup.numFile)
                break;
            current[k] = 0;
        }
    }
}

/*
    Compute the factorial of a number
    Used to compute the total number of permutations
*/

uint64_t factorial(int n)
{
    uint64_t result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

void createOptimizedDemands(Setup &setup)
{
    vector<vector<int>> tempDemands;
    vector<vector<int>> allDemands_num;
    int totalCombinations = 1;

    // Compute the total number of combinations
    for (int i = 0; i < setup.numUser; i++)
        totalCombinations *= setup.numFile;
    try
    {

        // Reserve space for all combinations
        tempDemands.reserve(totalCombinations);
    }
    catch (const bad_alloc &e)
    {
        cerr << "Memory allocation failed during creation of all tempDemands: " << e.what() << endl;
        exit(1);
    }
    vector<int> current(setup.numUser, 0);

    // Generate all combinations
    for (int idx = 0; idx < totalCombinations; idx++)
    {
        tempDemands.push_back(current);
        for (int k = setup.numUser - 1; k >= 0; k--)
        {
            if (++current[k] < setup.numFile)
                break;
            current[k] = 0;
        }
    }

    std::cout << "TempDemands size: " << tempDemands.size() << endl;
    for (auto demand : tempDemands)
    {
        vector<int> demandNum(setup.numFile, 0);
        for (size_t i = 0; i < demand.size(); i++)
        {
            demandNum[demand[i]]++;
        }
        // sort highest to lowest
        sort(demandNum.begin(), demandNum.end(), greater<int>());
        int sum = 0;
        for (int num : demandNum)
        {
            sum += num;
        }
        if (sum != setup.numUser)
        {
            continue;
        }
        // if not in allDemands_num, add it
        if (find(allDemands_num.begin(), allDemands_num.end(), demandNum) == allDemands_num.end())
        {
            allDemands_num.push_back(demandNum);
            setup.demands.push_back(demand);
        }
    }
}

void createAllDifferentDemandsForSpecificDemandType(Setup &setup)
{
    // Generate all possible combinations of requests, then add to setup.demands only the ones that respect the condition of setup.demand_requests
    vector<vector<int>> tempDemands;
    createAllDemands(setup);
    cout << "TempDemands size: " << tempDemands.size() << endl;

    for (auto demand : tempDemands)
    {
        vector<int> demandNum(setup.numFile, 0);
        for (size_t i = 0; i < demand.size(); i++)
        {
            demandNum[demand[i]]++;
        }
        // sort highest to lowest
        sort(demandNum.begin(), demandNum.end(), greater<int>());
        if (demandNum == setup.demand_requests)
        {
            setup.demands.push_back(demand);
        }
    }
}

void createOneDifferentDemandsForSpecificDemandType(Setup &setup)
{
    vector<int> demand(setup.numUser, 0);
    int idx = 0;
    for (int i = 0; i < setup.numFile; i++)
    {
        for (int j = 0; j < setup.demand_requests[i]; j++)
        {
            demand[idx++] = i;
        }
    }
    setup.demands.push_back(demand);
    if (setup.cycleDemands)
    {
        for (int i = 1; i < setup.numUser; i++)
        {
            vector<int> demandCycle(setup.numUser, 0);
            for (int j = 0; j < setup.numFile; j++)
            {
                demandCycle[(i + j) % setup.numUser] = demand[j];
            }
            setup.demands.push_back(demandCycle);
        }
    }
}
/*
    Based on the flag used
    Generate all possible setup.demands
    Generate all possible permutations of setup.demands
    Generate one demand with all different requests
    Store them in the setup.demands vector
*/

void createDemands(Setup &setup)
{
    // Generate all possible setup.demands if user requested
    if (setup.allDemands)
    {
        createAllDemands(setup);
    }
    if (setup.allDemandsOpt)
    {
        createOptimizedDemands(setup);
    }

    // Generate all possible permutations of setup.demands if all different
    else if (setup.allDifferent)
    {
        createAllDifferentDemandsForSpecificDemandType(setup);
    }
    // Generate one demand with all different requests
    else if (setup.oneDifferent)
    {
        createOneDifferentDemandsForSpecificDemandType(setup);
    }
    if (Debug > 0)
    {
        outFile << "Demands number: " << setup.demands.size() << endl;
        outFile << "Demands: " << endl;
        for (auto &demand : setup.demands)
        {
            for (auto &d : demand)
            {
                outFile << d << " ";
            }
            outFile << endl;
        }
    }
}

/*
    Generate all permutations of a vector
    Store them in the permutations vector
*/

void generatePermutations(vector<int> vec, vector<vector<int>> &permutations)
{

    // Ensure the vector is sorted
    sort(vec.begin(), vec.end());

    // Compute N!
    uint64_t totalPermutations = factorial(vec.size());
    try
    {

        // Reserve space for permutations
        permutations.reserve(totalPermutations);
    }
    catch (const bad_alloc &e)
    {
        cerr << "Memory allocation failed during permutation generation: " << e.what() << endl;
        exit(1);
    }
    do
    {

        // Store each permutation
        permutations.push_back(vec);

        // Generate next permutation
    } while (next_permutation(vec.begin(), vec.end()));
}

/*
    Generate all possible permutations of the users and files
    Store them in the userPerms and filePerms vectors
*/

void createUserFilePermutations(Setup &setup, vector<vector<int>> &userPerms, vector<vector<int>> &filePerms)
{
    vector<int> user(setup.numUser);
    for (int i = 0; i < setup.numUser; i++)
    {
        user[i] = i;
    }
    generatePermutations(user, userPerms);
    vector<int> file(setup.numFile);
    for (int i = 0; i < setup.numFile; i++)
    {
        file[i] = i;
    }
    generatePermutations(file, filePerms);
}

/*
    Generate the inverse permutation of a permutation
    Used to permute the setup.demands
*/

vector<int> invertPermutation(const vector<int> &perm)
{
    int n = perm.size();
    vector<int> inv(n);
    for (int i = 0; i < n; i++)
    {
        inv[perm[i]] = i;
    }
    return inv;
}

/*
    Generate all possible permutations of the table
    Store them in the allPermutations vector
    For each permutation of files and users
    Generate a new permutation of the table
    If the new permutation is not in allPermutations, add it
    Use Tian et al. 2019 method to generate unique permutations
*/

void createTablePermutations(Setup &setup, vector<vector<int>> &filePerms,
                             vector<vector<int>> &userPerms, unordered_set<vector<int>, VectorHash> &allPermutations,
                             int numSingleVar)
{
    auto start = chrono::high_resolution_clock::now();
    try
    {
        // #pragma omp parallel for schedule(dynamic, 1) collapse(2) // Use dynamic scheduling for better load balancing
        for (auto &filePerm : filePerms)
        {
            for (auto &userPerm : userPerms)
            {
                // Generate the inverse permutation of the user permutation, used to permute the setup.demands
                vector<int> invUserPerm = invertPermutation(userPerm);

                vector<int> P(numSingleVar, -1);

                // Generate the new permutation for the files
                for (int i = 0; i < setup.numFile; i++)
                {
                    P[i] = filePerm[i];
                }

                // Generate the new permutation for the users
                for (int i = 0; i < setup.numUser; i++)
                {
                    P[setup.numFile + i] = setup.numFile + userPerm[i];
                }
                // bool found = true;
                //  Generate the new permutation for the setup.demands
                size_t totalNotFound = 0;
                for (size_t i = 0; i < setup.demands.size(); i++)
                {
                    auto demand = setup.demands[i];
                    vector<int> newDemand(setup.numUser);

                    // Permute the setup.demands using the inverse user permutation and the file permutation
                    for (int j = 0; j < setup.numUser; j++)
                    {
                        newDemand[j] = filePerm[demand[invUserPerm[j]]];
                    }

                    // Find the position of the new demand in the setup.demands vector
                    auto position = find(setup.demands.begin(), setup.demands.end(), newDemand);

                    // If the demand is found, add it to the permutation
                    if (position != setup.demands.end())
                    {
                        P[setup.numUser + setup.numFile + setup.commonInformations + i] = setup.numUser + setup.numFile + setup.commonInformations + (position - setup.demands.begin());
                    }
                    else
                    {
                        // found = false;
                        if (Debug > 0)
                        {
                            outFile << "Demand not found" << endl;
                        }
                        P[setup.numUser + setup.numFile + setup.commonInformations + i] = -1;
                        totalNotFound++;
                    }
                }
                if (totalNotFound == setup.demands.size())
                {
                    // If all setup.demands are not found, skip the permutation
                    continue;
                }

                // If P not in allPermutations, add it
                // #pragma omp critical
                {
                    allPermutations.insert(P);
                }
            }
        }
        // outFile << "Permutations number: " << allPermutations.size() << endl;
        if (Debug > 1)
        {
            outFile << "Permutations: " << endl;
            for (auto &perm : allPermutations)
            {
                for (auto &p : perm)
                {
                    printValue(p, setup);
                }
                outFile << endl;
            }
        }
    }
    catch (const bad_alloc &e)
    {
        cerr << "Memory allocation failed during table permutation generation: " << e.what() << endl;
        exit(1);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time taken to generate permutations: " << duration << " ms" << endl;
}

/*
    Generate all possible subsets of a vector
    Store them in the result vector
    maxNum is used to generate subsets with only one request at a time
*/

void generateSubsets(const vector<int> &elements, int k, int start, vector<int> &current, vector<vector<int>> &result, int maxNum)
{
    if ((int)current.size() == k)
    {
        result.push_back(current);
        return;
    }
    for (int i = start; i < (int)elements.size(); i++)
    {
        // If last element of current is >= maxNum skip
        // This is done if we want to generate subsets with only one request at a time
        if (current.size() > 0 && current.back() >= maxNum)
            continue;
        current.push_back(elements[i]);
        // Recursive call to generate the next element
        generateSubsets(elements, k, i + 1, current, result, maxNum);
        current.pop_back();
    }
}

/*
    Add constraints for the symmetric user-file constraints
    For each subset of users and files, add a constraint that the value of the subset is the same as the value of the symmetric subset
    Add the constraint to the model
*/
void addSymmetricUserFileConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    vector<vector<int>> subsetsUsers;
    vector<int> allUser(setup.numUser);
    for (int i = 0; i < setup.numUser; i++)
    {
        allUser[i] = i + setup.numFile;
    }
    vector<vector<int>> subsetsFiles;
    vector<int> allFile(setup.numFile);
    for (int i = 0; i < setup.numFile; i++)
    {
        allFile[i] = i;
    }
    vector<int> current;
    for (int i = 0; i <= setup.numFile; i++)
    {
        // start from 1, as j = 0 already covered previously.
        for (int j = 0; j <= setup.numUser; j++)
        {
            auto r = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "symrefvar");
            current.clear();

            generateSubsets(allUser, i, 0, current, subsetsUsers, setup.numFile + setup.numUser);
            for (auto filevec : subsetsUsers)
            {
                int idx = 0;
                for (auto ix : filevec)
                {
                    idx += 1 << ix;
                }
                current.clear();
                generateSubsets(allFile, j, 0, current, subsetsFiles, setup.numFile + setup.numUser);
                int jdxBest = 0;
                for (auto cachevec : subsetsFiles)
                {
                    int jdx = idx;
                    for (auto jx : cachevec)
                    {
                        jdx += 1 << jx;
                    }
                    if (jdxBest == 0)
                        jdxBest = jdx;
                    else if (jdx < jdxBest)
                    {
                        setup.symmetryCheck[jdxBest] = jdx;
                        jdxBest = jdx;
                    }
                    else
                    {
                        setup.symmetryCheck[jdx] = jdxBest;
                    }
                    model.addConstr(bitmapTable[jdx] == r, "symmetri_to_" + bitmapTable[jdx].get(GRB_StringAttr_VarName));
                }
                subsetsFiles.clear();
            }
            subsetsUsers.clear();
        }
    }
}

void addSymmetricUserFileConstraintsNotFullVariables(unordered_set<vector<int>, VectorHash> &allPermutations, Setup &setup, GRBModel &model,
                                                     GRBVar *bitmapTable)
{
    vector<vector<int>> subsets;
    subsets.push_back(vector<int>());
    vector<int> allVariables(setup.numUser + setup.numFile);
    for (int i = 0; i < setup.numUser + setup.numFile; i++)
    {
        allVariables[i] = i;
    }
    // For each subset of variables, add the constraint that the sum of the variables in the subset is equal to the sum of the variables in the subset for all permutations
    for (int subsetSize = 1; subsetSize < setup.numUser + setup.numFile; subsetSize++)
    {
        vector<int> current;
        generateSubsets(allVariables, subsetSize, 0, current, subsets, setup.numUser + setup.numFile);
    }
    subsets.erase(
        std::remove_if(subsets.begin(), subsets.end(),
                       [setup](const vector<int> &subset)
                       {
                           for (size_t i = 0; i < subset.size(); i++)
                           {
                               if (subset[i] >= setup.numUser + setup.numFile + setup.commonInformations)
                                   return false;
                           }
                           return true;
                       }),
        subsets.end());

    vector<vector<int>> subsetsDemand;
    for (size_t i = 0; i < allPermutations.begin()->size() - setup.numUser - setup.numFile - setup.commonInformations; i++)
    {
        int demandIdx = setup.numUser + setup.numFile + setup.commonInformations + i;

        // All subset in subsets + index of demand
        for (auto &subset : subsets)
        {
            vector<int> newSubset = subset;
            newSubset.push_back(demandIdx);
            subsetsDemand.push_back(newSubset);
        }
    }
    uint64_t totalEquations = 0;

    // Loop over each subset in your collection of subsets.

    for (auto &subset : subsetsDemand)
    {
        // Compute sumSubset (if needed for later, e.g. for a Gurobi constraint)
        uint64_t sumSubset = 0;
        for (int s : subset)
        {
            if (s < setup.numUser + setup.numFile + setup.commonInformations)
                sumSubset += 1ULL << s;
            else
                sumSubset += ((s - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
        }

        // Use an unordered_set for fast lookup of already processed permutations.
        unordered_set<uint64_t> alreadyProcessed;
        // Preallocate a temporary vector for collecting permutation values.
        vector<int> values;
        // Reserve enough space in 'values' to avoid reallocation in each inner loop.
        try
        {
            values.reserve(subset.size());
        }
        catch (const bad_alloc &e)
        {
            cerr << "Memory allocation failed during symmetry subset processing: " << e.what() << endl;
            exit(1);
        }

        // Skip the first permutation using an iterator
        auto it = allPermutations.begin();
        if (it == allPermutations.end())
            return; // Nothing to process

        ++it; // Skip the first permutation

        for (; it != allPermutations.end(); ++it)
        {
            values.clear();

            const std::vector<int> &perm = *it;
            for (int s : subset)
            {
                values.push_back(perm[s]);
            }

            if (find(values.begin(), values.end(), -1) != values.end())
                continue;

            // Compute the sum for the permutation.
            uint64_t sumPermutation = 0;
            for (int v : values)
            {
                if (v < setup.numUser + setup.numFile + setup.commonInformations)
                    sumPermutation += 1ULL << v;
                else
                    sumPermutation += ((v - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
            }
            if (sumPermutation > sumSubset)
            {
                setup.symmetryCheck[sumPermutation] = sumSubset;
            }

            // Skip if this permutation (by its sum) has already been processed.
            if (alreadyProcessed.find(sumPermutation) != alreadyProcessed.end() || sumSubset >= sumPermutation)
                continue;

            // Mark as processed and update your equation count.
            alreadyProcessed.insert(sumPermutation);
            totalEquations++;

            // TODO add  Gurobi constraint: sumSubset == sumPermutation.
            {
                model.addConstr(bitmapTable[sumSubset] == bitmapTable[sumPermutation], "Equality between " + bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + " and " + bitmapTable[sumPermutation].get(GRB_StringAttr_VarName));
            }
            if (Debug > 1)
            {
                outFile << "H (";
                for (auto &s : subset)
                {
                    outFile << s << ",";
                }
                outFile << "=" << sumSubset << ") = H (";
                // Only this to maintain
                for (auto &s : subset)
                {
                    outFile << perm[s] << ",";
                }
                outFile << "=" << sumPermutation << ")" << endl;
            }
        }
    }
}

/*
    Generate constraints for the symmetric case
    H(subset) = H(permutation of subset)
    Generate all possible subsets of the User and Files
    Then create the constraints in which exactly one of the variables is a demand
    Add the constraints to the model
*/

void addSymmetricDemandsConstraints(unordered_set<vector<int>, VectorHash> &allPermutations, Setup &setup, GRBModel &model,
                                    GRBVar *bitmapTable)
{
    if (!setup.fullVariables)
    {
        addSymmetricUserFileConstraintsNotFullVariables(allPermutations, setup, model, bitmapTable);
    }
    else
    {
        vector<vector<int>> subsets;
        subsets.push_back(vector<int>());
        vector<int> allVariables(allPermutations.begin()->size() - setup.commonInformations);
        for (int i = 0; i < int(allPermutations.begin()->size()); i++)
        {
            if (i < setup.numUser + setup.numFile)
                allVariables[i] = i;
            else if (i >= setup.numUser + setup.numFile + setup.commonInformations)
                allVariables[i - setup.commonInformations] = i;
        }
        // For each subset of variables, add the constraint that the sum of the variables in the subset is equal to the sum of the variables in the subset for all permutations
        for (int subsetSize = 1; subsetSize < allPermutations.begin()->size() - setup.commonInformations; subsetSize++)
        {
            vector<int> current;
            generateSubsets(allVariables, subsetSize, 0, current, subsets, allPermutations.begin()->size() + setup.commonInformations);
        }
        cout << "Subsets before filtering: " << subsets.size() << endl;
        /*
        for (int i = 0; i < subsets.size(); i++){
            cout << "Subset " << i << ": ";
            for (auto &s : subsets[i])
            {
                cout<<s<<" ";
            }
            cout << endl;
        }
            */
        subsets.erase(
            std::remove_if(subsets.begin(), subsets.end(),
                           [setup](const vector<int> &subset)
                           {
                               int position = 0;
                               for (size_t i = 0; i < subset.size(); i++)
                               {
                                   position += 1 << subset[i];
                               }
                               if (setup.maxValueSubsets[position] == 1)
                                   return true;
                               for (size_t i = 0; i < subset.size(); i++)
                               {
                                   if (subset[i] >= setup.numUser + setup.numFile + setup.commonInformations)
                                       return false;
                               }
                               return true;
                           }),
            subsets.end());

        uint64_t totalEquations = 0;
        // Print all permutations
        if (Debug > 1)
        {
            outFile << "Permutations: " << endl;
            for (auto &perm : allPermutations)
            {
                for (auto &p : perm)
                {
                    printValue(p, setup);
                }
                outFile << endl;
            }
        }
        // Loop over each subset in your collection of subsets.
        // #pragma omp parallel for schedule(dynamic, 1) // Use dynamic scheduling for better load balancing
        for (auto &subset : subsets)
        {
            // Compute sumSubset (if needed for later, e.g. for a Gurobi constraint)
            uint64_t sumSubset = 0;
            for (int s : subset)
            {
                sumSubset += 1ULL << s;
            }

            // Use an unordered_set for fast lookup of already processed permutations.
            unordered_set<uint64_t> alreadyProcessed;
            // Preallocate a temporary vector for collecting permutation values.
            vector<int> values;
            // Reserve enough space in 'values' to avoid reallocation in each inner loop.
            try
            {
                values.reserve(subset.size());
            }
            catch (const bad_alloc &e)
            {
                cerr << "Memory allocation failed during symmetry subset processing: " << e.what() << endl;
                exit(1);
            }
            // Skip the first permutation using an iterator
            auto it = allPermutations.begin();

            ++it; // Skip the first permutation

            for (; it != allPermutations.end(); ++it)
            {
                values.clear();

                const std::vector<int> &perm = *it;
                for (int s : subset)
                {
                    values.push_back(perm[s]);
                }
                // if values contains -1, skip
                if (find(values.begin(), values.end(), -1) != values.end())
                    continue;
                // Compute the sum for the permutation.
                uint64_t sumPermutation = 0;
                for (int v : values)
                {
                    sumPermutation += 1ULL << v;
                }
                if (sumPermutation > sumSubset)
                {
                    setup.symmetryCheck[sumPermutation] = sumSubset;
                }

                // Skip if this permutation (by its sum) has already been processed.
                if (alreadyProcessed.find(sumPermutation) != alreadyProcessed.end() || sumSubset >= sumPermutation)
                    continue;

                // Mark as processed and update your equation count.
                alreadyProcessed.insert(sumPermutation);
                totalEquations++;

                // TODO add  Gurobi constraint: sumSubset == sumPermutation.
                // #pragma omp critical
                {
                    model.addConstr(bitmapTable[sumSubset] == bitmapTable[sumPermutation], "Equality between " + bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + " and " + bitmapTable[sumPermutation].get(GRB_StringAttr_VarName));
                }
            }
        }
    }

    // outFile << "Total equations: " << totalEquations << endl;
}

/*
    Generate constraints for the cross entropy
    I(A,B|C) = H(A|C) + H(B|C) - H(A,B|C) - H(C)
    Get as input a vector of unique subsets
    Generate all permutations of size 2 of the last subset (it has all variables)
    These will become A and B of the cross entropy, the rest will be C
*/

void generateCrossEntropyConstraints(vector<int> &subset, vector<int> &innerSubset, Setup &setup, GRBModel &model,
                                     GRBVar *bitmapTable, int totalEquations)
{
    // Generate variable number for Gurobi constraints
    uint64_t entropyA = 0;
    uint64_t entropyB = 0;
    uint64_t entropyC = 0;
    if (!setup.fullVariables)
    {
        if (innerSubset[0] < setup.numUser + setup.numFile + setup.commonInformations)
            entropyA = 1ULL << innerSubset[0];
        else
            entropyA = (innerSubset[0] - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations);
        if (innerSubset[1] < setup.numUser + setup.numFile + setup.commonInformations)
            entropyB = 1ULL << innerSubset[1];
        else
            entropyB = (innerSubset[1] - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations);
    }
    else
    {
        entropyA = 1ULL << innerSubset[0];
        entropyB = 1ULL << innerSubset[1];
    }
    for (size_t i = 0; i < subset.size(); i++)
    {
        if (subset[i] != innerSubset[0] && subset[i] != innerSubset[1])
        {
            if (!setup.fullVariables)
            {
                if (subset[i] < setup.numUser + setup.numFile + setup.commonInformations)
                    entropyC += 1ULL << subset[i];
                else
                    entropyC += ((subset[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
            }
            else
            {
                entropyC += 1ULL << subset[i];
            }
        }
    }
    // #pragma omp critical
    if (setup.maxValueSubsets[entropyC] == 0)
    {
        // Add the constraint to the model
        // I(A,B|C) = H(A|C) + H(B|C) - H(A,B|C) - H(C)
        // H(A|C) = H(A) + H(C) - I(A,C)
        // H(B|C) = H(B) + H(C) - I(B,C)
        // I(A,C) = H(A) + H(C) - H(A,C)
        // I(B,C) = H(B) + H(C) - H(B,C)
        // I(A,B,C) = H(A,B) + H(C) - H(A,B,C)
        if (setup.symmetryCheck[entropyC] == 0 || setup.symmetryCheck[entropyB + entropyC] == 0 || setup.symmetryCheck[entropyA + entropyB + entropyC] == 0 || setup.symmetryCheck[entropyC] == 0)
        {
            model.addConstr(bitmapTable[entropyA + entropyC] + bitmapTable[entropyB + entropyC] - bitmapTable[entropyA + entropyB + entropyC] - bitmapTable[entropyC] >= 0, "I(" + bitmapTable[entropyA].get(GRB_StringAttr_VarName) + "," + bitmapTable[entropyB].get(GRB_StringAttr_VarName) + "|" + bitmapTable[entropyC].get(GRB_StringAttr_VarName) + ")");
        }
    }
    // TODO add Gurobi constraints
    if (Debug > 1)
    {
        outFile << "I (";
        for (size_t i = 0; i < innerSubset.size(); i++)
        {
            outFile << innerSubset[i];
            if (i != innerSubset.size() - 1)
                outFile << ",";
        }
        outFile << "|";
        for (size_t i = 0; i < subset.size(); i++)
        {
            if (subset[i] != innerSubset[0] && subset[i] != innerSubset[1])
            {
                outFile << subset[i];
                if (i != subset.size() - 1)
                    outFile << ",";
            }
        }
        outFile << ") = H (" << entropyA + entropyC << ") + H (" << entropyB + entropyC
                << ") - H (" << entropyA + entropyB + entropyC << ") - H (" << entropyC << ")" << endl;
    }
    totalEquations++;
}

uint64_t genContraintsFromSubset(vector<vector<int>> &subsets, Setup &setup, GRBModel &model,
                                 GRBVar *bitmapTable)
{
    vector<vector<int>> subsetDim2Permutations;
    vector<int> current;
    vector<int> values;

    // Generate all permutations of size 2 of the last subset, that is the one with all variables
    for (size_t i = 0; i < subsets.back().size(); i++)
    {
        values.push_back(subsets.back()[i]);
    }
    if (!setup.fullVariables)
        generateSubsets(values, 2, 0, current, subsetDim2Permutations, setup.numUser + setup.numFile + setup.commonInformations);
    else
        generateSubsets(values, 2, 0, current, subsetDim2Permutations, subsets.back().size());
    uint64_t totalEquations = 0;

    // For each subset generate all possible constraints using 1 variable as A, 1 variable as B and the rest as C
    // #pragma omp parallel for reduction(+:totalEquations) schedule(dynamic, 1)
    for (auto &subset : subsets)
    {
        for (auto &innerSubset : subsetDim2Permutations)
        {

            // If innerSubset (subset of 2 variables) is not in subset, skip
            // Every subset is unique, so we don't check for duplicates
            // The 2 variables in innerSubset represent A and B, so we need to check if they are in subset
            if (find(subset.begin(), subset.end(), innerSubset[0]) == subset.end() || find(subset.begin(), subset.end(), innerSubset[1]) == subset.end())
                continue;

            generateCrossEntropyConstraints(subset, innerSubset, setup, model, bitmapTable, totalEquations);
        }
    }
    return totalEquations;
}

/*
    Generate constraints for the user-file entropy
    I(A,B|C) = H(A|C) + H(B|C) - H(A,B|C) - H(C)
    A is Random Variable
    B is Random Variable
    C is a subset remaining Random Variables (null set is allowed)
*/

void generateUserFileConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    vector<vector<int>> subsets;
    vector<int> userFilePermutations(setup.numUser + setup.numFile + setup.commonInformations);

    // Generate vector of indices
    for (int i = 0; i < setup.numUser + setup.numFile + setup.commonInformations; i++)
    {
        userFilePermutations[i] = i;
    }

    // Generate all subsets of size 2 or more
    for (int subsetSize = 2; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + 1; subsetSize++)
    {
        vector<int> current;
        generateSubsets(userFilePermutations, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations);
    }
    // outFile << "User-File subsets number: " << subsets.size() << endl;
    
    subsets.erase(
        std::remove_if(subsets.begin(), subsets.end(),
                       [setup](const vector<int> &subset)
                       {
                           int position = 0;
                           for (size_t i = 0; i < subset.size(); i++)
                           {
                               position += 1 << subset[i];
                           }
                           if (setup.symmetryCheck[position] != 0)
                               return true;
                           return false;
                       }),
        subsets.end());
    // For each subset generate all possible constraints using 1 variable as A, 1 variable as B and the rest as C
    uint64_t totalEquations = genContraintsFromSubset(subsets, setup, model, bitmapTable);

    if (Debug > 0)
        outFile << "Total equations in user-file constraints: " << totalEquations << endl;
}

/*
    Generate constraints for the message demand
    I(A,B|C) = H(A|C) + H(B|C) - H(A,B|C) - H(C)
    A U B U C must contain exactly one demand
    A is Random Variable
    B is Random Variable
    C is a subset remaining Random Variables (null set is allowed)
*/

void generateMessageDemandConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    cout << "Generating message demand constraints" << endl;
    if (!setup.fullVariables)
    {

        vector<vector<int>> subsets;
        vector<int> userFileDemandPermutations(setup.numUser + setup.numFile + setup.commonInformations);

        // Generate vector of indices
        for (int i = 0; i < setup.numUser + setup.numFile + setup.commonInformations; i++)
        {
            userFileDemandPermutations[i] = i;
        }

        // Generate all subsets of size 1 or more
        for (int subsetSize = 1; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + 1; subsetSize++)
        {
            vector<int> current;
            generateSubsets(userFileDemandPermutations, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations);
        }
        uint64_t totalEquations = 0;
        
        vector<vector<int>> typeDemands;
        for (size_t i = 0; i < setup.demands.size(); i++)
        {
            vector<int> demandIdx(setup.numFile, 0);
            for (size_t j = 0; j < setup.demands[i].size(); j++)
            {
                demandIdx[setup.demands[i][j]]++;
            }
            // sort highest to lowest
            sort(demandIdx.begin(), demandIdx.end(), greater<int>());
            if (find(typeDemands.begin(), typeDemands.end(), demandIdx) == typeDemands.end())
            {
                typeDemands.push_back(demandIdx);
            }
        }
        vector<int> numDemandsperType(typeDemands.size(), 0);
        vector<int> demandIdxInType(setup.demands.size(), 0);
        // Count the number of setup.demands of each type
        for (size_t i = 0; i < setup.demands.size(); i++)
        {
            vector<int> demandIdx(setup.numFile, 0);
            for (size_t j = 0; j < setup.demands[i].size(); j++)
            {
                demandIdx[setup.demands[i][j]]++;
            }
            // sort highest to lowest
            sort(demandIdx.begin(), demandIdx.end(), greater<int>());
            auto it = find(typeDemands.begin(), typeDemands.end(), demandIdx);
            if (it != typeDemands.end())
            {
                numDemandsperType[it - typeDemands.begin()]++;
                demandIdxInType[i] = it - typeDemands.begin();
            }
        }
        cout << "Number of setup.demands per type: " << endl;
        for (size_t i = 0; i < typeDemands.size(); i++)
        {
            cout << "Type " << i << ": " << numDemandsperType[i] << endl;
        }
        cout << "Demand index in type: " << endl;
        for (size_t i = 0; i < setup.demands.size(); i++)
        {
            cout << "Demand " << i << ": " << demandIdxInType[i] << endl;
        }

        // For each demand add it to all subsets and generate constraints
        for (size_t i = 0; i < setup.demands.size(); i++)
        {
            if (demandIdxInType[i] > 0)
                continue;
            int demandIdx = setup.numUser + setup.numFile + setup.commonInformations + i;
            vector<vector<int>> subsetsDemand;

            // All subset in subsets + index of demand
            for (auto &subset : subsets)
            {
                vector<int> newSubset = subset;
                newSubset.push_back(demandIdx);
                subsetsDemand.push_back(newSubset);
            }
            if (Debug > 0)
            {
                outFile << "Working on demand " << i << endl;
            }
            // For each subset generate all possible constraints using 1 variable as A, 1 variable as B and the rest as C
            totalEquations += genContraintsFromSubset(subsetsDemand, setup, model, bitmapTable);
        }
        // outFile << "Total equations in message demand constraints: " << totalEquations << endl;
    }
    else
    {
        vector<vector<int>> subsets;
        vector<int> userFileDemandPermutations(setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());

        // Generate vector of indices
        for (size_t i = 0; i < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); i++)
        {
            userFileDemandPermutations[i] = i;
        }

        // Generate all subsets of size 2 or more
        for (size_t subsetSize = 2; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size() + 1; subsetSize++)
        {
            vector<int> current;
            generateSubsets(userFileDemandPermutations, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());
        }
        outFile << "User-File-Demand subsets number: " << subsets.size() << endl;
        uint64_t totalEquations = 0;
        subsets.erase(
            std::remove_if(subsets.begin(), subsets.end(),
                           [setup](const vector<int> &subset)
                           {
                               int position = 0;
                               bool toDelete = true;
                               for (size_t i = 0; i < subset.size(); i++)
                               {
                                   if (subset[i] >= setup.numUser + setup.numFile + setup.commonInformations)
                                       toDelete = false;
                                   position += 1 << subset[i];
                               }
                               if (setup.symmetryCheck[position] != 0)
                                   return true;
                               return toDelete;
                           }),
            subsets.end());
        // Delete if a demand is present, but not all the one before it
        totalEquations += genContraintsFromSubset(subsets, setup, model, bitmapTable);
    }
}

/*
    Add constraints for the common informations
    H(W0)=H(W1)=...=H(Wn)=1
    H(W0,W1)=H(W0,W2)=...=H(Wn-1,Wn)=2
    H(W0,W1,W2)=H(W0,W1,W3)=...=H(Wn-2,Wn-1,Wn)=3
    ...
*/
void addFileIndepenceConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    vector<int> filesIndependence(setup.numFile);
    for (int i = 0; i < setup.numFile; i++)
    {
        filesIndependence[i] = i;
    }
    vector<vector<int>> subsets;
    // Generate all subsets of size 1 or more
    for (int subsetSize = 1; subsetSize < setup.numFile + 1; subsetSize++)
    {
        vector<int> current;
        // Generate all subsets containing only files
        generateSubsets(filesIndependence, subsetSize, 0, current, subsets, setup.numFile);
    }
    uint64_t totalEquations = 0;
    for (auto &subset : subsets)
    {
        uint64_t sumSubset = 0;
        for (int s : subset)
        {
            sumSubset += 1ULL << s;
        }
        int size = (int)subset.size();
        model.addConstr(bitmapTable[sumSubset] == size, bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + " = " + to_string(size));
        totalEquations++;
        if (Debug > 1)
        {
            outFile << "H (";
            for (size_t i = 0; i < subset.size(); i++)
            {
                outFile << subset[i];
                if (i != subset.size() - 1)
                    outFile << ",";
            }
            outFile << ") = " << size << endl;
        }
    }
}

void monotonicalIncreasingFunction(Setup &setup, GRBModel &model, GRBVar *bitmapTable,
                                   vector<vector<int>> &subsets, vector<int> &allVariables)
{
    // Adding constraints H(any_variable + X) - H(X) >= 0 for all X
    for (auto &subset : subsets)
    {
        uint64_t sumSubset = 0;
        for (int s : subset)
        {
            if (!setup.fullVariables)
            {
                if (s < setup.numUser + setup.numFile + setup.commonInformations)
                    sumSubset += 1ULL << s;
                else
                    sumSubset += ((s - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)));
            }
            else
            {
                sumSubset += 1ULL << s;
            }
        }
        for (size_t i = 0; i < allVariables.size(); i++)
        {
            // If the variable is already in the subset, skip it
            // If the variable is a demand and the subset is not empty, skip it if not setup.fullVariables
            if (find(subset.begin(), subset.end(), allVariables[i]) != subset.end() || (!setup.fullVariables && allVariables[i] >= setup.numUser + setup.numFile + setup.commonInformations && sumSubset >= (1ULL << (setup.numUser + setup.numFile + setup.commonInformations))))
                continue;
            if (Debug > 0)
            {
                outFile << "H (";
                for (size_t j = 0; j < subset.size(); j++)
                {
                    outFile << subset[j];
                    if (j != subset.size() - 1)
                        outFile << ",";
                }
                outFile << "," << allVariables[i] << ") - H (";
                for (size_t j = 0; j < subset.size(); j++)
                {
                    outFile << subset[j];
                    if (j != subset.size() - 1)
                        outFile << ",";
                }
                outFile << ") >= 0   ";
                outFile << "H (";
                if (!setup.fullVariables)
                {
                    if (allVariables[i] < setup.numUser + setup.numFile + setup.commonInformations)
                        outFile << sumSubset + (1ULL << allVariables[i]);
                    else
                        outFile << sumSubset + ((allVariables[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)));
                }
                else
                {
                    outFile << sumSubset + (1ULL << allVariables[i]);
                }
                outFile << ") - H (" << sumSubset << ") >= 0" << endl;
            }
            // I can also take out if more than 2 setup.demands in the constraint
            if (!setup.fullVariables)
            {
                if (allVariables[i] < setup.numUser + setup.numFile + setup.commonInformations)
                    model.addConstr(bitmapTable[sumSubset + (1ULL << allVariables[i])] - bitmapTable[sumSubset] >= 0, "H(" + bitmapTable[sumSubset + (1ULL << allVariables[i])].get(GRB_StringAttr_VarName) + ") - H(" + bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + ") >= 0");
                else
                    model.addConstr(bitmapTable[sumSubset + ((allVariables[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)))] - bitmapTable[sumSubset] >= 0, "H(" + bitmapTable[sumSubset + ((allVariables[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)))].get(GRB_StringAttr_VarName) + ") - H(" + bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + ") >= 0");
            }
            else
            {
                model.addConstr(bitmapTable[sumSubset + (1ULL << allVariables[i])] - bitmapTable[sumSubset] >= 0, "H(" + bitmapTable[sumSubset + (1ULL << allVariables[i])].get(GRB_StringAttr_VarName) + ") - H(" + bitmapTable[sumSubset].get(GRB_StringAttr_VarName) + ") >= 0");
            }
        }
    }
}

void addMaxRateConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable,
                           GRBVar &r)
{
    // Adding constraints H(X_i) <= R for all i

    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        if (!setup.fullVariables)
        {
            uint64_t demandVal = (i + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
            model.addConstr(bitmapTable[demandVal] <= r, "MaximumRate_demand" + to_string(i));
        }
        else
        {
            uint64_t demandVal = 1ULL << (setup.numUser + setup.numFile + setup.commonInformations + i);
            model.addConstr(bitmapTable[demandVal] <= r, "MaximumRate_demand" + to_string(i));
        }
        // TODO add Gurobi constraints
        if (Debug > 0)
        {
            outFile << "H(" << ((i + 1) * (1ULL << (setup.numFile + setup.numUser))) << ") <= R" << endl;
        }
    }
}

void addCacheDemandDependenceConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    uint64_t allWvalues = 0;
    for (int i = 0; i < setup.numFile; i++)
    {
        allWvalues += 1ULL << i;
    }

    // Adding constraints H(Z_i | all W) = 0 for all i

    for (int i = 0; i < setup.numUser; i++)
    {
        uint64_t userVal = 1ULL << (i + setup.numFile);
        model.addConstr(bitmapTable[userVal + allWvalues] - bitmapTable[allWvalues] == 0,
                        "H(Z" + to_string(i) + "| all W) = 0");
        // TODO add Gurobi constraints
        if (Debug > 0)
        {
            outFile << "H(Z" << i << "| ";
            for (int j = 0; j < setup.numFile; j++)
            {
                outFile << "W" << j;
                if (j != setup.numFile - 1)
                    outFile << ",";
            }
            outFile << ") = H(" << userVal + allWvalues << ") - H(" << allWvalues << ") = 0" << endl;
        }
    }

    // Adding constraints H(X_i | all W) = 0 for all i

    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        auto demand = setup.demands[i];
        uint64_t demandVal = 0;
        if (!setup.fullVariables)
        {
            demandVal = (i + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
        }
        else
        {
            demandVal = 1ULL << (setup.numUser + setup.numFile + setup.commonInformations + i);
        }
        model.addConstr(bitmapTable[demandVal + allWvalues] - bitmapTable[allWvalues] == 0,
                        "H(" + bitmapTable[demandVal].get(GRB_StringAttr_VarName) + "| all W) = 0");
        // TODO add Gurobi constraints
        if (Debug > 0)
        {
            outFile << "H(X";
            for (int j = 0; j < setup.numUser; j++)
            {
                outFile << demand[j];
            }
            outFile << "| ";
            for (int j = 0; j < setup.numFile; j++)
            {
                outFile << "W" << j;
                if (j != setup.numFile - 1)
                    outFile << ",";
            }
            outFile << ") = H(" << demandVal + allWvalues << ") - H(" << allWvalues << ") = 0" << endl;
        }
    }
}

void addFileReconstructionConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    // Adding constraints H(W_d_i | Z_i, X) = 0 for all i and X
    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        auto demand = setup.demands[i];
        uint64_t demandVal = 0;
        if (!setup.fullVariables)
        {
            demandVal = (i + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
        }
        else
        {
            demandVal = 1ULL << (setup.numUser + setup.numFile + setup.commonInformations + i);
        }
        for (int j = 0; j < setup.numUser; j++)
        {
            uint64_t fileVal = 1ULL << demand[j];
            uint64_t userVal = 1ULL << (j + setup.numFile);
            model.addConstr(bitmapTable[fileVal + userVal + demandVal] - bitmapTable[userVal + demandVal] == 0,
                            "H(W" + to_string(demand[j]) + "| Z" + to_string(j) + "," + bitmapTable[demandVal].get(GRB_StringAttr_VarName) + ") = 0");
            // TODO add Gurobi constraints
            if (Debug > 0)
            {
                outFile << "H(W" << demand[j] << "| ";
                outFile << "Z" << j << ", X";
                for (int k = 0; k < setup.numUser; k++)
                {
                    outFile << demand[k];
                }
                outFile << ") = H(" << fileVal + userVal + demandVal << ") - H(" << userVal + demandVal << ") = 0" << endl;
            }
        }
    }
}

/*
    Add the basic constraints of the problem
    setup.M >= 0
    setup.M <= maxValM
    H(any_variable) >= 0
    H(Z_i) <= setup.M for all i
    H(X_i) <= R for all i
    H(Z_i | all W) = 0 for all i
    H(X_i | all W) = 0 for all i
    H(W_d_i | Z_i, X) = 0 for all i and X
*/

void addBasicProblemConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable, GRBVar &r)
{
    if (Debug > 0)
    {
        outFile << "Basic problem constraints: " << endl;
        outFile << "setup.M >= 0" << endl;
        outFile << "setup.M <= maxValM" << endl;
    }

    // Null set ==0
    model.addConstr(bitmapTable[0] == 0);

    vector<vector<int>> subsets;
    vector<int> allVariables(setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());
    for (size_t i = 0; i < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); i++)
    {
        allVariables[i] = i;
    }
    // Generate all subsets of size 1 or more
    if (!setup.fullVariables)
    {
        // At most one demand in the subset
        for (size_t subsetSize = 1; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); subsetSize++)
        {
            vector<int> current;
            generateSubsets(allVariables, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations);
        }
    }
    else
    {
        // Also all setup.demands in the subset
        for (size_t subsetSize = 1; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); subsetSize++)
        {
            vector<int> current;
            generateSubsets(allVariables, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());
        }
    }
    // Add empty subset
    subsets.push_back(vector<int>());

    subsets.erase(
        std::remove_if(subsets.begin(), subsets.end(),
                       [setup](const vector<int> &subset)
                       {
                           int position = 0;
                           for (size_t i = 0; i < subset.size(); i++)
                           {
                               position += 1 << subset[i];
                           }

                           if (setup.maxValueSubsets[position] == 1)
                           {
                               return true;
                           }
                           if (setup.symmetryCheck[position] != 0)
                           {
                               return true;
                           }
                           return false;
                       }),
        subsets.end());

    // Adding constraints H(X + any_variable) - H(X) >= 0 for all X and any_variable
    monotonicalIncreasingFunction(setup, model, bitmapTable, subsets, allVariables);

    // Adding constraints H(File1) = H(File2) = ... = H(FileN) = 1
    // Adding constraints H(File1, File2) = H(File1, File3) = ... = H(FileN-1, FileN) = 2
    // ...
    addFileIndepenceConstraints(setup, model, bitmapTable);

    // Adding constraints H(X_i) <= R for all i
    addMaxRateConstraints(setup, model, bitmapTable, r);
    // SetUp bit values for all W variables

    // Adding Cache and Demand dependence on Files
    //  Adding constraints H(Z_i | all W) = 0 for all i
    //  Adding constraints H(X_i | all W) = 0 for all i
    addCacheDemandDependenceConstraints(setup, model, bitmapTable);

    // Adding constraints H(W_d_i | Z_i, X) = 0 for all i and X
    addFileReconstructionConstraints(setup, model, bitmapTable);
}

// Backtracking function to generate all unique partitions
void generatePartitions(int index, const vector<int> &subset, vector<vector<int>> &partitions, vector<vector<vector<int>>> &allPartitions, set<vector<vector<int>>> &uniquePartitions)
{
    if (index == (int)subset.size())
    {
        // Ensure all 4 groups are non-empty
        bool valid = true;
        for (const auto &group : partitions)
        {
            if (group.empty())
            {
                valid = false;
                break;
            }
        }
        if (valid)
        {
            vector<vector<int>> sortedPartition = partitions;
            for (auto &part : sortedPartition)
            {
                sort(part.begin(), part.end());
            }
            sort(sortedPartition.begin(), sortedPartition.end());

            // Avoid duplicate partitions
            if (uniquePartitions.insert(sortedPartition).second)
            {
                allPartitions.push_back(partitions);
            }
        }
        return;
    }

    // Try placing the current element in each group
    for (int i = 0; i < 4; i++)
    {
        partitions[i].push_back(subset[index]);
        generatePartitions(index + 1, subset, partitions, allPartitions, uniquePartitions);
        partitions[i].pop_back(); // Backtrack
    }
}

/*
    Add user-file constraints
    Add user-file-one demand constraints
*/

void addShannonTypeConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    outFile << "Adding Shannon type constraints" << endl;
    generateUserFileConstraints(setup, model, bitmapTable);
    outFile << "Adding user-file-one demand constraints" << endl;
    generateMessageDemandConstraints(setup, model, bitmapTable);
}

/*
    Add non Shannon type constraints
    Add constraints for the cross entropy
*/

void addNonShannonTypeConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    cout << "Adding non Shannon type constraints" << endl;
    auto start = chrono::high_resolution_clock::now();
    auto CMI = [&](int x, int y, int z)
    {
        return bitmapTable[x + z] + bitmapTable[y + z] - bitmapTable[x + y + z] - bitmapTable[z];
    };
    auto ZhangYeung = [&](int a, int b, int c, int d)
    {
        return 2 * CMI(c, d, a) + CMI(a, c, d) + CMI(d, a, c) + CMI(c, d, b) + CMI(a, b, 0) - CMI(c, d, 0);
    };
    int numVariables = setup.numUser + setup.numFile + setup.commonInformations;
    vector<int> allVariables(numVariables);
    for (int i = 0; i < numVariables; i++)
    {
        allVariables[i] = i;
    }
    int k = 3; // Minimum number of variables for the Zhang Yeung inequality
    vector<vector<int>> subsetsInitial;
    vector<vector<int>> subsets;
    for (int i = k; i < numVariables; i++)
    {
        vector<int> current;
        generateSubsets(allVariables, i, 0, current, subsetsInitial, numVariables);
    }
    for (auto &subset : subsetsInitial)
    {
        subsets.push_back(subset);
    }
    // Add setup.demands to the subsets
    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        int demandIdx = setup.numUser + setup.numFile + setup.commonInformations + i;
        // All subset in subsets + index of demand
        for (auto &subset : subsetsInitial)
        {
            vector<int> newSubset = subset;
            newSubset.push_back(demandIdx);
            subsets.push_back(newSubset);
        }
    }
    // delete subsetsInitial;
    subsetsInitial.clear();

    subsets.erase(
        std::remove_if(subsets.begin(), subsets.end(),
                       [](const vector<int> &subset)
                       { return subset.size() < 4; }),
        subsets.end());
    int counter = 0;
    for (auto &subset : subsets)
    {
        vector<vector<int>> partitions(4);
        vector<vector<vector<int>>> allPartitions;
        set<vector<vector<int>>> uniquePartitions; // To track unique partitions

        generatePartitions(0, subset, partitions, allPartitions, uniquePartitions);
        for (auto &partition : allPartitions)
        {
            vector<int> idx(4, 0);
            for (int i = 0; i < 4; i++)
            {
                for (auto &group : partition[i])
                {
                    if (group < setup.numUser + setup.numFile + setup.commonInformations)
                        idx[i] += 1ULL << group;
                    else
                        idx[i] += ((group - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
                }
            }
            auto a = idx[0];
            auto b = idx[1];
            auto c = idx[2];
            auto d = idx[3];
            if (Debug > 0)
            {
                outFile << "Using subset: ";
                for (auto &s : subset)
                {
                    outFile << s << " ";
                }
                outFile << endl;
                for (int i = 0; i < 4; i++)
                {
                    outFile << "Group " << i << ": ";
                    for (auto &group : partition[i])
                    {
                        outFile << group << " ";
                    }
                    outFile << "Final index: " << idx[i] << endl;
                }
            }
            counter++;
            model.addConstr(ZhangYeung(a, b, c, d) >= 0);
            // model.addConstr(5*CMI(a,b,c)+3*CMI(a,c,b)+CMI(b,c,a)+2*CMI(a,b,d)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(4*CMI(a,b,c)+2*CMI(a,c,b)+CMI(b,c,a)+3*CMI(a,b,d)+CMI(a,d,b)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(4*CMI(a,b,c)+4*CMI(a,c,b)+CMI(b,c,a)+2*CMI(a,b,d)+CMI(a,d,b)+CMI(b,d,a)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(3*CMI(a,b,c)+3*CMI(a,c,b)+3*CMI(b,c,a)+2*CMI(a,b,d)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(3*CMI(a,b,c)+4*CMI(a,c,b)+2*CMI(b,c,a)+3*CMI(a,b,d)+CMI(a,d,b)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(3*CMI(a,b,c)+2*CMI(a,c,b)+2*CMI(b,c,a)+2*CMI(a,b,d)+CMI(a,d,b)+CMI(b,d,a)+2*CMI(c,d,0)-2*CMI(a,b,0)>=0);
            // model.addConstr(4*CMI(a,b,c)+9*CMI(a,c,b)+3*CMI(b,c,a)+6*CMI(a,b,d)+3*CMI(a,d,b)+3*CMI(c,d,0)-3*CMI(a,b,0)>=0);
        }
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if (Debug > 0)
    {
        outFile << "Total number of constraints added: " << counter << endl;
        outFile << "Time taken to add constraints: " << duration.count() << " milliseconds" << endl;
    }
    cout << "Total number of constraints added: " << counter << endl;
}

/*
    Add additional K constraints
    For each group of common informations
    Generate the sum of the variables before and after the common informations
    Generate the sum of the variables for the common informations
    Add the constraint H(before) + H(after) - H(before + after) = H(common informations)
    Add the constraint H(common informations + before) - H(before) = 0
    Add the constraint H(common informations + after) - H(after) = 0
    If Debug is enabled, print the constraints
*/
void addKConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable, vector<GRBConstr> &kConstraints)
{
    if (setup.commonInformationsStrings.size() == 0)
    {
        cerr << "Error: No additional K constraints provided.\n";
        exit(1);
    }
    int counter = 0;
    for (auto &group : setup.commonInformationsStrings)
    {
        vector<int> elements;
        for (auto &element : group)
        {
            if (element[0] == 'X')
            {
                string temp = element.substr(1);
                string token;
                istringstream ss(temp);
                vector<int> demandX(setup.numUser);
                int count = 0;
                while (getline(ss, token, ','))
                {
                    if (!isNumber(token))
                    {
                        cerr << "Error: Invalid number format inside -setup.commonInformations.\n";
                        exit(1);
                    }
                    if (stoi(token) >= setup.numFile)
                    {
                        cerr << "Error: Invalid number format inside -setup.commonInformations. Number must be less than setup.numFile.\n";
                        exit(1);
                    }
                    demandX[count] = stoi(token);
                    count++;
                }
                auto position = find(setup.demands.begin(), setup.demands.end(), demandX);
                if (position == setup.demands.end())
                {
                    cerr << "Error: Invalid demand format inside -setup.commonInformations. Demand not found.\n";
                    exit(1);
                }
                elements.push_back(position - setup.demands.begin() + setup.numUser + setup.numFile + setup.commonInformations);
            }
            else if (element[0] == 'Z')
            {
                string temp = element.substr(1);
                if (!isNumber(temp))
                {
                    cerr << "Error: Invalid number format inside -setup.commonInformations.\n";
                    exit(1);
                }
                if (stoi(temp) >= setup.numUser)
                {
                    cerr << "Error: Invalid number format inside -setup.commonInformations. Number must be less than setup.numUser.\n";
                    exit(1);
                }
                elements.push_back(stoi(temp) + setup.numFile);
            }
            else if (element[0] == 'W')
            {
                string temp = element.substr(1);
                if (!isNumber(temp))
                {
                    cerr << "Error: Invalid number format inside -setup.commonInformations.\n";
                    exit(1);
                }
                if (stoi(temp) >= setup.numFile)
                {
                    cerr << "Error: Invalid number format inside -setup.commonInformations. Number must be less than setup.numFile.\n";
                    exit(1);
                }
                elements.push_back(stoi(temp));
            }
            else if (element != "and")
            {
                cerr << "Error: Invalid format for -setup.commonInformations. Use -setup.commonInformations <num> <string> ... <string>.\n";
                exit(1);
            }
        }
        int sumBefore = 0;
        for (size_t i = 0; i < 2; i++)
        {
            if (elements[i] < setup.numUser + setup.numFile + setup.commonInformations)
                sumBefore += 1ULL << elements[i];
            else
                sumBefore += ((elements[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
        }
        int sumAfter = 0;
        for (size_t i = 2; i < elements.size(); i++)
        {
            if (elements[i] < setup.numUser + setup.numFile + setup.commonInformations)
                sumAfter += 1ULL << elements[i];
            else
                sumAfter += ((elements[i] - setup.numUser - setup.numFile - setup.commonInformations + 1) * 1ULL << (setup.numUser + setup.numFile + setup.commonInformations));
        }
        int kValue = 1ULL << (setup.numUser + setup.numFile + counter);
        counter++;
        string constraintName = "setup.commonInformations" + to_string(counter);
        GRBConstr kConstraint1 = model.addConstr(bitmapTable[sumBefore] + bitmapTable[sumAfter] - bitmapTable[sumBefore + sumAfter] == bitmapTable[kValue],
                                                 "" + bitmapTable[sumBefore].get(GRB_StringAttr_VarName) + " + " + bitmapTable[sumAfter].get(GRB_StringAttr_VarName) + " - " + bitmapTable[sumBefore + sumAfter].get(GRB_StringAttr_VarName) + " == " + bitmapTable[kValue].get(GRB_StringAttr_VarName));
        kConstraints.push_back(kConstraint1);
        GRBConstr kConstraint2 = model.addConstr(bitmapTable[kValue + sumBefore] - bitmapTable[sumBefore] == 0,
                                                 "H(" + bitmapTable[kValue + sumBefore].get(GRB_StringAttr_VarName) + ") - H(" + bitmapTable[sumBefore].get(GRB_StringAttr_VarName) + ") = 0");
        kConstraints.push_back(kConstraint2);
        GRBConstr kConstraint3 = model.addConstr(bitmapTable[kValue + sumAfter] - bitmapTable[sumAfter] == 0,
                                                 "H(" + bitmapTable[kValue + sumAfter].get(GRB_StringAttr_VarName) + ") - H(" + bitmapTable[sumAfter].get(GRB_StringAttr_VarName) + ") = 0");
        kConstraints.push_back(kConstraint3);
        if (Debug > 0)
        {
            outFile << "Additional K constraint: ";
            outFile << "I(";
            for (size_t i = 0; i < 2; i++)
            {
                outFile << elements[i];
                if (i != elements.size() - 1)
                    outFile << ",";
            }
            outFile << " and ";
            for (size_t i = 2; i < elements.size(); i++)
            {
                outFile << elements[i];
                if (i != elements.size() - 1)
                    outFile << ",";
            }
            outFile << ") ==" << kValue << endl;
            outFile << "H(" << sumBefore << ") + H(" << sumAfter << ") - H(" << sumBefore + sumAfter << ") = H(" << kValue << ")" << endl;
        }
    }
}

void checkAllMValues(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    auto solveTradeoff = [&](double cacheSize)
    {
        auto cacheConstraints = vector<GRBConstr>({});
        for (int i = 0; i < setup.numUser; i++)
        {
            auto constr = model.addConstr(bitmapTable[1 << (i + setup.numFile)] <= cacheSize, "MaximumCacheUser" + to_string(i));
            cacheConstraints.push_back(constr);
        }
        model.update();
        model.optimize();
        double load = model.get(GRB_DoubleAttr_ObjVal);
        for (auto constr : cacheConstraints)
        {
            model.remove(constr);
        }
        model.update();
        return load;
    };
    list<pair<double, double>> tradeoffs;
    if (setup.maxM == 0 && setup.minM == 0)
    {
        cout << "No M values provided, using default tradeoffs." << endl;
        tradeoffs.push_back({0.0, min(setup.numFile, setup.numUser)});
        tradeoffs.push_back({setup.numFile, 0});
    }
    else
    {
        double minR = solveTradeoff(setup.minM);
        double maxR = solveTradeoff(setup.maxM);
        cout << "Minimum R for M = " << setup.minM << ": " << minR << endl;
        cout << "Maximum R for M = " << setup.maxM << ": " << maxR << endl;
        tradeoffs.push_back({setup.minM, minR});
        tradeoffs.push_back({setup.maxM, maxR});
    }

    vector<list<pair<double, double>>::iterator> s;
    s.push_back(tradeoffs.begin());

    model.update();
    while (!s.empty())
    {
        auto start = chrono::high_resolution_clock::now();
        auto head = s.back();
        auto x1 = head->first;
        auto y1 = head->second;
        auto tail = next(head);
        auto x2 = tail->first;
        auto y2 = tail->second;

        if (fabs(x2 - x1) < 1e-1)
        {
            s.pop_back();
            continue;
        }
        double x3 = (x1 + x2) / 2;
        double y3 = solveTradeoff(x3);
        // cout << x1 << "," << x2 << "->";
        // cout << "(" << x3 << "," << y3 << ")" << endl;

        if (ColinearCheck(x1, y1, x2, y2, x3, y3) == 1)
        {
            s.pop_back();
            continue;
        }
        auto mid = tradeoffs.insert(tail, pair<double, double>(x3, y3));
        s.push_back(mid);
        auto end = chrono::high_resolution_clock::now();
        cout << "Tradeoff: " << x3 << "," << y3 << " Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() / (double)1000 << "s" << endl;
    }
    solutionFile << "Tradeoffs, N = " << setup.numFile << ", K = " << setup.numUser << ", Demands = ";
    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        solutionFile << "X";
        for (int j = 0; j < setup.numUser; j++)
        {
            solutionFile << setup.demands[i][j];
        }
        if (i != setup.demands.size() - 1)
            solutionFile << ",";
    }
    if (setup.commonInformationsStrings.size() > 0)
    {
        solutionFile << ", Common Informations = ";
        for (auto &group : setup.commonInformationsStrings)
        {
            solutionFile << "{";
            for (auto &element : group)
            {
                solutionFile << element << ",";
            }
            solutionFile << "}";
        }
    }
    solutionFile << endl;
    solutionFile << "[";
    for (auto v : tradeoffs)
    {
        solutionFile << v.first;
        if (v != tradeoffs.back())
            solutionFile << ",";
    }
    solutionFile << "]" << endl
                 << "[";
    for (auto v : tradeoffs)
    {
        solutionFile << v.second;
        if (v != tradeoffs.back())
            solutionFile << ",";
    }
    solutionFile << "]" << endl;
}

bool allIncluded(const std::vector<int> &subset, const std::vector<int> &superset)
{
    return std::all_of(subset.begin(), subset.end(), [&](int x)
                       { return std::find(superset.begin(), superset.end(), x) != superset.end(); });
}
void printSubset(const std::vector<int> &subset, Setup &setup)
{
    for (size_t j = 0; j < subset.size(); j++)
    {
        if (subset[j] < setup.numFile)
        {
            outFile << "W" << subset[j] << " ";
        }
        else if (subset[j] < setup.numUser + setup.numFile)
        {
            outFile << "Z" << subset[j] - setup.numFile << " ";
        }
        else if (subset[j] < setup.numUser + setup.numFile + setup.commonInformations)
        {
            outFile << "K" << subset[j] - setup.numFile - setup.numUser << " ";
        }
        else
        {
            outFile << "X";
            // Get values from setup.demands[subset[j]-setup.numFile-setup.numUser-setup.commonInformations]
            for (int k = 0; k < setup.numUser; k++)
            {
                outFile << setup.demands[subset[j] - setup.numFile - setup.numUser - setup.commonInformations][k];
            }
            outFile << " ";
        }
    }
    outFile << endl;
}

void addSubsetEqualityToModel(Setup &setup, vector<int> &subset,
                              GRBModel &model, GRBVar *bitmapTable)
{
    if (!setup.fullVariables)
    {
        uint64_t demandVal = 0;
        for (size_t j = 0; j < subset.size(); j++)
        {
            if (subset[j] < setup.numUser + setup.numFile + setup.commonInformations)
                demandVal += 1ULL << subset[j];
            else
                demandVal += ((subset[j] - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)));
        }
        model.addConstr(bitmapTable[demandVal] == setup.numFile, "H(" + bitmapTable[demandVal].get(GRB_StringAttr_VarName) + ") = " + to_string(setup.numFile));
    }
    else
    {
        uint64_t demandVal = 0;
        for (size_t j = 0; j < subset.size(); j++)
        {
            demandVal += 1ULL << subset[j];
        }
        model.addConstr(bitmapTable[demandVal] == setup.numFile, "H(" + bitmapTable[demandVal].get(GRB_StringAttr_VarName) + ") = " + to_string(setup.numFile));
    }
}

void defineMaxDimensionDemands(Setup &setup, GRBModel &model, GRBVar *bitmapTable)
{
    std::cout << "Defining max dimension demands" << std::endl;
    // A demand is max dimension if contains all files or using file + caches and setup.demands can create all the setup.demands
    vector<vector<int>> demandsWithMaxExtensionTemp;
    vector<vector<int>> subsets;
    vector<int> allVariables(setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());
    for (size_t i = 0; i < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); i++)
    {
        allVariables[i] = i;
    }
    // Generate all subsets of size 1 or more
    if (!setup.fullVariables)
    {
        for (size_t subsetSize = setup.numFile; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); subsetSize++)
        {
            vector<int> current;
            generateSubsets(allVariables, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations);
        }
    }
    else
    {
        for (size_t subsetSize = setup.numFile; subsetSize < setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size(); subsetSize++)
        {
            vector<int> current;
            generateSubsets(allVariables, subsetSize, 0, current, subsets, setup.numUser + setup.numFile + setup.commonInformations + setup.demands.size());
        }
    }
    for (size_t i = 0; i < subsets.size(); i++)
    {
        auto subset = subsets[i];
        vector<int> containedFiles;
        int value = 0;
        for (size_t j = 0; j < subset.size(); j++)
        {
            if (subset[j] < setup.numUser)
            {
                containedFiles.push_back(subset[j]);
            }
            else if (subset[j] < setup.numUser + setup.numFile)
            {
                for (size_t k = j + 1; k < subset.size(); k++)
                {
                    if (subset[k] >= setup.numUser + setup.numFile + setup.commonInformations)
                    {
                        containedFiles.push_back(setup.demands[subset[k] - setup.numUser - setup.numFile - setup.commonInformations][subset[j] - setup.numUser]);
                    }
                }
            }

            if (!setup.fullVariables)
            {
                if (subset[j] < setup.numUser + setup.numFile + setup.commonInformations)
                {
                    value += 1ULL << subset[j];
                }
                else
                {
                    value += ((subset[j] - setup.numUser - setup.numFile - setup.commonInformations + 1) * (1ULL << (setup.numUser + setup.numFile + setup.commonInformations)));
                }
            }
            else
            {
                value += 1ULL << subset[j];
            }
        }
        int numContainedFiles = 0;
        for (int j = 0; j < setup.numFile; j++)
        {
            // If the file is not in the subset, continue
            if (find(containedFiles.begin(), containedFiles.end(), j) == containedFiles.end())
            {
                continue;
            }
            numContainedFiles++;
        }
        if (numContainedFiles == setup.numFile)
        {
            model.addConstr(bitmapTable[value] == setup.numFile, "H(" + bitmapTable[value].get(GRB_StringAttr_VarName) + ") = " + to_string(setup.numFile));
            setup.maxValueSubsets[value] = 1;
        }

        /*
        //Add more rows, doesn't change time to resolve
        else{
            model.addConstr(bitmapTable[value] >= numContainedFiles, "demandMax");
        }
            */
    }
}

void renameVariables(GRBVar *bitmapTable, Setup &setup, int numDemands, int maxVar, GRBModel &model)
{
    if (!setup.fullVariables)
    {
        // Rename variables
        for (int i = 0; i < maxVar; i++)
        {
            std::string varName = "H(";
            for (int j = 0; j < setup.numUser + setup.numFile + setup.commonInformations; j++)
            {
                if (i & (1 << j))
                {
                    if (j < setup.numFile)
                    {
                        varName += "W" + to_string(j);
                    }
                    else if (j < setup.numFile + setup.numUser)
                    {
                        varName += "Z" + to_string(j - setup.numFile);
                    }
                    else if (j < setup.numFile + setup.numUser + setup.commonInformations)
                    {
                        varName += "K" + to_string(j - setup.numFile - setup.numUser);
                    }
                    else
                    {
                        varName += "X";
                        // Get values from setup.demands[j]-setup.numFile-setup.numUser-setup.commonInformations
                        for (int k = 0; k < setup.numUser; k++)
                        {
                            varName += to_string(setup.demands[j - setup.numFile - setup.numUser - setup.commonInformations][k]);
                        }
                    }
                    varName += ",";
                }
            }
            if (i >= (1 << (setup.numUser + setup.numFile + setup.commonInformations)))
            {
                int tempI = i;
                string tempString = "";
                for (int j = (int)numDemands; j > 0; j--)
                {
                    int demandValue = j * (1 << (setup.numUser + setup.numFile + setup.commonInformations));
                    if (tempI >= demandValue)
                    {
                        string newString = "X";
                        for (int k = 0; k < setup.numUser; k++)
                        {
                            newString += to_string(setup.demands[j - 1][k]);
                        }
                        newString += ",";
                        tempI -= demandValue;
                        tempString = newString + tempString;
                    }
                }
                varName += tempString;
            }
            // Remove last comma
            if (varName[varName.size() - 1] == ',')
                varName = varName.substr(0, varName.size() - 1);
            varName += ")";
            if (Debug > 0)
            {
                outFile << "Element: " << i << " Name: " << varName << endl;
            }
            bitmapTable[i].set(GRB_StringAttr::GRB_StringAttr_VarName, varName);
        }
    }
    else
    {
        // Create variables for all possible combinations of user,file,setup.commonInformations and setup.demands
        // Rename variables
        for (int i = 0; i < maxVar; i++)
        {
            std::string varName = "H(";
            for (int j = 0; j < int(setup.numUser + setup.numFile + setup.commonInformations + numDemands); j++)
            {
                if (i & (1 << j))
                {
                    if (j < setup.numFile)
                    {
                        varName += "W" + to_string(j);
                    }
                    else if (j < setup.numFile + setup.numUser)
                    {
                        varName += "Z" + to_string(j - setup.numFile);
                    }
                    else if (j < setup.numFile + setup.numUser + setup.commonInformations)
                    {
                        varName += "K" + to_string(j - setup.numFile - setup.numUser);
                    }
                    else
                    {
                        varName += "X";
                        // Get values from setup.demands[j]-setup.numFile-setup.numUser-setup.commonInformations
                        for (int k = 0; k < setup.numUser; k++)
                        {
                            varName += to_string(setup.demands[j - setup.numFile - setup.numUser - setup.commonInformations][k]);
                        }
                    }
                    varName += ",";
                }
            }
            // Remove last comma
            if (varName[varName.size() - 1] == ',')
                varName = varName.substr(0, varName.size() - 1);
            varName += ")";
            if (Debug > 0)
            {
                outFile << "Element: " << i << " Name: " << varName << endl;
            }
            bitmapTable[i].set(GRB_StringAttr::GRB_StringAttr_VarName, varName);
        }
    }
    cout << "Number of variables: " << model.get(GRB_IntAttr_NumVars) << endl;
    model.update();
}

void modelSetup(GRBModel &model)
{
    // Get macx omp threads
    model.set(GRB_IntParam_Threads, omp_get_max_threads());


    // Use barrier method for faster solution
    model.set(GRB_IntParam_Method, 2);    // 2 = Barrier method


    // Use dual simplex to be able to print active constraints
    // model.set(GRB_IntParam_Method, 1); // 1 = Dual Simple
    //printSolutions = true;

    model.set(GRB_IntParam_Crossover, 0); // Disable crossover
    model.set(GRB_IntParam_Presolve, 2);
    model.set(GRB_IntParam_Aggregate, 2);   // Enable aggregation
    model.set(GRB_IntParam_PreDual, 1);     // Dualize the model before presolving
    model.set(GRB_IntParam_PreSparsify, 1); // Reduce nonzeros in constraints
    model.set(GRB_IntParam_PreDepRow, 1);   // Remove dependent rows
    // model.getEnv().set(GRB_IntParam_OutputFlag, 0);
    //  set model precision
    model.set(GRB_DoubleParam_IntFeasTol, 1e-6);
    model.set(GRB_DoubleParam_FeasibilityTol, 1e-6);
    model.set(GRB_DoubleParam_MIPGap, 1e-6); // MIP gap tolerance set to 1e-6
}

void printSetup(const Setup &setup)
{
    if (Debug > 0)
    {
        outFile << "Setup: " << endl;
        outFile << "Number of users: " << setup.numUser << endl;
        outFile << "Number of files: " << setup.numFile << endl;
        outFile << "All Demands: " << setup.allDemands << endl;
        outFile << "All Demands Optimal: " << setup.allDemandsOpt << endl;
        outFile << "All Different Demands: " << setup.allDifferent << endl;
        outFile << "One Different Demand: " << setup.oneDifferent << endl;
        outFile << "Delete Permutations: " << setup.deletePerm << endl;
        outFile << "User Demands: " << setup.userDemands << endl;
        outFile << "All M values: " << setup.allMvalues << endl;
        outFile << "Cycle Demands: " << setup.cycleDemands << endl;
        outFile << "Full Variables: " << setup.fullVariables << endl;
        outFile << "Common Informations: " << setup.commonInformations << endl;
        outFile << "Common Informations File: " << setup.commonInformationsFile << endl;
        outFile << "Multi Common Informations: " << setup.multiCommonInformations << endl;
        outFile << "Common Informations Strings: " << endl;
        for (const auto &group : setup.commonInformationsStrings)
        {
            outFile << "{ ";
            for (const auto &element : group)
            {
                outFile << element << " ";
            }
            outFile << "} ";
        }
        outFile << endl;
        outFile << "Demand Requests: ";
        for (const auto &demand : setup.demand_requests)
        {
            outFile << demand << " ";
        }
        outFile << endl;
        outFile << "Max Value Subsets: ";
        for (const auto &value : setup.maxValueSubsets)
        {
            outFile << value << " ";
        }
        outFile << endl;
        outFile << "Demands: " << endl;
        for (const auto &demand : setup.demands)
        {
            outFile << "{ ";
            for (const auto &element : demand)
            {
                outFile << element << " ";
            }
            outFile << "} ";
        }
        outFile << endl;
        outFile << "M value: " << setup.M << endl;
        outFile << "M min Value: " << setup.minM << endl;
        outFile << "M max Value: " << setup.maxM << endl;
        outFile << "Solution File: " << setup.solutionFile << endl;
        outFile << "Debug Level: " << setup.Debug << endl;
        outFile << "SymmetryCheck: " << endl;
        for (const auto &symmetry : setup.symmetryCheck)
        {
            outFile << symmetry << " ";
        }
        outFile << endl;
        outFile.flush();
    }
}

void addModelConstraints(Setup &setup, GRBModel &model, GRBVar *bitmapTable, chrono::high_resolution_clock::time_point &startChrono,
                         int maxVar, int numSingleVar)
{
    // Add Objective function
    auto r = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "load");
    GRBLinExpr obj = r;
    model.setObjective(obj, GRB_MINIMIZE);

    setup.symmetryCheck.resize(maxVar, 0);
    setup.maxValueSubsets.resize(maxVar, 0);
    startChrono = chrono::high_resolution_clock::now();

    defineMaxDimensionDemands(setup, model, bitmapTable);
    model.update();
    cout << " Number of constraints after defining max dimension setup.demands: " << model.get(GRB_IntAttr_NumConstrs) << endl;

    auto endChrono = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endChrono - startChrono);

    if (setup.deletePerm)
    {
        // outFile << "Deleting symmetric permutations" << endl;
        auto start = chrono::high_resolution_clock::now();
        auto filePerms = vector<vector<int>>();
        auto userPerms = vector<vector<int>>();
        createUserFilePermutations(setup, userPerms, filePerms);

        std::unordered_set<std::vector<int>, VectorHash> allPermutations;
        cout << "Creating permutations" << endl;
        createTablePermutations(setup, filePerms, userPerms, allPermutations, numSingleVar);
        // print allPermutations
        cout << "Permutations created: " << allPermutations.size() << endl;
        // TODO add to Gurobi constraints
        addSymmetricDemandsConstraints(allPermutations, setup, model, bitmapTable);
        model.update();
        cout << " Number of constraints after adding symmetric setup.demands constraints: " << model.get(GRB_IntAttr_NumConstrs) << endl;
        cout << "Adding symmetric user-file constraints" << endl;
        // print symmetry check
        if (Debug > 0)
        {
            outFile << "Symmetry Check: ";
            for (size_t i = 0; i < setup.symmetryCheck.size(); i++)
            {
                outFile << setup.symmetryCheck[i] << " \n";
            }
            outFile << endl;
        }
        addSymmetricUserFileConstraints(setup, model, bitmapTable);
        model.update();
        cout << " Number of constraints after adding symmetric user-file constraints: " << model.get(GRB_IntAttr_NumConstrs) << endl;
        auto end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        // outFile << "Time: " << duration.count() << "ms" << endl;
    }
    // print symmetry check
    if (Debug > 0)
    {
        outFile << "Symmetry Check: ";
        for (size_t i = 0; i < setup.symmetryCheck.size(); i++)
        {
            outFile << setup.symmetryCheck[i] << " \n";
        }
        outFile << endl;
    }

    // outFile << "Time: " << duration.count() << "ms" << endl;
    outFile << "Adding basic problem constraints" << endl;
    addBasicProblemConstraints(setup, model, bitmapTable, r);
    model.update();
    cout << " Number of constraints after adding basic problem constraints: " << model.get(GRB_IntAttr_NumConstrs) << endl;
    outFile << "Adding Shannon type constraints" << endl;
    addShannonTypeConstraints(setup, model, bitmapTable);
    model.update();
    cout << " Number of constraints after adding Shannon type constraints: " << model.get(GRB_IntAttr_NumConstrs) << endl;
    // addNonShannonTypeConstraints(setup.numUser, setup.numFile, setup.demands,model,bitmapTable,setup.commonInformations);
    endChrono = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(endChrono - startChrono);
    // outFile << "Time: " << duration.count() << "ms" << endl;
    model.update();
}

void checkDefinedMValue(Setup &setup, GRBModel &model, GRBVar *bitmapTable, int i, vector<string> &commonInformationsValueStrings,
                        vector<pair<double, string>> &tradeoffs, string &timestamp)
{
    // Adding constraints H(Z_i) <= setup.M for all i
    std::vector<GRBConstr> userCacheConstraints; // Store the constraints
    userCacheConstraints.reserve(setup.numUser); // Reserve space for efficiency
    std::cout << "Setting up persistent cache constraints..." << std::endl;
    for (int i = 0; i < setup.numUser; ++i)
    {
        // Add constraint with a placeholder RHS (e.g., 0.0 or an initial guess)
        // Ensure bitmapTable access is correct and returns a GRBVar or valid expression part
        auto constr = model.addConstr(bitmapTable[1 << (i + setup.numFile)] <= setup.M, "cache_user_" + std::to_string(i));
        userCacheConstraints.push_back(constr);
    }

    model.update();
    auto startOpt = chrono::high_resolution_clock::now();
    model.optimize();
    // model.write("model.mps");
    auto endOpt = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endOpt - startOpt);
    // outFile << "Optimization time: " << duration.count() << "ms" << endl;
    // outFile << "Objective value: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << "Objective value: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

    if(printSolutions)
    {
        model.write("solution.json");
        model.write("model.lp");
    }

    // --- Get results ---
    int optimstatus = model.get(GRB_IntAttr_Status);

    if (optimstatus == GRB_OPTIMAL)
    {
        std::cout << "Optimal solution found." << std::endl;
        solutionFile << "Tradeoffs, N = " << setup.numFile << ", K = " << setup.numUser << ", Demands = ";
        for (size_t i = 0; i < setup.demands.size(); i++)
        {
            solutionFile << "X";
            for (int j = 0; j < setup.numUser; j++)
            {
                solutionFile << setup.demands[i][j];
            }
            if (i != setup.demands.size() - 1)
                solutionFile << ",";
        }
        if (setup.commonInformationsStrings.size() > 0)
        {
            solutionFile << ", Common Informations = ";
            for (auto &group : setup.commonInformationsStrings)
            {
                solutionFile << "{";
                for (auto &element : group)
                {
                    solutionFile << element << ",";
                }
                solutionFile << "}";
            }
        }
        solutionFile << endl;
        solutionFile << "[";
        solutionFile << setup.M;
        solutionFile << "]" << endl
                     << "[";
        solutionFile << model.get(GRB_DoubleAttr_ObjVal);
        solutionFile << "]" << endl;

        
        if(printSolutions)
        {
            std::cout << "\n--- Constraint Information ---" << std::endl;

            // Get all constraints from the model
            GRBConstr *constraints = model.getConstrs();
            int num_constraints = model.get(GRB_IntAttr_NumConstrs);
            // ofstream outputSolutionFile("solutionOutput.txt");
            // outputSolutionFile << "Solution: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            int counter = 0;
            int counterBasisOne = 0;
            int counterBasisMinusOne = 0;
            int counterBasisZero = 0;
            for (int i = 0; i < num_constraints; ++i)
            {
                GRBConstr current_constr = constraints[i];
                std::string constr_name = current_constr.get(GRB_StringAttr_ConstrName);
                double slack = current_constr.get(GRB_DoubleAttr_Slack);
                double tolerance = 1e-9; // Or get model.get(GRB_DoubleParam_FeasibilityTol);

                if (std::abs(slack) < tolerance)
                {
                    counter++;
                }
                if (current_constr.get(GRB_IntAttr_CBasis) == 1)
                {
                    counterBasisOne++;
                }
                if (current_constr.get(GRB_IntAttr_CBasis) == -1)
                {
                    counterBasisMinusOne++;
                    //outputSolutionFile<< "Constraint '" << current_constr.get(GRB_StringAttr_ConstrName)<< "' is binding (active)." << std::endl;
                    //print constraint expression
                    GRBLinExpr expr = model.getRow(current_constr);
                    outputSolutionFile << "Constraint: ";
                    for (int j = 0; j < expr.size(); j++)
                    {
                        double coeff = expr.getCoeff(j);
                        std::string varName = expr.getVar(j).get(GRB_StringAttr_VarName);
                        outputSolutionFile << coeff << "*" << varName << " ";
                        if (j < expr.size() - 1)
                            outputSolutionFile << "+ ";
                    }
                    char sense = current_constr.get(GRB_CharAttr_Sense);
                    double rhs = current_constr.get(GRB_DoubleAttr_RHS);
                    outputSolutionFile << " " << sense << " " << rhs;
                    outputSolutionFile << std::endl;
                }
                if (current_constr.get(GRB_IntAttr_CBasis) == 0)
                {
                    counterBasisZero++;
                }
            }
            cout << "Number of binding constraints: " << counter << endl;
            cout << "Number of constraints with basis 1: " << counterBasisOne << endl;
            cout << "Number of constraints with basis -1: " << counterBasisMinusOne << endl;
            cout << "Number of constraints with basis 0: " << counterBasisZero << endl;

            // Important: Gurobi manages the memory for the array returned by getConstrs().
            // Do NOT delete[] constraints;
            // However, it's good practice to set the pointer to null after use if needed.
            constraints = nullptr;
        }
    }
    else if (optimstatus == GRB_INF_OR_UNBD)
    {
        std::cout << "Model is infeasible or unbounded" << std::endl;
    }
    else if (optimstatus == GRB_INFEASIBLE)
    {
        std::cout << "Model is infeasible" << std::endl;
        // You might want to compute an IIS here if needed
    }
    else if (optimstatus == GRB_UNBOUNDED)
    {
        std::cout << "Model is unbounded" << std::endl;
    }
    else
    {
        std::cout << "Optimization was stopped with status: " << optimstatus << std::endl;
    }

    // delete modelcache constraints
    if (setup.multiCommonInformations)
    {
        cout << "Adding tradeoffs" << endl;
        if (i >= (int)commonInformationsValueStrings.size())
        {
            cerr << "Index out of bounds: i = " << i << endl;
            cout << "Deleting user cache constraints" << endl;
            for (int i = 0; i < setup.numUser; ++i)
            {
                model.remove(userCacheConstraints[i]);
            }
            return;
        }

        if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
        {
            cerr << "Model not optimized successfully!" << endl;
            cout << "Deleting user cache constraints" << endl;
            for (int i = 0; i < setup.numUser; ++i)
            {
                model.remove(userCacheConstraints[i]);
            }
            return;
        }

        double load = model.get(GRB_DoubleAttr_ObjVal);
        cout << "Load: " << load << endl;

        pair<double, string> tradeoff = {load, commonInformationsValueStrings[i]};
        cout << "Tradeoff: " << tradeoff.first << " " << tradeoff.second << endl;
        tradeoffs.push_back(tradeoff);
        cout << "Deleting user cache constraints" << endl;
        for (int i = 0; i < setup.numUser; ++i)
        {
            model.remove(userCacheConstraints[i]);
        }
        ofstream outputFileCommonInformations("solutionCommonInformations" + timestamp + ".txt", ios::app);
        outputFileCommonInformations << "Tradeoff " << i << ": " << tradeoffs[i].first << " " << tradeoffs[i].second << endl;
    }
}

void mainLoop(Setup &setup, GRBModel &model, GRBVar *bitmapTable, vector<string> &commonInformationsValueStrings,
              int numberOfIterations, string &timestamp)
{
    vector<pair<double, string>> tradeoffs;
    for (int i = 0; i < numberOfIterations; i++)
    {
        if (setup.multiCommonInformations)
        {
            setup.commonInformationsStrings.clear();
            // Divide the line in multiple string based on :
            string line = commonInformationsValueStrings[i];
            // Divide the line into values divided by :
            std::istringstream iss(line);
            std::string value;
            std::vector<std::string> group;
            while (getline(iss, value, ':'))
            {
                group.push_back(value);
            }
            cout << "Group: " << endl;
            for (size_t j = 0; j < group.size(); j++)
            {
                cout << group[j] << "\n";
            }
            ParsedExpression parsedExpressions = parseCommonInformations(group);
            setup.commonInformationsStrings = parsedExpressions.groups;
            cout << "Parsed expressions size: " << parsedExpressions.groups.size() << endl;
            for (size_t j = 0; j < parsedExpressions.groups.size(); j++)
            {
                cout << "Parsed expression " << j << ": ";
                for (size_t k = 0; k < parsedExpressions.groups[j].size(); k++)
                {
                    cout << parsedExpressions.groups[j][k] << " ";
                }
                cout << endl;
            }
            checkK(setup);
            cout << "Common Informations: " << endl;
            for (size_t j = 0; j < setup.commonInformationsStrings.size(); j++)
            {
                cout << "Common Information " << j << ": ";
                for (size_t k = 0; k < setup.commonInformationsStrings[j].size(); k++)
                {
                    cout << setup.commonInformationsStrings[j][k] << " ";
                }
                cout << endl;
            }
        }
        std::vector<GRBConstr> commonInformationsConstraints; // Store the constraints
        if (setup.commonInformations > 0)
        {
            cout << "Adding additional K constraints" << endl;
            addKConstraints(setup, model, bitmapTable, commonInformationsConstraints);
            model.update();
            cout << " Number of constraints after adding additional K constraints: " << model.get(GRB_IntAttr_NumConstrs) << endl;
        }
        if (setup.allMvalues)
        {
            auto start = chrono::high_resolution_clock::now();
            checkAllMValues(setup, model, bitmapTable);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            // outFile << "Time: " << duration.count() << "ms" << endl;
        }
        else
        {
            checkDefinedMValue(setup, model, bitmapTable, i, commonInformationsValueStrings, tradeoffs, timestamp);
        }
        // delete common informations constraints
        if (setup.multiCommonInformations)
        {
            cout << "Deleting common informations constraints" << endl;
            for (auto &constr : commonInformationsConstraints)
            {
                model.remove(constr);
            }
            model.update();
            cout << "Deleted common informations constraints" << endl;
        }
    }
    // Sort tradeoffs by first element from highest to lowest
    if (setup.multiCommonInformations && !tradeoffs.empty())
    {
        std::sort(tradeoffs.begin(), tradeoffs.end(), [](const std::pair<double, string> &a, const std::pair<double, string> &b)
                  {
                      return a.first > b.first; // Sort in descending order based on the first element
                  });
        // Print tradeoffs
        cout << "Tradeoffs: " << endl;
        int maxTradeoff = 25;
        if ((int)tradeoffs.size() < maxTradeoff)
            maxTradeoff = tradeoffs.size();
        string fileName = "solutionCommonInformations" + to_string(setup.numUser) + "U_" + to_string(setup.numFile) + "F" +
                          to_string(setup.commonInformations) + "K" + timestamp + ".txt";
        ofstream outputFileCommonInformations(fileName, ios::app);
        for (int i = 0; i < maxTradeoff; i++)
        {
            cout << "Tradeoff " << i << ": " << tradeoffs[i].first << " " << tradeoffs[i].second << endl;
            outputFileCommonInformations << tradeoffs[i].second << "\n";
        }
        for (size_t i = 0; i < tradeoffs.size(); i++)
        {
            outputFileCommonInformations << "Tradeoff " << i << ": " << tradeoffs[i].first << " " << tradeoffs[i].second << endl;
        }
        outputFileCommonInformations.close();
        // Delete temp file
        string tempFileName = "solutionCommonInformations" + timestamp + ".txt";
        if (remove(tempFileName.c_str()) != 0)
        {
            cerr << "Error deleting temporary file: " << tempFileName << endl;
        }
        else
        {
            cout << "Temporary file deleted: " << tempFileName << endl;
        }
    }
}

string getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm); // thread-safe on POSIX; for Windows, use localtime_s(&tm, &t);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm); // Correct format
    return std::string(buffer);
}

int main(int argc, char *argv[])
{
    auto initialStartingTime = chrono::high_resolution_clock::now();
    string timestamp = getTimestamp();
    outFile = ofstream("output.txt");
    Setup setup = parseArgs(argc, argv);
    if (!setup.solutionFile.empty())
        solutionFile = ofstream(setup.solutionFile, std::ios::app);
    else
    {
        if (mkdir("Results", 0777) == -1 && errno != EEXIST)
        {
            cerr << "Error creating Results directory: " << strerror(errno) << endl;
            return 1; // Exit or handle error as needed
        }
        solutionFile = ofstream("Results/solution.txt", std::ios::app);
    }
    Debug = setup.Debug;

    auto initialTime = chrono::high_resolution_clock::now();
    auto startChrono = chrono::high_resolution_clock::now();
    if (!setup.userDemands)
        createDemands(setup);
    else
        cout << "User setup.demands" << endl;
    auto numDemands = setup.demands.size();
    // print setup.demands
    cout << "setup.demands: " << endl;
    for (size_t i = 0; i < setup.demands.size(); i++)
    {
        cout << "Demand " << i << ": ";
        for (size_t j = 0; j < setup.demands[i].size(); j++)
        {
            cout << setup.demands[i][j] << " ";
        }
        cout << endl;
    }
    auto numSingleVar = setup.numUser + setup.numFile + setup.commonInformations + numDemands;
    printSetup(setup);

    try
    {
        auto env = GRBEnv();
        GRBModel model = GRBModel(env);
        GRBVar *bitmapTable = nullptr;
        int maxVar = 0;
        if (!setup.fullVariables)
        {
            maxVar = ((int)setup.demands.size() + 1) * (1 << (setup.numUser + setup.numFile + setup.commonInformations));
            bitmapTable = model.addVars(maxVar, GRB_CONTINUOUS);
        }
        else
        {
            maxVar = 1ULL << (setup.numUser + setup.numFile + setup.commonInformations + numDemands);
            bitmapTable = model.addVars(maxVar, GRB_CONTINUOUS);
        }

        renameVariables(bitmapTable, setup, numDemands, maxVar, model);

        modelSetup(model);

        addModelConstraints(setup, model, bitmapTable, startChrono, maxVar, numSingleVar);

        auto finalTime = chrono::high_resolution_clock::now();
        auto durationFinal = chrono::duration_cast<chrono::milliseconds>(finalTime - initialStartingTime);
        cout << "Total time: " << durationFinal.count() << "ms" << endl;

        int numberOfIterations = 1;
        // Read common informations from file
        ifstream commonInformationsFileStream;
        vector<string> commonInformationsValueStrings;
        if (setup.multiCommonInformations)
        {
            numberOfIterations = setup.commonInformations;
            commonInformationsFileStream.open(setup.commonInformationsFile);
            if (!commonInformationsFileStream.is_open())
            {
                std::cerr << "Error opening file: " << setup.commonInformationsFile << std::endl;
                return 1; // Exit or handle error as needed
            }
            // Divide the file into lines
            string line;
            while (getline(commonInformationsFileStream, line))
            {
                while (line.back() == '\n' || line.back() == '\r')
                {
                    line.pop_back();
                }
                commonInformationsValueStrings.push_back(line);
            }
            numberOfIterations = commonInformationsValueStrings.size();
            cout << "Number of iterations: " << numberOfIterations << endl;
            cout << "Common informations values: " << endl;
            for (size_t i = 0; i < commonInformationsValueStrings.size(); i++)
            {
                cout << "Common Information " << i << ": " << commonInformationsValueStrings[i] << endl;
            }
        }
        cout << " Number of strings: " << commonInformationsValueStrings.size() << endl;

        mainLoop(setup, model, bitmapTable, commonInformationsValueStrings, numberOfIterations, timestamp);
    }
    catch (GRBException e)
    {
        cerr << "Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
    }
    catch (...)
    {
        cerr << "Exception during optimization" << endl;
    }
    auto finalTime = chrono::high_resolution_clock::now();
    auto durationFinal = chrono::duration_cast<chrono::milliseconds>(finalTime - initialStartingTime);
    cout << "Total time: " << durationFinal.count() << "ms" << endl;
    return 0;
}