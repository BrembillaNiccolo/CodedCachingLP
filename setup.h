#ifndef SETUP_H
#define SETUP_H

#include <string>
#include <vector>

using std::string;
using std::vector;

struct EqVar{
    int variable;
    int main_symmetry;
    int final_position;
};


struct Setup
{
    int numUser = 0;
    int numFile = 0;
    bool fullVariables = false;
    string solutionFile;
    int Debug = 0;

    bool allDemands = false;
    bool allDemandsOpt = false;
    bool allDifferent = false;
    bool oneDifferent = false;
    bool userDemands = false;
    bool cycleDemands = false;

    bool deletePerm = false;

    int commonInformations = 0;
    string commonInformationsFile = "";
    bool multiCommonInformations = false;
    vector<vector<string>> commonInformationsStrings;
    
    bool allMvalues = false;
    double M = 0;
    double minM = 0;
    double maxM = 0;
    
    vector<int> demand_requests;
    vector<vector<int>> demands;
    vector<int> symmetryCheck;
    vector<int> maxValueSubsets;
    vector<EqVar> equalVariables;
};

struct ParsedExpression
{
    int commonInformations;
    std::vector<std::vector<std::string>> groups; // Each inner vector contains elements of a group
};

#endif // SETUP_H
