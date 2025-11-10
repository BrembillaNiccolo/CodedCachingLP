#include "parser.h"
#include "setup.h"
using namespace std;

bool isNumber(const string &s)
{
    istringstream iss(s);
    double d;
    return (iss >> d) && (iss.eof());
}

// Function to trim spaces from a string
string trim(const string &str)
{
    string s = str;
    s.erase(remove_if(s.begin(), s.end(), ::isspace), s.end());
    return s;
}

// Function to parse a single {a,b,c} demand
vector<int> parseSingleDemand(const string &str)
{
    if (str.front() != '{' || str.back() != '}')
        return {};
    string content = str.substr(1, str.size() - 2); // Remove { and }
    vector<int> values;
    istringstream ss(content);
    string token;

    while (getline(ss, token, ','))
    {
        token = trim(token); // Remove spaces
        if (!isNumber(token))
            return {}; // Check valid number
        values.push_back(stoi(token));
    }
    return values;
}

// Function to parse the outer {{...},{...}} structure
vector<vector<int>> parseDemands(const string &str)
{
    if (str.front() != '{' || str.back() != '}')
        return {};
    string content = str.substr(1, str.size() - 2); // Remove outer {}
    vector<vector<int>> demands;

    istringstream ss(content);
    string token;
    int depth = 0;
    string currentDemand;
    for (char c : content)
    {
        if (c == '{')
        {
            if (depth == 0)
                currentDemand.clear(); // Start new demand
            depth++;
            currentDemand += c;
        }
        else if (c == '}')
        {
            currentDemand += c;
            if (depth == 1)
            {
                auto demand = parseSingleDemand(currentDemand);
                if (demand.empty())
                    return {}; // Invalid format
                demands.push_back(demand);
            }
            depth--;
        }
        else
        {
            currentDemand += c;
        }
    }

    return demands;
}

/*
    Parse the content inside brackets
    Parse the content inside brackets in the format {a,b,c}
*/
vector<string> parseBracketsContent(string str)
{
    if (str.front() != '{' || str.back() != '}')
        return {};

    string content = str.substr(1, str.size() - 2); // Remove { and }
    vector<string> values;
    istringstream ss(content);
    string token;

    while (getline(ss, token, ','))
    {
        // Trim spaces from each number
        token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
        values.push_back(token);
    }
    return values;
}

// Function to extract X{...} values as a single concatenated string without commas
string parseXValues(const string &str)
{
    string result;
    regex xPattern(R"(X\{([\d,]+)\})");
    smatch match;

    if (regex_search(str, match, xPattern))
    {
        result = "X" + match[1].str(); // Concatenate all numbers inside X{} without commas
    }
    return result;
}

/*
    Parse the part of the common informations
    Parse the part of the common informations in the format Z{...} or W{...}
    Parse the part of the common informations in the format X{...}
*/
vector<string> parsePart(const string &part)
{
    vector<string> group;
    stringstream ss(part);
    string token;
    string tempPart = part;
    // if Z or W get the value until , if X get the value until }
    while (tempPart.size() > 0)
    {
        if (tempPart[0] == 'Z' || tempPart[0] == 'W')
        {
            token = tempPart.substr(1, tempPart.find(",") - 1);
            group.push_back(tempPart[0] + token);
            if (tempPart.find(",") == string::npos)
            {
                break;
            }
            tempPart = tempPart.substr(tempPart.find(",") + 1);
        }
        else if (tempPart[0] == 'X')
        {
            token = tempPart.substr(0, tempPart.find("}") + 1);
            token = parseXValues(token);
            group.push_back(token);
            if (tempPart.find("}") == string::npos)
            {
                break;
            }
            tempPart = tempPart.substr(tempPart.find("}") + 1);
            if (tempPart.size() > 0 && tempPart[0] == ',')
            {
                tempPart = tempPart.substr(1);
            }
        }
        else
        {
            cerr << "Error: Invalid format for -commonInformations (Not W,Z or X). Use -commonInformations <num> <string> ... <string>.\n";
            exit(1);
        }
    }
    return group;
}

/*
    Parse the common informations
    Parse the common informations in the format {X{...} and Z{...} and W{...}}
    Parse the common informations in the format {X{...} and Z{...}}
    Parse the common informations in the format {X{...} and W{...}}
    Parse the common informations in the format {Z{...} and W{...}}
    Parse the common informations in the format {X{...}}
    Parse the common informations in the format {Z{...}}
    Parse the common informations in the format {W{...}}
*/
ParsedExpression parseCommonInformations(vector<string> &args)
{
    ParsedExpression result;

    for (auto &arg : args)
    {
        // Split the expression into two parts by "and"
        if (arg.front() == '{' && arg.back() == '}')
        {
            arg = arg.substr(1, arg.size() - 2); // Remove outer {}

            size_t andPos = arg.find(" and ");
            if (andPos != string::npos)
            {
                string beforeAnd = arg.substr(0, andPos); // Part before "and"
                string afterAnd = arg.substr(andPos + 5); // Part after "and"

                // Parse before and after "and"
                vector<string> beforeGroup = parsePart(beforeAnd);
                vector<string> afterGroup = parsePart(afterAnd);

                // Validate that before has 2 elements, and after has 1 or 2 elements
                if (beforeGroup.size() >= 2 && afterGroup.size() >= 1)
                {
                    vector<string> group = beforeGroup;
                    group.push_back("and");
                    group.insert(group.end(), afterGroup.begin(), afterGroup.end());

                    result.groups.push_back(group);
                }
                else
                {
                    cerr << "Error: Invalid number of elements before or after 'and'.\n";
                    exit(1);
                }
            }
        }
    }

    return result;
}

/*
    Check if the values of X are valid
    Check if the values of Z are valid
    Check if the values of W are valid
    Check if the number of elements in X is equal to numUser
    Check if the number of elements in Z is equal to 1
    Check if the number of elements in W is equal to 1
*/
void checkK(Setup &setup)
{
    if (setup.commonInformationsStrings.size() == 0)
    {
        return;
    }
    for (auto &group : setup.commonInformationsStrings)
    {
        for (auto &element : group)
        {
            if (element[0] == 'X')
            {
                // from element[2] divide by ,
                string temp = element.substr(1);
                string token;
                istringstream ss(temp);
                int count = 0;
                while (getline(ss, token, ','))
                {
                    if (!isNumber(token))
                    {
                        cerr << "Error: Invalid number format inside -commonInformations.\n";
                        exit(1);
                    }
                    if (stoi(token) >= setup.numFile)
                    {
                        cerr << "Error: Invalid number format inside -commonInformations. Number must be less than numFile.\n";
                        exit(1);
                    }
                    count++;
                }
                if (count != setup.numUser)
                {
                    cerr << "Error: Invalid number of elements in X inside -commonInformations. Number of elements must be equal to numUser.\n";
                    exit(1);
                }
            }
            else if (element[0] == 'Z')
            {
                string temp = element.substr(1);
                if (!isNumber(temp))
                {
                    cerr << "Error: Invalid number format inside -commonInformations.\n";
                    exit(1);
                }
                if (stoi(temp) >= setup.numUser)
                {
                    cerr << "Error: Invalid number format inside -commonInformations. Number must be less than numUser.\n";
                    exit(1);
                }
            }
            else if (element[0] == 'W')
            {
                string temp = element.substr(1);
                if (!isNumber(temp))
                {
                    cerr << "Error: Invalid number format inside -commonInformations.\n";
                    exit(1);
                }
                if (stoi(temp) >= setup.numFile)
                {
                    cerr << "Error: Invalid number format inside -commonInformations. Number must be less than numFile.\n";
                    exit(1);
                }
            }
            else if (element != "and")
            {
                cerr << "Error: Invalid format for -commonInformations (not and). Use -commonInformations <num> <string> ... <string>.\n";
                exit(1);
            }
        }
    }
}

/*
    Parse the arguments of the program
    -nUser <numUser> number of users
    -nFile <numFile> number of files
    -alldemands use all demands
    -alldifferent use all different demands (each user has a different file request)
    -onedifferent use one different demand (use only one demand in which all requests are different)
    -deletePerm delete permutations (Delete permutations as for Tian et al. 2019 Paper)
    -debug <debug> debug level (0 only time, 1 print basic constraints and info, 2 print all constraints,permutations and info)
*/

Setup parseArgs(int argc, char *argv[])
{
    Setup setup;
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        cout << "Parsing argument: " << arg << endl; // Debugging output
        if (arg == "-nUser" && i + 1 < argc)
        {
            setup.numUser = atoi(argv[++i]);
        }
        if (arg == "-nFile" && i + 1 < argc)
        {
            setup.numFile = atoi(argv[++i]);
        }
        if (arg == "-alldemands")
        {
            setup.allDemands = true;
        }
        if (arg == "-alldemandsOpt")
        {
            setup.allDemandsOpt = true;
        }
        if (arg == "-alldifferent")
        {
            setup.allDifferent = true;
            // Check next argument as "{int,int,...}" as number of int equal to numFile and all int = numUser
            if (i + 1 < argc)
            {
                setup.demand_requests = parseSingleDemand(argv[++i]);
                // sort highest to lowest
                sort(setup.demand_requests.begin(), setup.demand_requests.end(), greater<int>());
                if (setup.demand_requests.size() != (size_t)setup.numUser)
                {
                    cerr << "Error: Number of elements in demand must be equal to number of users.\n";
                    exit(1);
                }
                // The sum of all elements must be equal to numUser
                int sum = 0;
                for (int num : setup.demand_requests)
                {
                    sum += num;
                }
                if (sum != setup.numUser)
                {
                    cerr << "Error: Sum of all elements in demand must be equal to number of users.\n";
                    exit(1);
                }
            }
        }
        if (arg == "-onedifferent")
        {
            setup.oneDifferent = true;
            // Check next argument as "{int,int,...}" as number of int equal to numFile and all int = numUser
            if (i + 1 < argc)
            {
                setup.demand_requests = parseSingleDemand(argv[++i]);
                // sort highest to lowest
                sort(setup.demand_requests.begin(), setup.demand_requests.end(), greater<int>());
                if (setup.demand_requests.size() != (size_t)setup.numUser)
                {
                    cerr << "Error: Number of elements in demand must be equal to number of users.\n";
                    exit(1);
                }
                int sum = 0;
                for (int num : setup.demand_requests)
                {
                    sum += num;
                }
                if (sum != setup.numUser)
                {
                    cerr << "Error: Sum of all elements in demand must be equal to number of users.\n";
                    exit(1);
                }
            }
        }
        if (arg == "-fullVariables")
        {
            setup.fullVariables = true;
        }
        if (arg == "-cycleDemands")
        {
            setup.cycleDemands = true;
        }
        if (arg == "-deletePerm")
        {
            setup.deletePerm = true;
        }
        if (arg == "-debug")
        {
            setup.Debug = atoi(argv[++i]);
        }
        if (arg == "-M" && i + 1 < argc)
        {
            vector<string> values = parseBracketsContent(argv[++i]);

            if (values.empty() || values.size() > 2)
            {
                cerr << "Error: Invalid format for -M. Use -M {numerator,denominator} or -M {numerator}.\n";
                exit(1);
            }

            // Check if values are numbers
            for (const auto &v : values)
            {
                if (!isNumber(v))
                {
                    cerr << "Error: Invalid number format inside -M.\n";
                    exit(1);
                }
            }
            

            // Convert and compute M
            double numerator = stod(values[0]);
            double denominator = (values.size() == 2) ? stod(values[1]) : 1.0;

            if (denominator == 0.0)
            {
                cerr << "Error: Denominator cannot be zero.\n";
                exit(1);
            }

            setup.M = numerator / denominator;
        }
        if (std::string(argv[i]) == "-demands" && i + 1 < argc)
        {
            setup.userDemands = true;
            setup.demands = parseDemands(argv[++i]);

            if (setup.demands.empty())
            {
                std::cerr << "Error: Invalid format for -demands. Use -demands \"{{a,b,...},{c,d,...},...}\".\n";
                exit(1);
            }

            // Validate each demand
            for (const auto &demand : setup.demands)
            {
                for (int num : demand)
                {
                    if (num >= setup.numFile || num < 0)
                    {
                        std::cerr << "Error: Demand number out of range. Must be 0 <= num < " << setup.numFile << ".\n";
                        exit(1);
                    }
                }
                // Number of elements in demand must be equal to number of users
                if (demand.size() != (size_t)setup.numUser)
                {
                    cerr << "Error: Number of elements in demand must be equal to number of users.\n";
                    exit(1);
                }
            }

            // Validate number of demands
            if (setup.demands.size() > pow(setup.numUser, setup.numFile))
            {
                std::cerr << "Error: Number of demands exceeds limit (" << pow(setup.numUser, setup.numFile) << ").\n";
                exit(1);
            }
            // If 2 demands are the same, remove one
            for (size_t k = 0; k < setup.demands.size(); k++)
            {
                for (size_t j = k + 1; j < setup.demands.size(); j++)
                {
                    if (setup.demands[k] == setup.demands[j])
                    {
                        setup.demands.erase(setup.demands.begin() + j);
                        j--;
                    }
                }
            }
        }
        if (arg == "-allMValues")
        {
            setup.allMvalues = true;
            vector<string> minValues = parseBracketsContent(argv[++i]);
            vector<string> maxValues = parseBracketsContent(argv[++i]);

            if (maxValues.empty() || maxValues.size() > 2 || minValues.empty() || minValues.size() > 2)
            {
                setup.minM = 0;
                setup.maxM = 0;
                i -=2; // Reset i to reprocess the arguments
            }
            else{
                int numeratorMax = stoi(maxValues[0]);
                int denominatorMax = (maxValues.size() == 2) ? stoi(maxValues[1]) : 1;
                int numeratorMin = stoi(minValues[0]);
                int denominatorMin = (minValues.size() == 2) ? stoi(minValues[1]) : 1;
                if (denominatorMax == 0 || denominatorMin == 0)
                {
                    cerr << "Error: Denominator cannot be zero.\n";
                    exit(1);
                }
                setup.maxM = static_cast<double>(numeratorMax) / denominatorMax;
                setup.minM = static_cast<double>(numeratorMin) / denominatorMin;
                if (setup.maxM < setup.minM || setup.minM < 0 || setup.maxM > min(setup.numUser, setup.numFile))
                {
                    cerr << "Error: maxM cannot be less than minM.\n";
                    exit(1);
                }
            }
        }
        if (arg == "-commonInformations" && i + 2 < argc)
        {
            setup.commonInformations = atoi(argv[++i]);
            for (int j = 0; j < setup.commonInformations; j++)
            {
                if (argv[i + j + 1][0] == '-')
                {
                    cerr << "Error: Invalid format for -commonInformations (use of - after). Use -commonInformations <num> <string> ... <string>.\n";
                    exit(1);
                }
            }
            vector<string> args;
            for (int j = 0; j < setup.commonInformations; j++)
            {
                args.push_back(argv[i + j + 1]);
            }
            ParsedExpression parsed = parseCommonInformations(args);
            setup.commonInformationsStrings = parsed.groups;
        }
        if (strcmp(argv[i], "-solutionFile") == 0)
        {
            setup.solutionFile = argv[i + 1];
            i += 1;
        }
        if (strcmp(argv[i], "-multiCommonInformations") == 0)
        {
            cout << "Multi common informations" << endl;
            setup.multiCommonInformations = true;
            setup.commonInformations = stoi(argv[i + 1]);
            setup.commonInformationsFile = argv[i + 2];
            i += 2;
        }
    }
    int totalDemands = setup.allDemands + setup.allDemandsOpt + setup.allDifferent + setup.oneDifferent + setup.userDemands;
    // Check the arguments, if invalid, print usage and exit
    if (setup.numUser == 0 || setup.numFile == 0 || totalDemands != 1)
    {
        cout << "Usage: test -nUser <numUser> -nFile <numFile> -M \"{num,den} \"[-alldemands | -alldemandsOpt | -alldifferent | -onedifferent | -demands \"{{a,b,...},{d,e,...},...} \"] [-deletePerm] [-debug <debug>]" << endl;
        if (setup.numUser == 0)
        {
            cerr << "Error: numUser must be greater than 0." << endl;
        }
        if (setup.numFile == 0)
        {
            cerr << "Error: numFile must be greater than 0." << endl;
        }
        if (setup.allDemandsOpt + setup.allDemands + setup.allDifferent + setup.oneDifferent + setup.userDemands != 1)
        {
            cerr << "Error: Exactly one demand type must be specified." << endl;
        }
        exit(1);
    }
    if (setup.M == 0 && !setup.allMvalues)
    {
        cout << "Set a value for M using -M \"{num,den} \" or use -allMValues to use all possible values" << endl;
        exit(1);
    }
    checkK(setup);
    /*
    outFile << "numUser: " << numUser << "\n";
    outFile << "numFile: " << numFile << "\n";
    if (allDemands) outFile << "Using all demands\n";
    else if (allDifferent) outFile << "Using all different demands\n";
    else if (oneDifferent) outFile << "Using one different demand\n";
    if (deletePerm) outFile << "Deleting permutations\n";
    outFile << "M: " << M << "\n";
    outFile << "Debug: " << Debug << "\n";
    */
    return setup;
}