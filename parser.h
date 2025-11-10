#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <regex>

#include "setup.h"

// Utility function declarations
bool isNumber(const std::string &s);
std::string trim(const std::string &str);
std::vector<int> parseSingleDemand(const std::string &str);
std::vector<std::vector<int>> parseDemands(const std::string &str);
std::vector<std::string> parseBracketsContent(std::string &str);

// Core parsing functions
ParsedExpression parseCommonInformations(std::vector<std::string> &args);
void checkK(Setup &setup);

// Simplified argument parsing: returns a Setup object filled with values from the command-line
Setup parseArgs(int argc, char *argv[]);

#endif // PARSER_H
