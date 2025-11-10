# CodedCachingLP

LP-based formulation and symmetry-exploiting reductions for coded caching converse bounds.

---

## 🧩 Overview

This repository contains the C++ implementation of the LP-based converse approach developed for our IEEE ICC paper.  
The method exploits **user–file symmetries** and **entropy equivalences** to reduce the dimensionality of the linear program describing the coded caching converse problem.  
By pruning redundant constraints and merging equivalent variables, the formulation enables the exploration of larger configurations (e.g., up to **6 users and 6 files**).

All results assume **linear coding** for both placement and delivery phases.

---

## ⚙️ Main Features

- LP formulation for coded caching converse under linear coding assumptions  
- Automatic symmetry detection and entropy-equivalence merging  
- Constraint pruning using intrinsic information-theoretic relationships  
- Efficient C++ implementation supporting large-scale instances  
- Example results for configurations from **4U4F to 8U8F**

---

## 📂 Repository Structure

```text
CacheLP/
│
├── main.cc                   # Main LP formulation and solver driver
├── main_reducingVar.cc      # Implementation with variable reduction (Use only if no CIs are in the problem formulation)
├── parser.cc / parser.h       # Parser and entropy expression handler
├── setup.h                    # Configuration and solver setup
├── Makefile                   # Build configuration
│
├── final/                     # Processed results and paper figures
│   ├── 4U4F/                  # 4U4F Problem best tradeoffs found
│   │   └── paper/tradeoff_plots/  # 4U4F Plotted results
│   ├── 5U5F/                  # 5U5F Problem best tradeoffs found
│   │   └── paper/tradeoff_plots/  # 5U5F Plotted results
│   ├── 6U6F/                  # 6U6F Problem best tradeoffs found
│   │   └── paper/tradeoff_plots/  # 6U6F Plotted results
│   ├── 7U7F/                  # 7U7F Problem best tradeoffs found
│   │   └── paper/tradeoff_plots/  # 7U7F Plotted results
│   └── 8U8F/                  # 8U8F Problem best tradeoffs found
│   │   └── paper/tradeoff_plots/  # 8U8F Plotted results
│
├── time_testing/            # Timing benchmarks and performance validation
 ```
---

## 🧮 Compilation 

### Requirements
- **C++17 or later**
- **Make** utility
- [Gurobi](https://www.gurobi.com) LP Solver

### Makefile

The `Makefile` is structured as follows:

```makefile
# Define the C++ compiler to use
CXX = g++

# Define any compile-time flags
CXXFLAGS = -std=c++14 -Wall -Wextra -pedantic -Wno-unused-parameter -fopenmp -O3
GRBPATH = $(GUROBI_HOME)

# Define the C++ source files
SOURCES_TEST = main.cc parser.cc
SOURCES_COMBO =

# Define the object files for test sources
OBJECTS_TEST = $(SOURCES_TEST:.cc=.o)

# Define combo executables and corresponding object files
COMBO_EXE = $(basename $(SOURCES_COMBO))
COMBO_OBJS = $(SOURCES_COMBO:.cc=.o)

# Define the test executable name
MAIN_TEST = main
.PHONY: all clean

# Default target: build test executable and all combo executables
all: $(MAIN_TEST) $(COMBO_EXE)
	@echo "Compilation complete! Executables: $(MAIN_TEST) $(COMBO_EXE)"

# Rule to build the test executable
$(MAIN_TEST): $(OBJECTS_TEST)
	$(CXX) $(CXXFLAGS) $(OBJECTS_TEST) -o $(MAIN_TEST) -I$(GRBPATH)/include -L$(GRBPATH)/lib -lgurobi_c++ -lgurobi120

# Pattern rule to build each combo executable from its corresponding object file
# For example, createCombinationsAndResult.o -> createCombinationsAndResult
$(COMBO_EXE): %: %.o
	$(CXX) $(CXXFLAGS) $< -o $@

# Pattern rule to compile .cc files into .o object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -I$(GRBPATH)/include -c $< -o $@

# Clean rule to remove all object files and executables
clean:
	$(RM) $(OBJECTS_TEST) $(COMBO_OBJS) $(MAIN_TEST) $(COMBO_EXE) *~
	@echo "Clean complete!"
```

### Build
From the repository root:
```bash
make
 ```
---

## 🚀 Usage

After compilation, the main executable can be run with the following syntax:

```bash
./main -nUser <num_users> -nFile <num_files> \
       -M "{numerator,denominator}" | -allMValues \
       -alldemands | -demands "{{demand1},{demand2},...}" \
       | -onedifferent "{a,b,c,...}" | -alldifferent "{a,b,c,...}" | -alldemandsOpt \
       -debug <level> \
       -commonInformations <num_info> "{variable1,variable2,...,variablek and variablek+1,...,variablen}" \
       -solutionFile "<path>"
 ```
### Parameters

| Argument | Description |
|---|---|
| `-nUser <num_users>` | Number of users (K) in the caching system. |
| `-nFile <num_files>` | Number of files (N) available in the library. |
| `-M "{numerator,denominator}"` | Defines the normalized cache size (M/N). Example: `"{1,2}"` means ($M = \frac{1}{2}N$). |
| `-allMValues` | Runs the LP for a predefined set of memory values and writes all results to the file specified by `-solutionFile`. Useful to characterize the full memory–rate tradeoff curve. |
| `-alldemands` | Generates all possible demand combinations for the given (K, N). |
| `-alldemandsOpt` | Optimized (lighter) version of `-alldemands` for heavy computations / large instances. |
| `-demands "{{demand1},{demand2},...}"` | Specifies custom demand sets; each inner set represents file requests from all users. Example: `{{0,1,2},{1,1,0}}`. |
| `-alldifferent "{a,b,c,...}"` | Generates **all** demands of a specified type. Example: `{1,1,1}` for (K=N=3) means all users request different files (sum = (N), length = (K)). |
| `-onedifferent "{a,b,c,...}"` | Generates **one** representative demand of a given type. Use `-deletePerm` to remove equivalent permutations when you want results comparable to `-alldifferent`. |
| `-debug <level>` | Controls verbosity (0 = minimal, 1 = medium, 2 = detailed). Level 2 prints all constraints to `output.txt`. |
| `-commonInformations <num_info> "{var1,... and var3,...}"` | Specifies variables / information to print. Variables include:<br>• `W<num>` — file entropy (e.g., `W0`)<br>• `Z<num>` — cache variable (e.g., `Z0`)<br>• `X{req1,req2,...}` — transmission for a specific demand (e.g., `X{0,1,2}`)<br>Example: `{W0, X{0,1,2} and Z0}` |
| `-solutionFile "<path>"` | Path where results and solution outputs (for all `M` values) will be saved. |


---
## 🧠 Example Commands
### Example 1 => 4 users, 4 files (Optimality at M=1)
```bash
./main -nUser 4 -nFile 4 -M "{1}" -demands "{{0,0,1,2}}" -cycleDemands -debug 0 -solutionFile outputSolution.txt -deletePerm -fullVariables 
```
✅ This setup reaches optimality for M=1 in the 4U4F configuration using two demands and two common information variables → Max value obtained = 1.5

### Example 2 => 5 users, 5 files (Optimality at M=1)
```bash
./main -nUser 5 -nFile 5 -M "{1}" -demands "{{0,0,1,2,3},{1,4,0,0,3}}" -cycleDemands -deletePerm -debug 0 -solutionFile test3U3F.txt -fullVariables -commonInformations 2 "{W0,W1 and X{0,0,1,2,3}}" "{W0,Z0 and X{0,0,1,2,3},Z3}"
```
✅ This configuration achieves optimality for M=1 in the 5U5F case using two demands and two common information variables → Max value obtained = 2

### Example 3 => 5 users, 5 files (Alternative configuration)
```bash
./main -nUser 5 -nFile 5 -M "{1}" -demands "{{0,0,1,2,3},{1,4,0,0,3},{0,1,2,3,0}}" -cycleDemands -deletePerm -debug 0 -solutionFile test3U3F.txt -fullVariables -commonInformations 1 "{W0,W1 and X{0,0,1,2,3}}"
```
✅ This setup uses three demands and one common information variable, still achieving → Max value obtained = 2

---
## ⏱️ Computational Setup

All experiments were executed on a single node of **Delta** at **NCSA**, a high-performance computing cluster.  
The LPs were solved using the **Gurobi 12.0.1** optimizer.

The node configuration is as follows:
- **CPU:** 2× AMD EPYC 7763 (“Milan”) processors — 128 cores total @ 2.45 GHz  
- **Memory:** 256 GB DDR4 RAM  

These specifications were used for all time and performance measurements reported in the paper.

---
## 📝 Notes

- All runs assume linear coding for placement and delivery.

- Results and logs are saved to the path provided with -solutionFile.

- To replicate the paper configurations, see the corresponding folders in final/4U4F, final/5U5F, and final/6U6F.

---
## 📜 License

This project is released under the **MIT License**.  
See the [LICENSE](https://github.com/BrembillaNiccolo/CodedCachingLP/blob/main/LICENSE) file for details.
