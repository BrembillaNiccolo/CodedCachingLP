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
