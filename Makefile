# Define the compiler and flags
CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -g

# Define the source files
SOURCES = $(wildcard src/*.cpp)

# Define the object files
OBJECTS = $(SOURCES:.cpp=.o)

# Define the executable name
EXECUTABLE = seismic-cpp

# Default target
all: $(EXECUTABLE)

# Link the object files to create the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTABLE)

# Compile the source files to create object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean the object files and executable
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)