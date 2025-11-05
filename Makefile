CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Iinclude

SRC_DIR = src
BUILD_DIR = build

SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ_FILES) main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build complete: ./$(TARGET)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo "Cleaned up build files."
