#!/bin/bash

set -e  # Exit on any error

echo "=== KOZ Project Demo Script ==="
echo "This script demonstrates the functionality of the koz service manager"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if koz exists
if [[ ! -f "koz" ]]; then
    print_error "koz not found in current directory"
    exit 1
fi

# Step 1: Attach model
print_step "Step 1: Attaching Ollama model"
echo "Running: python koz model attach"
echo "Selecting Ollama server (option 2) with default model qwen2.5-coder:7b"
echo

# Use expect-like functionality with printf and echo
{
    echo "2"  # Choose Ollama server
    echo ""   # Use default model name (qwen2.5-coder:7b)
} | python koz model attach

if [[ $? -eq 0 ]]; then
    print_success "Model attached successfully"
else
    print_error "Failed to attach model"
    exit 1
fi

echo
sleep 2

# Step 2: Test model with API request
print_step "Step 2: Testing local model API"
echo "Making request to http://localhost:11434/api/generate"

# Create the JSON payload
JSON_PAYLOAD='{
    "model": "qwen2.5-coder:7b",
    "prompt": "Hi, how are you",
    "stream": false,
    "format": "json",
    "options": {
        "stop": ["\\n\\n", "```"]
    }
}'

echo "Request payload:"
echo "$JSON_PAYLOAD"
echo

# Make the API request
echo "Response:"
curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" | python -m json.tool

if [[ $? -eq 0 ]]; then
    print_success "API request successful"
else
    print_warning "API request failed - this might be expected if Ollama is still starting"
fi

echo
sleep 2

# Step 3: Initialize project
print_step "Step 3: Initializing new project"
echo "Running: python koz init my-simple-crud ./test-projects"

# Create test-projects directory if it doesn't exist
mkdir -p ./test-projects

# Initialize project with default choices
{
    echo ""  # Press enter for default choices
    echo ""
    echo ""
    echo ""
    echo ""
} | python koz init my-simple-crud ./test-projects

if [[ $? -eq 0 ]]; then
    print_success "Project initialized successfully"
else
    print_error "Failed to initialize project"
    exit 1
fi

echo
sleep 2
