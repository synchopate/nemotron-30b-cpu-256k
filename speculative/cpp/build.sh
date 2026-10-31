#!/bin/bash
# Build libroutecrack.so
cd "$(dirname "$0")"
g++ -O2 -shared -fPIC -o libroutecrack.so rc_predictor_lib.cpp -lpthread
echo "Built: $(pwd)/libroutecrack.so"
