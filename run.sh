#!/bin/bash

for i in $(seq 1 200);
do
  python cma-es-engine.py --local-search-steps 1 --pop-size 10 --image-size 128 --max-gens 30 --save-folder 2024/experiments
done
