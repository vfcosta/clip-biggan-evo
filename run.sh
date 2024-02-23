#!/bin/bash

for i in $(seq 1 100);
do
  python cma-es-engine.py --local-search-steps 5 --pop-size 10 --image-size 128 --max-gens 30
done
