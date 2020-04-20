#!/bin/bash

for f in "bit2byte" "dist" "quantizy"
do
  for i in {18..25}
  do
    export UUID="$uuidgen" 
    export "tensor-size"=$i 
    export function=$f 
    python benchmark.py
  done
done