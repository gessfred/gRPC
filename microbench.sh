#!/bin/bash

for f in "bit2byte" "dist" "quantizy"
do
  for i in {18..25}
  do
    export UUID="$uuidgen" 
    export "size"=$i 
    export function=$f 
    echo "$UUID $size $function"
    python benchmark.py
  done
done