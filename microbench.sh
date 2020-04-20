#!/bin/bash

for f in "bit2byte" "dist" "quantizy"
do
  for i in {18..25}
  do
    UUID="$uuidgen" "tensor-size"=$i function=$f python benchmark.py
  done
done