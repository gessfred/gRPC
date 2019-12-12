#!/bin/bash
for v in numpy ext ext_par #type
do
    for sz in {16..25} #size
    do
        python3 benchmark.py -i 10000 -sz $sz -v $v --empty
    done
done


