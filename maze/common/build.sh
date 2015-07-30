#!/bin/bash

make -C ../ clean
make -C ../
cp ../maze .

cd ../../common
rm -f *.so *.o
for i in $(ls *.c)
do
    R CMD SHLIB "$i"
done
