#!/bin/bash

echo "parse file: $1"
cat $1 | grep -Eo "total reward = [0-9]+\.[0-9]+" | grep -Eo "[0-9]+\.[0-9]+" > "$1"_totalreward
