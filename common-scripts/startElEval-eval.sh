#!/bin/bash

source startElEval_env.sh

for ((experimentcount=1; experimentcount <= number_experiments; experimentcount++))
do
    if(((experimentcount >= 4) && (experimentcount <= 9)))
    then
        dir="$experimentDir""$experimentcount"
    elif(( (experimentcount == 2) || (experimentcount == 3) || ((experimentcount >= 10) && (experimentcount <= 13)) || ((experimentcount >= 16) && (experimentcount <= 21)) ))
    then
        dir="$experimentDir2""$experimentcount"
    else
        dir=experiment"$experimentcount"
    fi

    echo "parsing directory $dir"
    
    cd "$dir"
    
    # delete the old log files
    rm -f log*_benchmark_totalreward
    rm -f log*_benchmark_score

    # parse the log files and extract the values
    for i in $( ls log*_benchmark )
    do
        if((score != 0))
        then
            ../extractvals_score.sh "$i"
        else
            ../extractvals.sh "$i"
        fi
    done
    
    cd ../
done

R -e "source('startElEval.R')"
