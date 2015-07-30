#!/bin/bash

source startElEval_env.sh

if [ ! -z "$1" ]
then
    echo "Overwriting steps ($steps) with ($1)"
    steps="$1"
fi

startT=0

if [ ! -z "$2" ]
then
    echo "Overwriting startT ($startT) with ($2)"
    startT="$2"
fi

# single agent
experimentcount=1
echo "experiment $experimentcount / $number_experiments"
echo "single agent tests for $testruns_single_agent agents"
for ((t=startT; t <= $steps; t++))
do
    echo "step $t / $steps"
    
    if ((onlyBenchmark == 0 ))
    then
        for ((i=$testrunStartSingle; i < $testruns_single_agent; i++))
        do
            echo "agent test run $i, training"
        
            # get seed from seed file
            # get value in line i + 1 (starting index is 1)
            seed=$( cat $seedFile | head -"$(($i + 1))" | tail -1 )
            seed2=$(( seed + t - 1))
        
            # create initial weights
            if (( t > 1 ))
            then
                told=$(( $t - 1 ))
                if [ ! -e experiment1/"$savFile""$i"_"$told"_0 ]
                then
                    echo file experiment1/"$savFile""$i"_"$told"_0 does not exist
                    exit 1
                else
                    cp experiment1/"$savFile""$i"_"$told"_0 experiment1/"$savFile""$i"_"$t"_0
                fi
            else
                if [ ! -e experiment1/"$savFile""$i"_0_0 ]
                then
                    echo file experiment1/"$savFile""$i"_0_0 does not exist, creating it
                    
                    ./"$executableBinary" $environmentParameters --createInitWeights --seed "$seed" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile experiment1/"$savFile""$i"_0 $extraOptions > /dev/null
                fi
                
                if [ ! -e experiment1/"$savFile""$i"_0_0 ]
                then
                    echo file experiment1/"$savFile""$i"_0_0 does not exist
                    exit 1
                else
                    cp experiment1/"$savFile""$i"_0_0 experiment1/"$savFile""$i"_1_0
                fi
            fi
            
            if ((t > 0))
            then            
                # start training
                ./"$executableBinary" $environmentParameters --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile experiment1/"$savFile""$i"_"$t" $extraOptions --iterationsPerAgent $iterationsPerAgent > experiment1/log"$i"_"$t" &
            fi
            
            # Limit number of parallel testruns 
            if ((((i + 1) % maxParallelTestruns == 0) && (i > 0)))
            then
                wait
            fi            
        done # for i
        
        wait
    fi
    
    if(( t % $benchmarkIncrement == 0 ))
    then    
        for ((i=$testrunStartSingle; i < $testruns_single_agent; i++))
        do
            echo "agent test run $i, benchmark"
        
            # get seed from seed file
            # get value in line i + 1 (starting index is 1)
            seed=$( cat $seedFile | head -"$(($i + 1))" | tail -1 )    
            seed2=$(( seed + t - 1))
        
            # start benchmark
            ./"$executableBinary" $environmentParameters --benchmark $benchmarkExtraOptions --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile experiment1/"$savFile""$i"_"$t" $extraOptions --iterationsPerAgent $iterationsPerAgent > experiment1/log"$i"_"$t"_benchmark &

            # Limit number of parallel testruns 
            if (((i % maxParallelTestruns == 0) && (i > 0)))
            then
                wait
            fi
        done # for i
        
        wait
    fi
    
done # for t
