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

# committee, no ensemble decisions during learning but during benchmark
for experimentcount in 2 3 10 11 12 13 16 17 18 19 20 21
do
    echo "experiment $experimentcount / $number_experiments"

    if ((experimentcount == 2 || experimentcount == 3))
    then
        agentsTmp=$agents
    elif ((experimentcount == 10 || experimentcount == 11))
    then
        agentsTmp=$agents2
    elif ((experimentcount == 12 || experimentcount == 13))
    then
        agentsTmp=$agents3
    elif ((experimentcount == 16 || experimentcount == 17))
    then
        agentsTmp=$agents4
    elif ((experimentcount == 18 || experimentcount == 19))
    then
        agentsTmp=$agents5
    elif ((experimentcount == 20 || experimentcount == 21))
    then
        agentsTmp=$agents6
    fi

    # prepare the conf file
    if ((experimentcount == 2 || experimentcount == 10 || experimentcount == 12 || experimentcount == 16 || experimentcount == 18 || experimentcount == 20))
    then
        sed -r "s/agents = [0-9]+/agents = $agentsTmp/g;s/averageDecisionBenchmark = n/averageDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp"$experimentcount".conf
    elif ((experimentcount == 3 || experimentcount == 11 || experimentcount == 13 || experimentcount == 17 || experimentcount == 19 || experimentcount == 21))
    then    
        sed -r "s/agents = [0-9]+/agents = $agentsTmp/g;s/votingDecisionBenchmark = n/votingDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp"$experimentcount".conf
    fi
    
    if((useEnsembleIndicesFile != 0))
    then
        if [ ! -e $ensembleIndicesFile2"$agentsTmp" ]
        then
            echo file $ensembleIndicesFile2"$agentsTmp" does not exists
            exit 1
        fi
    
        nLines=$(cat $ensembleIndicesFile2"$agentsTmp" | wc -l) 
        testrunsTmp=$(( $nLines / $agentsTmp ))
    else
        testrunsTmp=$(( $testruns_single_agent / $agentsTmp ))
    fi
    
    echo "$agentsTmp agents, $testrunsTmp test runs"

    for ((t=$startT; t <= $steps; t = t + $benchmarkIncrement))
    do
        echo "step $t / $steps"

        for ((i=0; i < $testrunsTmp; i++))
        do
            echo "agent test run $i, benchmark"

            # get seed from seed file
            # get value in line i + 1 (starting index is 1)
            seed=$( cat $seedFile | head -"$((($i * $agentsTmp) + 1))" | tail -1 )
            seed2=$(( seed + t - 1))
            
            # copy over weights from reference agent (i.e. single agent)
            for ((a=0; a < agentsTmp; a++))
            do
                if((useEnsembleIndicesFile != 0))
                then
                    ind=$((agentsTmp * i + a + 1))
                
                    ind2=$(cat $ensembleIndicesFile2"$agentsTmp" | head -"$ind" | tail -1)
                
                    i2=$(( $ind2 -1 ))
                else
                    i2=$((agentsTmp * i + a))
                fi
                
                echo Copy over experiment1/"$savFile""$i2"_"$t"_0
                 
                if [ ! -e experiment1/"$savFile""$i2"_"$t"_0 ]
                then
                    echo file experiment1/"$savFile""$i2"_"$t"_0 does not exists
                    exit 1
                fi

                cp experiment1/"$savFile""$i2"_"$t"_0 "$experimentDir2""$experimentcount"/"$savFile""$i"_"$t"_"$a"
            done
            
            # start benchmark
            ./"$executableBinary" $environmentParametersRandomlySelective --benchmark $benchmarkExtraOptions --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile "$experimentDir2""$experimentcount"/"$savFile""$i"_"$t" $extraOptions --iterationsPerAgent $iterationsPerAgent > "$experimentDir2""$experimentcount"/log"$i"_"$t"_benchmark &
            
            # Limit number of parallel testruns 
            if ((((i + 1) % maxParallelTestruns == 0) && (i > 0)))
            then
                wait
            fi            
        done # i
        
        wait
        
        # remove savefile (not needed, were just copied over from experiment1)
        rm -f "$experimentDir2""$experimentcount"/"$savFile"*_"$t"_*
        
    done # t    
done # experimentcount
