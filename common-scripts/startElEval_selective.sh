#!/bin/bash

source startElEval_env.sh

if [ $# -ne 5 ]
then
  echo "Usage: `basename $0` start-iteration end-iteration steps isAverage path"
  exit -1
fi

start=$1
end=$2
steps2=$3
isAverage=$4
path=$5

if (( $(echo "$start < 1 || $start > $steps || $start > $end || $end < 1 || $end > $steps" | bc -l) ))
then
    echo "Check start and end iterations.\n"
    exit -1
fi

if ((isAverage == 0))
then
    sequence=15
else
    sequence=14
fi

# selective ensemble learning: committee, no ensemble decisions during learning but during benchmark
for experimentcount in $sequence
do
    echo "experiment $experimentcount / $number_experiments"

    testrunsTmp=$(cat $ensembleSizesFile | wc -l)

    echo "$testrunsTmp test runs"

    # prepare the conf files
    for ((i=0; i < $testrunsTmp; i++))
    do
        agentsTmp=$(cat $ensembleSizesFile | head -$((i + 1)) | tail -1)

        echo "$agentsTmp agents"
        
        if((experimentcount == 14))
        then        
            sed -r "s/agents = [0-9]+/agents = $agentsTmp/g" "$conf_prefix"_mlp.conf-average > "$conf_prefix"_mlp"$experimentcount"_"$i".conf
        else
            sed -r "s/agents = [0-9]+/agents = $agentsTmp/g" "$conf_prefix"_mlp.conf-voting > "$conf_prefix"_mlp"$experimentcount"_"$i".conf
        fi
    done
    
    z=$(echo "scale=0; $start / 1" | bc)
    for t in $(LC_NUMERIC="POSIX" seq $start $steps2 $end)
    do
        echo "step $t / $steps"
        
        ind3=0
        
        for ((i=0; i < $testrunsTmp; i++))
        do
            echo "agent test run $i, benchmark"

            agentsTmp=$(cat $ensembleSizesFile | head -$((i + 1)) | tail -1)

            echo "$agentsTmp agents"
            
            # get seed from seed file
            # get value in line i + 1 (starting index is 1)
            seed=$( cat $seedFile | head -"$((ind3 + 1))" | tail -1 )
            seed2=$(( seed + z - 1))
            
            # copy over weights from reference agent (i.e. single agent)
            for ((a=0; a < agentsTmp; a++))
            do
                ind=$((ind3 + a + 1))
                
                ind2=$(cat $ensembleIndicesFile | head -"$ind" | tail -1)
                
                i2=$(( $ind2 -1 ))

                echo Copy over experiment1/"$savFile""$i2"_"$t"_0
                
                if [ ! -e experiment1/"$savFile""$i2"_"$t"_0 ]
                then
                    echo file experiment1/"$savFile""$i2"_"$t"_0 does not exists
                    exit 1
                fi

                cp experiment1/"$savFile""$i2"_"$t"_0 experiment"$experimentcount"/"$savFile""$i"_"$t"_"$a"
            done
            
            weightsString="$agentsTmp"
            for ((a=0; a < agentsTmp; a++))
            do
                ind=$((ind3 + a + 1))
                
                weight=$(cat $ensembleWeightsFile | head -"$ind" | tail -1)
                weightsString=$( echo "$weightsString $weight" )
            done
            
            # write to log*_50_benchmark instead of log*_50.0_benchmark
            t3=$t
            if (( $(echo "$t % 1 == 0" | bc) ))
            then
                t3=$(echo "scale = 0; $t / 1" | bc)
            fi

            # start benchmark
            ./"$executableBinary" $environmentParameters --benchmark $benchmarkExtraOptions --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount"_"$i".conf --savfile experiment"$experimentcount"/"$savFile""$i"_"$t" $extraOptions --iterationsPerAgent $iterationsPerAgent --decisionWeights $weightsString > experiment"$experimentcount"/log"$i"_"$t3"_benchmark &
            
            # Limit number of parallel testruns 
            if (((i + 1) % maxParallelTestruns == 0))
            then
                wait
            fi
            
            ind3=$((ind3 + agentsTmp))
        done # i
        
        z=$((z + 1))
        wait        
    done # t    
done # experimentcount

if((score != 0))
then
    file_suffix="_benchmark_score"
else
    file_suffix="_benchmark_totalreward"
fi
        
for experimentcount in $sequence
do
    if(((experimentcount >= 4) && (experimentcount <= 9)))
    then
        dir="$experimentDir""$experimentcount"
    elif(( (experimentcount == 2) || (experimentcount == 3) || ((experimentcount >= 10) && (experimentcount <= 13)) || ((experimentcount >= 16) && (experimentcount <= 19)) ))
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

R -e "experimentSeqSelAvgLen = $experimentSeqSelAvgLen; experimentSeqSelVotingLen = $experimentSeqSelVotingLen; nDigitsResults = $nDigitsResults; nDigitsPValues = $nDigitsPValues; file_suffix = '$file_suffix'; nTestrunsSingleReal = $testruns_single_agent; nTestrunsEnsemble = $testrunsTmp; nTestrunsEnsemblePolicyEnsemble = $testrunsSequentially; isAverage = $isAverage; nTestrunsSelectiveEnsemble = $testrunsTmp; selEnsembleStart = $start; selEnsembleEnd = $end; selEnsembleStep = $steps2; source('startElEval_selective.R')"
