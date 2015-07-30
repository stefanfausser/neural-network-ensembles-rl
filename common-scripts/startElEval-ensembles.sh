#!/bin/bash

source startElEval_env.sh

if [ ! -z "$1" ]
then
    echo "Overwriting steps ($steps2) with ($1)"
    steps2="$1"
fi

if [ ! -z "$2" ]
then
    echo "Overwriting offset2 ($offset2) with ($2)"
    offset2="$2"
fi

# committee, ensemble decisions during learning
# for experimentcount in 28 29 30 31
for experimentcount in 4 5 22 23
do
    echo "experiment $experimentcount / $number_experiments"

    if ((experimentcount == 22 || experimentcount == 23 || experimentcount == 28 || experimentcount == 29))
    then
        agentsTmp=$agentsSequentially2
    elif ((experimentcount == 24 || experimentcount == 25 || experimentcount == 32 || experimentcount == 33))
    then
        agentsTmp=$agentsSequentially3
    else
        agentsTmp=$agentsSequentially
    fi
    
    EXTRA_OPTIONS=""
    if ((experimentcount >= 28 && experimentcount <= 31))
    then
        EXTRA_OPTIONS="--jointDecisionsEpsilon 0.9"
    fi
    
    if((useEnsembleIndicesFile2 != 0))
    then
        if [ ! -e $ensembleIndicesFile3"$agentsTmp" ]
        then
            echo file $ensembleIndicesFile3"$agentsTmp" does not exists
            exit 1
        fi
    
        nLines=$(cat $ensembleIndicesFile3"$agentsTmp" | wc -l) 
        testrunsTmp=$(( $nLines / $agentsTmp ))
    else
        testrunsTmp=$testrunsSequentially
    fi
    
    # prepare the conf file
    if ((experimentcount == 4 || experimentcount == 22 || experimentcount == 28 || experimentcount == 30 || experimentcount == 32))
    then
        sed -r "s/agents = [0-9]+/agents = $agentsTmp/g;s/averageDecisionBenchmark = n/averageDecisionBenchmark = y/g;s/averageDecision = n/averageDecision = y/g;s/weightedDecisions = n/weightedDecisions = y/g;s/weightedAverage = n/weightedAverage = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp"$experimentcount".conf
    else
        sed -r "s/agents = [0-9]+/agents = $agentsTmp/g;s/votingDecisionBenchmark = n/votingDecisionBenchmark = y/g;s/votingDecision = n/votingDecision = y/g;s/weightedDecisions = n/weightedDecisions = y/g;s/weightedAverage = n/weightedAverage = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp"$experimentcount".conf
    fi
        
    echo "$agentsTmp agents, $testrunsTmp test runs (on benchmark), $testrunsSequentially test runs (on training)"

    # TODO: Remove dead code
#     if ((experimentcount == 8))
#     then
#         echo "Copying over experiment6 to experiment8 (same training, differs only in joint decision in benchmark)"
#         cp -f "$experimentDir"6/"$savFile"* "$experimentDir""$experimentcount"/
#     elif ((experimentcount == 26))
#     then
#         echo "Copying over experiment4 to experiment26 (same training, differs only in joint decision in benchmark)"
#         cp -f "$experimentDir"4/"$savFile"* "$experimentDir""$experimentcount"/
#     fi
        
    for ((t=$offset2; t <= $steps2; t++))
    do
        echo "step $t / $steps2"

        if ((experimentcount != 8 && experimentcount != 26 && onlyBenchmark == 0))
        then
            for ((i=$testrunStartSequentially; i < $testrunsTmp; i++))
            do
                echo "agent test run $i, training"

                # get seed from seed file
                # get value in line i + 1 (starting index is 1)
                
                ### uncomment following line for ensemble sizes > agentsSequentially
#                seed=$( cat $seedFile | head -"$((($i * $agentsTmp) + 1))" | tail -1 )
                seed=$( cat $seedFile | head -"$((($i * $agentsSequentially) + 1))" | tail -1 )
                seed2=$(( seed + t - 1))
                
                # copy over initial (or last) weights
                for ((a=0; a < agentsTmp; a++))
                do
                    ind=$((agentsTmp * i + a))
                    i2=$ind

                    told=$(( $t - 1 ))
                    
                    seed3=$( cat $seedFile | head -"$((($i * $agentsTmp) + a + 1))" | tail -1 )
                    seed4=$(( seed3 + t - 1))

                    if (( t > 1 ))
                    then                                        
                        a2=$((i2 % agentsTmp))
                        i2=$((i2 / agentsTmp))
                        
                        if [ ! -e "$experimentDir""$experimentcount"/"$savFile""$i2"_"$told"_"$a2" ]
                        then
                            echo file "$experimentDir""$experimentcount"/"$savFile""$i2"_"$told"_"$a2" does not exist
                            exit 1
                        else
                            echo "cp "$experimentDir""$experimentcount"/"$savFile""$i2"_"$told"_"$a2" "$experimentDir""$experimentcount"/"$savFile""$i"_"$t"_"$a""

                            cp "$experimentDir""$experimentcount"/"$savFile""$i2"_"$told"_"$a2" "$experimentDir""$experimentcount"/"$savFile""$i"_"$t"_"$a"                        
                        fi
                    else
                        if [ ! -e experiment1/"$savFile""$i2"_0_0 ]
                        then
#                             echo file experiment1/"$savFile""$i2"_0_0 does not exist, creating it
#                             
#                             ./"$executableBinary" $environmentParameters --createInitWeights --seed "$seed4" --conf "$conf_prefix"_mlp1.conf --savfile experiment1/"$savFile""$i2"_0 $extraOptions > /dev/null

                              echo file experiment1/"$savFile""$i2"_0_0 does not exist
                              exit 1
                        fi

                        echo "cp experiment1/"$savFile""$i2"_0_0 "$experimentDir""$experimentcount"/"$savFile""$i"_1_"$a""
                        
                        if ((t == 0))
                        then
                            cp experiment1/"$savFile""$i2"_0_0 "$experimentDir""$experimentcount"/"$savFile""$i"_0_"$a"
                        fi
                        
                        cp experiment1/"$savFile""$i2"_0_0 "$experimentDir""$experimentcount"/"$savFile""$i"_1_"$a"
                    fi                
                done
                
                if ((t > 0))
                then            
                    # start training
                    ./"$executableBinary" $environmentParameters --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile "$experimentDir""$experimentcount"/"$savFile""$i"_"$t" $extraOptions $EXTRA_OPTIONS --iterationsPerAgent $iterationsPerAgent > "$experimentDir""$experimentcount"/log"$i"_"$t" &
                fi

                # Limit number of parallel testruns 
                if ((((i + 1) % maxParallelTestruns == 0) && (i > 0)))
                then
                    wait
                fi            
            done # i
            
            wait        
        fi
        
        if(( t % $benchmarkIncrement == 0 ))
        then
            for ((i=0; i < $testrunsTmp; i++))
            do
                echo "agent test run $i, benchmark"
            
                # get seed from seed file
                # get value in line i + 1 (starting index is 1)
                seed=$( cat $seedFile | head -"$((($i * agentsTmp) + 1))" | tail -1 )
                seed2=$(( seed + t - 1))
            
                # copy over weights from reference agent (i.e. single agent)
                for ((a=0; a < agentsTmp; a++))
                do
                    if((useEnsembleIndicesFile2 != 0))
                    then
                        ind=$((agentsTmp * i + a + 1))
                    
                        ind2=$(cat $ensembleIndicesFile3"$agentsTmp" | head -"$ind" | tail -1)
                    
                        i2=$(( $ind2 -1 ))
                    else
                        i2=$((agentsTmp * i + a))
                    fi
                    
                    a2=$((i2 % agentsTmp))
                    i2=$((i2 / agentsTmp))
                    
                    if [ ! -e "$experimentDir""$experimentcount"/"$savFile""$i2"_"$t"_"$a2" ]
                    then
                        echo file "$experimentDir""$experimentcount"/"$savFile""$i2"_"$t"_"$a2" does not exists
                        exit 1
                    fi

                    echo "cp "$experimentDir""$experimentcount"/"$savFile""$i2"_"$t"_"$a2" "$experimentDir""$experimentcount"/"$savFile"_benchmark_"$i"_"$t"_"$a""
                    
                    cp "$experimentDir""$experimentcount"/"$savFile""$i2"_"$t"_"$a2" "$experimentDir""$experimentcount"/"$savFile"_benchmark_"$i"_"$t"_"$a"
                done
            
                # start benchmark
                ./"$executableBinary" $environmentParameters --benchmark $benchmarkExtraOptions --seed "$seed2" --conf "$conf_prefix"_mlp"$experimentcount".conf --savfile "$experimentDir""$experimentcount"/"$savFile"_benchmark_"$i"_"$t" $extraOptions --iterationsPerAgent $iterationsPerAgent > "$experimentDir""$experimentcount"/log"$i"_"$t"_benchmark &
                
                # Limit number of parallel testruns 
                if ((((i + 1) % maxParallelTestruns == 0) && (i > 0)))
                then
                    wait
                fi            
            done # for i    

            wait
        fi
    done # t    
done # experimentcount
