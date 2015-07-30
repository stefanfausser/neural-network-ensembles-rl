#!/bin/bash

source startElEval_env.sh

# overwrite agents
agents=50
# agents=30

if [ ! -e $ensembleIndicesFile2"$agents" ]
then
    echo file $ensembleIndicesFile2"$agents" does not exists
    exit 1
fi

nLines=$(cat $ensembleIndicesFile2"$agents" | wc -l) 
testruns=$(( $nLines / $agents ))

startJ=0
endJ=4

if [ ! -z "$1" ]
then
    echo "Overwriting startJ ($startJ) with ($1)"
    startJ="$1"
fi

if [ ! -z "$2" ]
then
    echo "Overwriting endJ ($endJ) with ($2)"
    endJ="$2"
fi

for ((z = 0; z < 2; z++))
do
    if ((z == 0))
    then
        conf="$conf_prefix"_mlp_getStateValues-average.conf
        subdir="average/"
        sed -r "s/agents = [0-9]+/agents = $agents/g;s/averageDecisionBenchmark = n/averageDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > $conf
        maxBestActionsTmp="$maxBestActions"
        retriesConsistencies=$retriesConsistenciesAverage
    else
        conf="$conf_prefix"_mlp_getStateValues-voting.conf
        subdir="voting/"
        sed -r "s/agents = [0-9]+/agents = $agents/g;s/votingDecisionBenchmark = n/votingDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > $conf
        maxBestActionsTmp=1
        retriesConsistencies=$retriesConsistenciesVoting
    fi

    actionStr="--bestActions $maxBestActionsTmp"
    if [ $maxBestActions -eq 0 ]
    then
        actionStr="--allActions"
    fi
    
    for ((j = $startJ; j < $endJ; j++))
    do
        if ((j == 0))
        then
            t1=$(( steps / 5 / 2))
            t2=$(( steps / 5 / 2))
        elif ((j == 1))
        then
            t1=$(( steps / 5 ))
            t2=$(( steps / 5 ))
        elif ((j == 2))
        then
            t1=$(( steps / 5 * 3 ))
            t2=$(( steps / 5 * 3 ))
        elif ((j == 3))
        then
            t1=$steps
            t2=$steps
        fi
        
        for ((i=0; i < $testruns; i++))
        do
            echo "committee test run $i"

            mkdir -p repo_it"$t2"/"$agents"/"$subdir"/"$i"
            
            # get seed from seed file
            # get value in line i + 1 (starting index is 1)
            seed=$( cat $seedFile | head -"$((($i * $agents) + 1))" | tail -1 )
            seed2=$(( seed + t2 - 1))
            
            # create state coding file

            # the selection of the agent(s) (single or ensemble) decides over
            # the real distribution of the V(s) or Q(s,a) pairs that are captured.
            # Further, the successor pairs are captured and can be retrieved
            # by using the --usePolicy argument.

            if [ ! -e repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepo"$t2" ]
            then
                # copy over weights from reference agent (i.e. single agent)
                for ((a=0; a < agents; a++))
                do
                    ind=$((agents * i + a + 1))
                
                    ind2=$(cat $ensembleIndicesFile2"$agents" | head -"$ind" | tail -1)
                
                    i2=$(( $ind2 -1 ))
                    
#                    echo Copy over experiment1/"$savFile""$i2"_"$t2"_0
                        
                    if [ ! -e experiment1/"$savFile""$i2"_"$t2"_0 ]
                    then
                        echo file experiment1/"$savFile""$i2"_"$t2"_0 does not exists
                        exit 1
                    fi

                    cp experiment1/"$savFile""$i2"_"$t2"_0 repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$savFile"0_"$t2"_"$a"
                done # a
                
                ./"$executableBinary" $environmentParametersBasic $getStatesBenchmarkExtras --benchmark $actionStr --seed "$seed2" --conf "$conf" --savfile repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$savFile"0_"$t2" --iterationsPerAgent 1 $benchmarkExtras --createStateRepo "$maxStates" --probToAddState "$probToAddState" --repoFile repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepo"$t2" --repoFileStateProbs repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepoSP"$t2" --repoFileNumberActions repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepoNA"$t2" > repo_it"$t2"/"$agents"/"$subdir"/"$i"/log

                # remove mlp weights to save memory
                rm -f repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$savFile"0_"$t2"_*
            fi

            # get state or state-action values
            
            for t in $(seq $t1 1 $t2)
            do
                for ((a=0; a < agents; a++))
                do
                    ind=$((agents * i + a + 1))
                
                    ind2=$(cat $ensembleIndicesFile2"$agents" | head -"$ind" | tail -1)
                
                    i2=$(( $ind2 -1 ))
                    
                    ./"$executableBinary" $environmentParametersBasic --benchmark $actionStr --retries "$retriesConsistencies" --seed "$seed2" --conf "$conf_prefix"_mlp1.conf --savfile experiment1/"$savFile""$i2"_"$t" --iterationsPerAgent 1 --calcConsistencies --evaluateStateRepo "$maxStates" --repoFile repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepo"$t2" --repoErrorsFile repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepoErrors"$i2"_"$t" --repoActionNumbersFile repo_it"$t2"/"$agents"/"$subdir"/"$i"/"$conf_prefix"StateRepoActionNumbers"$i2"_"$t" > /dev/null &
                    
                    # Limit number of parallel testruns
                    if ((((a + 1) % maxParallelTestruns == 0) && (a > 0)))
                    then
                        wait
                    fi            
                done # a
            done # t       
        done # i
        
        wait        
        
    done # j
done # z
