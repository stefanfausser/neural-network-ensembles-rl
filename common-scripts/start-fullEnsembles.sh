#!/bin/bash

if [ "$1" == "NEPL" ]
then
    cp startElEval_env.sh-NEPL startElEval_env.sh
    cp startElEval_env.R-NEPL startElEval_env.R
    conf="conf-NEPL"
else
    cp startElEval_env.sh-org startElEval_env.sh
    cp startElEval_env.R-org startElEval_env.R
    conf="conf"
fi

# preparations

cp "$conf"/* .

# training

./startElEval-singleAgents.sh
./startElEval-fullEnsembles.sh
if [ "$1" == "NEPL" ]
then
    ./startElEval-ensembles.sh
fi

# evaluate results

./startElEval-eval.sh
