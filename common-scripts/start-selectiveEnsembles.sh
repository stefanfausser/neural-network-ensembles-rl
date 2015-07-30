#!/bin/bash

# preparations

cp startElEval_env.sh-org startElEval_env.sh

cp conf/* .

source startElEval_env.sh

# prepare config files

sed -r "s/averageDecisionBenchmark = n/averageDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp.conf-average
sed -r "s/votingDecisionBenchmark = n/votingDecisionBenchmark = y/g" "$conf_prefix"_mlp1.conf > "$conf_prefix"_mlp.conf-voting

# state collection

./getStateValues.sh
./getStateValues-singleDecisions-allAgents.sh

# selective ensemble learning by solving B-QP problem

R -e "source('selectiveEnsembleLearning.R'); performTests()"

# evaluate results

R -e "source('parameterPlot1-thesis.R')"
R -e "source('getSelectiveEnsembleResults.R')"
