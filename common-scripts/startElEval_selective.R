source("../../common/startElEval_process.R")

# Following objects have to be initialized by the calling script:
# selEnsembleStart, selEnsembleStep, selEnsembleEnd, isAverage, nTestrunsSelectiveEnsemble, nTestrunsSingleReal, nTestrunsEnsemble, nTestrunsEnsemblePolicyEnsemble, file_suffix, nDigitsResults, nDigitsPValues, experimentSeqSelAvgLen, experimentSeqSelVotingLen

# nTestrunsSingleReal <- 100 # runs for single agent
# nTestrunsEnsemble <- 50 # runs for non-selective non-in-ensemble learning ensemble
# nTestrunsEnsemblePolicyEnsemble <- 20 # runs for in-ensemble learning ensemble

if(isAverage)
{
    experimentSeq <- c(14,16,18,20)[1:experimentSeqSelAvgLen]
        
} else
{
    experimentSeq <- c(15,17,19,21)[1:experimentSeqSelVotingLen]
}

if((selEnsembleStep %% 1) != 0)
{
    ## HACK:
    selEnsembleStart <- selEnsembleEnd
}

# file_suffix <- "_benchmark_totalreward"

ret <- getResultsMain(file_suffix, experimentSeq, experimentSeq, selEnsembleStart, selEnsembleStep, NULL, nTestrunsSingleReal, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, selEnsembleEnd, selEnsembleEnd, TRUE)

# nDigitsResults <- 2
# nDigitsPValues <- 4

cat("Results:\n");
print(round(ret$totalReward,digits=nDigitsResults))
print(round(ret$totalRewardSd,digits=nDigitsResults))
print(round(ret$pValues,digits=nDigitsPValues))

# write 'important' results
vals <- c(round(c(ret$totalReward, ret$totalRewardSd), nDigitsResults), round(ret$pValues, nDigitsPValues))

if(isAverage)
{
    write.table(vals, "selEnsembleAverageResults",col.names=FALSE,row.names=FALSE)
} else
{
    write.table(vals, "selEnsembleVotingResults",col.names=FALSE,row.names=FALSE)
}
