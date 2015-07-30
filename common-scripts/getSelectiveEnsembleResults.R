library(R.utils)
source("../../common/startElEval_process.R")
source("startElEval_env.R")

tSeq <- c(len / 5 / 2, len / 5, len / 5 * 3, len)
averageSeq <- c(TRUE, FALSE)
methodSeq <- c(0, 1)

maxM <- 40
maxPvalue <- 0.05
# negligibleDiff <- 0.01
negligibleDiff <- 0.005

nResults <- 6

bestResultMat <- matrix(0, nResults, length(tSeq))
bestResultSdMat <- matrix(0, nResults, length(tSeq))
bestMMat <- matrix(0, nResults, length(tSeq))
bestPvalueMat <- matrix(0, nResults, length(tSeq))
bestPvalueTTestMat <- matrix(0, nResults, length(tSeq))
bestResultDiffMat <- matrix(0, nResults, length(tSeq))
bestThresholdMat <- matrix(0, nResults, length(tSeq))

fileBestValsAverage <- "bestValsAverage50"
fileBestValsVoting <- "bestValsVoting50"

valsAverage <- NULL
valsVoting <- NULL

if(file.exists(fileBestValsAverage))
{
    valsAverage <- read.table(fileBestValsAverage)
}

if(file.exists(fileBestValsVoting))
{
    valsVoting <- read.table(fileBestValsVoting)
}

i <- 1
for(t in tSeq)
{
    j <- 1
    for(method in methodSeq)
    {
        useSingleRepo <- FALSE
        greedyEnsemble <- FALSE
        
        if(method == 1)
            useSingleRepo <- TRUE
        else if(method == 2)
            greedyEnsemble <- TRUE
    
        for(average in averageSeq)
        {
            if(average)
            {
                if(useSingleRepo)
                    file <- paste("results1-singleDecisions-allAgents-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-repo_it", t, sep="")
                else if(greedyEnsemble)
                    file <- paste("results1-greedy-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-repo_it", t, sep="")                    
                else
                    file <- paste("results1-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-repo_it", t, sep="")
            }
            else
            {
                if(useSingleRepo)
                    file <- paste("results1-singleDecisions-allAgents-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-repo_it", t, sep="")
                else if(greedyEnsemble)
                    file <- paste("results1-greedy-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-repo_it", t, sep="")
                else
                    file <- paste("results1-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-repo_it", t, sep="")
            }

            tab <- read.table(file)

            results <- tab$benchmarkMean
            resultsSd <- tab$benchmarkSd
            if(pValuesFrom == 20)            
                pvalues <- tab$pValue1 # 20 agents
            else if(pValuesFrom == 30)
                pvalues <- tab$pValue2 # 30 agents
            else
                pvalues <- tab$pValue3 # 50 agents

            committeeSizes <- unique(tab$committeeSizeAvg)

            bestResult <- -Inf
            lastBestResult <- -Inf
            first <- TRUE

            for(M in committeeSizes)
            {
                ind <- tab$committeeSizeAvg == M
                ind2 <- which.max(results[ind])

                res <- (results[ind])[ind2]
                pvalue <- (pvalues[ind])[ind2]
                thresh <- (tab$trueQuantileThreshold[ind])[ind2]

    #            if(res > bestResult && (pvalue < maxPvalue || first) && M <= maxM)
                if(res > bestResult && M <= maxM)
                {
                    if(!first)
                    {
                        lastBestM <- bestM
                        lastBestResult <- bestResult
                        lastBestResultSd <- bestResultSd
                        lastBestPvalue <- bestPvalue
                        lastBestThreshold <- bestThreshold
                    }
                
                    first <- FALSE
                
                    bestM <- M
                    bestResult <- res
                    bestResultSd <- (resultsSd[ind])[ind2]
                    bestPvalue <- pvalue
                    bestThreshold <- thresh
                }
            }

            # Take the last best result, if the difference in the result is small
            if(bestResult - negligibleDiff <= lastBestResult)
            {
                bestM <- lastBestM
                bestResult <- lastBestResult
                bestResultSd <- lastBestResultSd
                bestPvalue <- lastBestPvalue
                bestThreshold <- lastBestThreshold
            }

            bestResultMat[j,i] <- bestResult
            bestResultSdMat[j,i] <- bestResultSd
            bestMMat[j,i] <- bestM
            bestPvalueMat[j,i] <- bestPvalue
            bestThresholdMat[j,i] <- bestThreshold
            
            if(average)
            {
                if(file.exists(fileBestValsAverage))
                {
                    bestPvalueTTestMat[j,i] <- myttest(c(bestResult,valsAverage[1,i]),c(bestResultSd,valsAverage[2,i]),c(valsAverage[3,i],valsAverage[3,i]))$p.value
                    bestResultDiffMat[j,i] <- bestResult - valsAverage[1,i]
                }
            }
            else
            {
                if(file.exists(fileBestValsVoting))
                {
                    bestPvalueTTestMat[j,i] <- myttest(c(bestResult,valsVoting[1,i]),c(bestResultSd,valsVoting[2,i]),c(valsVoting[3,i],valsVoting[3,i]))$p.value
                    bestResultDiffMat[j,i] <- bestResult - valsVoting[1,i]
                }            
            }

            j <- j + 1
        } # average
    } # method

    i <- i + 1
}

cat("Latex table entries:\n")

i <- 1
for(method in methodSeq)
{
    useSingleRepo <- FALSE
    greedyEnsemble <- FALSE
    
    if(method == 1)
        useSingleRepo <- TRUE
    else if(method == 2)
        greedyEnsemble <- TRUE

    for(average in averageSeq)
    {
        str <- NULL
        if(useSingleRepo)
            str <- "Sl(1). &"
        else if(greedyEnsemble)
            str <- "GR. &"
        else
            str <- "Sl. &"

        if(average)
            str <- paste(str, " A", sep="")
        else
            str <- paste(str, " V", sep="")

        formatStr <- paste("%.", nDigits, "f", sep="")
            
        printf(paste("%s & $", formatStr, " \\pm ", formatStr, "$ & $", formatStr, " \\pm ", formatStr, "$ & $", formatStr, " \\pm ", formatStr, "$ & $", formatStr, " \\pm ", formatStr, "$\n", sep=""), str, round(bestResultMat[i,1],digits=nDigits), round(bestResultSdMat[i,1],digits=nDigits), round(bestResultMat[i,2],digits=nDigits), round(bestResultSdMat[i,2],digits=nDigits), round(bestResultMat[i,3],digits=nDigits), round(bestResultSdMat[i,3],digits=nDigits), round(bestResultMat[i,4],digits=nDigits), round(bestResultSdMat[i,4],digits=nDigits))
            
        i <- i + 1
    }
}

cat("Committee sizes:\n")
print(bestMMat)

cat("P-values (ANOVA):\n")
print(bestPvalueMat)

cat("Thresholds:\n")
print(bestThresholdMat)

cat("Result differences:\n")
print(round(bestResultDiffMat,nDigits))

cat("P-values (t-test):\n")
print(bestPvalueTTestMat)

cat("P-values (t-test) < 0.05:\n")
print(bestPvalueTTestMat < 0.05)

cat("P-values (t-test) < 0.01:\n")
print(bestPvalueTTestMat < 0.01)
