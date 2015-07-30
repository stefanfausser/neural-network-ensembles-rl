# The MIT License (MIT)
# 
# Copyright (c) 2013 - 2015 Stefan Faußer
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# selectiveEnsembleLearning library
# Modification history:
# 2013-08-01, S.Fausser - written

library(quadprog)
library(GA)

calculateMSE <- function(err, M, weights)
{
    N <- ncol(err)

    mse <- 0

    values <- .C("calcMSE", err=as.numeric(err), M=as.integer(M), N=as.integer(N), weights=as.numeric(weights), mse=as.numeric(mse))

    ret <- list(mse=as.numeric(values$mse))

    return(ret$mse)
}

calculateMSE2.err <- function(err, M, weights)
{
    N <- ncol(err)

    mse <- 0

    values <- .C("calcEnsembleMSE", err=as.numeric(err), M=as.integer(M), N=as.integer(N), weights=as.numeric(weights), mse=as.numeric(mse))

    ret <- list(mse=as.numeric(values$mse))

    return(ret$mse)
}

readValues <- function(repoDir, indices, filePrefix, nAgents, t)
{
    # get number of states

    file <- paste(repoDir, "/", filePrefix, "StateRepoErrors", indices[1], "_", t, sep="")

    values <- .C("getFileLines", path=as.character(file), nLines=as.integer(0))
    ret <- list(nLines=as.integer(values$nLines))

    maxVals <- ret$nLines

    nStates <- ret$nLines / 2

    cat("Number of states =",nStates,"\n")

    # read state probabilities p(s)

    file2 <- paste(repoDir, "/", filePrefix, "StateRepoSP", t, sep="")

    values <- .C("readFileAsVector", path=as.character(file2), maxVals=as.integer(maxVals), vals=as.numeric(rep(0, maxVals)), readVals=as.integer(0))
    ret <- list(vals=as.numeric(values$vals), readVals=as.integer(values$readVals))

    stateProbabilities <- ret$vals[1:ret$readVals]

    # read number of actions

    file3 <- paste(repoDir, "/", filePrefix, "StateRepoNA", t, sep="")

    values <- .C("readFileAsVector", path=as.character(file3), maxVals=as.integer(maxVals), vals=as.numeric(rep(0, maxVals)), readVals=as.integer(0))
    ret <- list(vals=as.numeric(values$vals), readVals=as.integer(values$readVals))

    nActions <- ret$vals[1:ret$readVals]

    statesSeq <- 1:(nStates * 2)
        
    yPredicted <- matrix(0, nAgents, nStates)
    yReal <- matrix(0, nAgents, nStates)
    actionIndices <- matrix(0, nAgents, nStates)

    for(i in 1:nAgents)
    {
        ind <- indices[i]
    
        file <- paste(repoDir, "/", filePrefix, "StateRepoErrors", ind, "_", t, sep="")

        if(!file.exists(file))
        {
            cat("file ", file, " does not exist\n")
        }

        values <- .C("readFileAsVector", path=as.character(file), maxVals=as.integer(maxVals), vals=as.numeric(rep(0, maxVals)), readVals=as.integer(0))
        ret <- list(vals=as.numeric(values$vals), readVals=as.integer(values$readVals))

        mat <- t(matrix(t(ret$vals)[statesSeq],2,nStates))
        
        yPredicted[i,] <- mat[,1]
        yReal[i,] <- mat[,2]

        # read action indices
        file4 <- paste(repoDir, "/", filePrefix, "StateRepoActionNumbers", ind, "_", t, sep="")

        if(!file.exists(file4))
        {
#            cat("file ", file4, " does not exist\n")
#            cat("Using dummy entries:\n")
            actionIndices[i,] <- 1:nStates
        }
        else
        {
            values <- .C("readFileAsVector", path=as.character(file4), maxVals=as.integer(maxVals), vals=as.numeric(rep(0, maxVals)), readVals=as.integer(0))
            ret <- list(vals=as.numeric(values$vals), readVals=as.integer(values$readVals))

            actionIndices[i,] <- t(ret$vals[1:ret$readVals])
        }
    }

    return(list(yPredicted = yPredicted, yReal = yReal, nStates = nStates, stateProbabilities = stateProbabilities, numberActions = nActions, actionIndices = actionIndices))
}

readBenchmarkResults <- function(nVals, t)
{
    # Get benchmark results

    vals <- rep(0, nVals)
    for(j in 1:nVals)
    {
        filename <- paste("experiment1/log", j - 1, "_", t, "_benchmark_totalreward", sep="")
#        filename <- paste("experiment1/log", j - 1, "_", t, "_benchmark_score", sep="")
        vals[j] <- as.numeric(read.table(filename))
    }

    return(list(vals=vals))    
}

calculateErrors <- function(yPredicted, yReal, nAgents, nStates, verbose = TRUE, absoluteError = TRUE, averagedYReal = FALSE, zeroMeanErrors = FALSE, unityVarianceErrors = FALSE, C = 1, C2 = 1, thresholdCust = 0, quantileThreshold = 0, quantileThresholdInv = 0, trueQuantileThreshold = 0)
{
    error <- matrix(0, nAgents, nStates)    

    useAverageYReal <- FALSE
    
    if(averagedYReal)
    {
        yRealAveraged <- rep(0, nStates)
        
        for(s in 1:nStates)
            yRealAveraged[s] <- mean(yReal[,s])
        
        useAverageYReal <- TRUE
    }
    
    if(unityVarianceErrors && C > 0)
        sds <- rep(0, nAgents)
    
    for(i in 1:nAgents)
    {
        if(absoluteError)
        {
            # absolute error
            
            if(useAverageYReal)
                error[i,] <- yPredicted[i,] - yRealAveraged
            else
                error[i,] <- yPredicted[i,] - yReal[i,]
        }
        else
        {
            # relative error
            
            if(useAverageYReal)
                error[i,] <- (yPredicted[i,] - yRealAveraged) / yRealAveraged
            else
                error[i,] <- (yPredicted[i,] - yReal[i,]) / yReal[i,]
        }
        if(unityVarianceErrors && C > 0)
            sds[i] <- sd(error[i,])
    }
    
    errorUnmodified <- error
    
    if(unityVarianceErrors && C > 0)
    {
        sdAvg <- mean(sds)
        if(verbose)
            cat("sdAvg: ", sdAvg, "\n")
    }
    
    if((unityVarianceErrors && C > 0)  || (zeroMeanErrors && C2 > 0))
    {
        for(i in 1:nAgents)
        {
            if(zeroMeanErrors)
            {
                ## ATTENTION: With this preprocessing, minimizing the squared bias
                ## (with selectiveEnsembleLearning.squaredBias) is equivalent to 
                ## minimizing the covariance (with selectiveEnsembleLearning.cov)

                # preprocess the errors; they shall have zero mean over the base agents
                error[i,] <- error[i,] - C2 * mean(error[i,])
            }
            
            if(unityVarianceErrors)
                error[i,] <- error[i,] / sds[i] * (sds[i] + C * (sdAvg - sds[i]))
        }
    }
        
    threshold <- 0
    thresholdInv <- 0
    
    if(thresholdCust > 0)
        threshold <- thresholdCust
    
    if(quantileThreshold > 0 || quantileThresholdInv > 0 || trueQuantileThreshold > 0)
    {
        quant <- NULL
        quantInv <- NULL
        for(i in 1:nAgents)
        {
            if(trueQuantileThreshold > 0)
                quant <- c(quant, as.numeric(quantile(abs(error[i,]), probs=trueQuantileThreshold)))
            else if(quantileThreshold > 0)
                quant <- c(quant, as.numeric(quantile(abs(error[i,]), probs=quantileThreshold)))
            
            if(quantileThresholdInv > 0)
                quantInv <- c(quantInv, as.numeric(quantile(abs(error[i,]), probs=quantileThresholdInv)))
        }

        if(quantileThreshold > 0 || trueQuantileThreshold > 0)
            threshold <- mean(quant)
            
        if(quantileThresholdInv > 0)
            thresholdInv <- mean(quantInv)            

        if(trueQuantileThreshold > 0 || quantileThreshold > 0)
            cat("Threshold: ", threshold, "\n")
        if(quantileThresholdInv > 0)
            cat("Threshold (inv): ", thresholdInv, "\n")
    }
    
    if(trueQuantileThreshold > 0 || thresholdCust > 0)
    {
        # hard criteria: error (1) or no error (0)
        error[abs(error) <= threshold] <- 0
        error[abs(error) > threshold] <- 1
    }
    else
    {
        # Remove some 'noise'
        if(quantileThreshold > 0)
            error[abs(error) < threshold] <- 0
        if(quantileThresholdInv > 0)
            error[abs(error) > thresholdInv] <- thresholdInv
    }
    
    meanErrorByAgent <- rep(0, nAgents)
    varErrorByAgent <- rep(0, nAgents)
    for(i in 1:nAgents)
    {
        meanErrorByAgent[i] <- mean(error[i,])
        varErrorByAgent[i] <- var(error[i,])
    }

    meanSquaredErrorByAgent <- rep(0, nAgents)
    for(i in 1:nAgents)
    {
        meanSquaredErrorByAgent[i] <- mean(error[i,]^2)
    }

    if(verbose)
    {
        bestSingleAgentIndices <- order(abs(meanErrorByAgent),decreasing=FALSE)
        cat("Absolute BIAS: Indices of the best agents, starting with the best:\n")
        print(bestSingleAgentIndices)
        cat("Errors of the best agents:\n")
        print(sort(abs(meanErrorByAgent),decreasing=FALSE))

        bestSingleAgentIndices <- order(abs(meanSquaredErrorByAgent),decreasing=FALSE)
        cat("MSE (= BIAS^2 + VAR) or TD ERROR: Indices of the best agents, starting with the best:\n")
        print(bestSingleAgentIndices)
        cat("Errors of the best agents:\n")
        print(sort(abs(meanSquaredErrorByAgent),decreasing=FALSE))

        # VAR + BIAS^2 = MSE
#    (varErrorByAgent + meanErrorByAgent^2) - meanSquaredErrorByAgent
        cat("sum of VAR + BIAS^2 - MSE (should be 0):\n")
        print(sum((varErrorByAgent + meanErrorByAgent^2) - meanSquaredErrorByAgent))
    }

    return(list(error = error, errorUnmodified = errorUnmodified, meanErrorByAgent = meanErrorByAgent, varErrorByAgent = varErrorByAgent, meanSquaredErrorByAgent = meanSquaredErrorByAgent))
}

getBestIndices <- function(method, M, ensembleSizeMax, nCommittees, indicesMat, err, errEvaluation, yPredicted, yReal, yProb, actionIndices, nActions, nActionsMax, C, C2, alphaMin, noDuplicates = FALSE, verbose = TRUE)
{
    e <- 1
    nDuplicates <- 0

    ensembleIndices <- matrix(M + 1, nCommittees, M)

    ensembleIndicesVec <- NULL
    ensembleSizesVec <- NULL
    ensembleWeightsVec <- NULL

    nLowerMSE <- 0
    nLowerMSEEquallyWeighted <- 0
    nLowerMSE2 <- 0
    nLowerMSE2EquallyWeighted <- 0
    
    sumMSEDiff <- 0
    sumMSEEqualWeightsDiff <- 0
    mse2Diff <- NULL
    mse2EqualWeightsDiff <- NULL
       
    while(1)
    {
        if(e > nCommittees)
            break # while

        if(e > nrow(indicesMat))
            break # while
            
        # Get M indices out of 'nAgents' without replacement
        indices <- indicesMat[e,]

        if(verbose)
        {
            cat("Using following agent indices:\n")
            print(sort(indices))
        }
        
        # Choose the best indices out of those M indices (i.e. perform selective ensemble learning)
        ret <- selectiveEnsembleLearning(err[indices,], yProb, actionIndices[indices,], nActions, nActionsMax, yPredicted[indices,], C, C2, alphaMin, build = method)

        ind <- indices[ret$indices]

        indOrderedIndices <- order(ind)
        nIndices <- length(ind)

        indOrdered <- ind[indOrderedIndices]
        weightsOrdered <- ret$weights[indOrderedIndices]
        
        if(verbose)
        {
            cat("(Best) Ensemble indices:\n")
            print(indOrdered)

            cat("(Best) Ensemble weights:\n")
            print(weightsOrdered)
        }
        
        # select up to 'ensembleSizeMax' from these indices
        indO <- order(weightsOrdered,decreasing=TRUE)[1:min(ensembleSizeMax, nIndices)]
        indOrdered <- indOrdered[indO]
        weightsOrdered <- weightsOrdered[indO]
        nIndices <- length(indOrdered)
        
        if(verbose)
        {
            cat("(Best) Ensemble indices (reduced):\n")
            print(indOrdered)

            cat("(Best) Ensemble weights (reduced):\n")
            print(weightsOrdered)

            cat("Sum of weights:\n")
            print(sum(weightsOrdered[1:nIndices]))
        }

        MSE <- calculateMSE(errEvaluation[indices,], M, rep(1,M))
        if(verbose)
        {
            cat("MSE (all):\n")
            print(MSE)
        }

        MSE.selective <- calculateMSE(errEvaluation[indOrdered[1:nIndices],], nIndices, weightsOrdered)

        if(verbose)
        {
            cat("MSE (selective):\n")
            print(MSE.selective)
        }

        MSE.selective2 <- calculateMSE(errEvaluation[indOrdered[1:nIndices],], nIndices, rep(1,nIndices))

        if(verbose)
        {
            cat("MSE (selective, equally weighted):\n")
            print(MSE.selective2)
        }

        sumMSEDiff <- sumMSEDiff + (MSE - MSE.selective)
        if(MSE.selective < MSE)
            nLowerMSE <- nLowerMSE + 1
        
        sumMSEEqualWeightsDiff <- sumMSEEqualWeightsDiff + (MSE - MSE.selective2)
        if(MSE.selective2 < MSE)
            nLowerMSEEquallyWeighted <- nLowerMSEEquallyWeighted + 1
        
        MSE2 <- calculateMSE2.err(errEvaluation[indices,], M, rep(1,M))
        
        if(verbose)
        {
            cat("MSE2 (all):\n")
            print(MSE2)
        }
        
        MSE2.selective <- calculateMSE2.err(errEvaluation[indOrdered[1:nIndices],], nIndices, weightsOrdered)

        if(verbose)
        {
            cat("MSE2 (selective):\n")
            print(MSE2.selective)
        }

        MSE2.selective2 <- calculateMSE2.err(errEvaluation[indOrdered[1:nIndices],], nIndices, rep(1,nIndices))

        if(verbose)
        {
            cat("MSE2 (selective, equally weighted):\n")
            print(MSE2.selective2)
        }

        mse2Diff <- c(mse2Diff, (MSE2 - MSE2.selective))
        if(MSE2.selective < MSE2)
            nLowerMSE2 <- nLowerMSE2 + 1
        
        mse2EqualWeightsDiff <- c(mse2EqualWeightsDiff, (MSE2 - MSE2.selective2))
        if(MSE2.selective2 < MSE2)
            nLowerMSE2EquallyWeighted <- nLowerMSE2EquallyWeighted + 1
        
        foundDouble <- 0

        if(noDuplicates)
        {
            # search for duplicates
            for(e2 in 1:e)
            {
                if(sum(sort(ensembleIndices[e2,1:nIndices]) == indOrdered) >= nIndices)
                    foundDouble <- 1        
            }
        }
        
        if(foundDouble)
        {
            cat("Ensemble indices are already in list, repeat\n")
            nDuplicates <- nDuplicates + 1
            if(nDuplicates >= 10)
                break # while
        }
        else
        {
            nDuplicates <- 0
            ensembleIndices[e,1:nIndices] <- indOrdered
            ensembleSizesVec <- c(ensembleSizesVec, nIndices)
            ensembleIndicesVec <- c(ensembleIndicesVec, indOrdered)
            ensembleWeightsVec <- c(ensembleWeightsVec, weightsOrdered)
            e <- e + 1
        }
    }

    if(verbose)
    {
        cat("Number of selective ensembles found: ", length(ensembleSizesVec), "\n")
        cat("Average number of agents in selective ensembles: ", mean(ensembleSizesVec), "\n")
        
        cat("Number of lower MSEs with selective ensembles: ", nLowerMSE, "\n")
        cat("Number of lower MSEs with selective ensembles (equal weights): ", nLowerMSEEquallyWeighted, "\n")

        cat("Sum of differences between MSE and MSE of selective ensemble: ", sumMSEDiff, "\n")
        cat("Sum of differences between MSE and MSE of selective ensemble (equal weights): ", sumMSEEqualWeightsDiff, "\n")
        
        cat("Number of lower MSE2s with selective ensembles: ", nLowerMSE2, "\n")
        cat("Number of lower MSE2s with selective ensembles (equal weights): ", nLowerMSE2EquallyWeighted, "\n")
            
        cat("Sum of differences between MSE2 and MSE2 of selective ensemble: ", sum(mse2Diff), "\n")
        cat("Sum of differences between MSE2 and MSE2 of selective ensemble (equal weights): ", sum(mse2EqualWeightsDiff), "\n")        
    }
    
    return(list(ensembleSizesVec = ensembleSizesVec, ensembleIndicesVec = ensembleIndicesVec, ensembleWeightsVec = ensembleWeightsVec, averageAgents = mean(ensembleSizesVec), averageMse2 = mean(mse2Diff), averageMse2EqualWeights = mean(mse2EqualWeightsDiff)))
}

calculateBellmanErrors <- function(threshold, C2, filePrefix, M, yPredicted, yReal, stateProbabilities, numberActions, maxActions)
{
    nStates <- length(numberActions)
    maxActions <- min(maxActions, max(numberActions))
    probs <- stateProbabilities

    if(C2 > 0)
        probs <- probs^C2 / sum(probs^C2)
    else
        probs <- rep(1 / nStates, nStates)

    errors3 <- array(NA, c(M, nStates, maxActions))

    for(m in 1:M)
    {
        lastInd <- 1
        for(i in 1:nStates)
        {
#            nActions <- numberActions[,i]
            nActions <- numberActions[i]

            ind <- lastInd:(lastInd + nActions - 1)

            o <- order(yPredicted[m,ind], decreasing = TRUE)

            nActionsMax <- min(nActions, maxActions)
            
            for(j in 1:nActionsMax)
            {
                action <- o[j]

                errReal <- abs(yPredicted[m,ind[action]] - yReal[m,ind[action]])

                if(threshold > 0)
                {
                    if(errReal > threshold)
                        err <- 1
                    else
                        err <- 0
                }
                else
                    err <- errReal

                errors3[m, i, j] <- probs[i] * nStates * err
            }

            lastInd <- lastInd + nActions
        }
    }

    errorsOther <- rep(0, M)
    errors <- matrix(0, M, maxActions)

    for(m in 1:M)
    {
        for(j in 1:maxActions)
            errors[m,j] <- mean(errors3[m,,j],na.rm=TRUE)

        errorsOther[m] <- mean(errors[m,-1])
    }

    return (list(errors=errors, errorsOther=errorsOther))
}

test <- function(fileToSave, filePrefix, cmdSelective, ensembleIndicesFile, MSeq, repoDir, averagedYRealSeq, averageSeq, methodSeq, t, equalWeightsSeq, alphaMinSeq, CMethodSeq, C2MethodSeq, actionsMaxSeq, thresholdEvalSeq, trueQuantileThresholdSeq, quantileThresholdSeq, quantileThresholdInvSeq, CSeq, C2Seq, useSingleRepo, greedySelectiveEnsemble, verbose = FALSE, maxT = Inf)
{
    results <- NULL

    ind <- 1
    T <- 1
    
    for(M in MSeq)
    {
        filename <- paste(ensembleIndicesFile,M,sep="")

        if(!file.exists(filename))
        {
            cat("file ", filename, " does not exist\n")
            return()
        }

        vals <- read.table(filename)
        nUniqueEnsembles <- nrow(vals) / M
        indicesMat <- t(matrix(t(vals),M,nUniqueEnsembles))

        eSeq <- 1:nUniqueEnsembles
#        eSeq <- c(1,2)

        retVA <- vector(mode = "list", length = nUniqueEnsembles)
        retVV <- vector(mode = "list", length = nUniqueEnsembles)

        if(greedySelectiveEnsemble)
        {
            cat("Greedy selective ensemble: Not reading values\n")

            # Get benchmark results
            
            ret <- readBenchmarkResults(max(vals), t)
            
            resvals <- ret$vals
            
            cat("Benchmark results:\n")
            print(resvals)
        }
        else
        {
            for(average in averageSeq)
            {
                for(e in eSeq)
                {
                    cat("Read values for committee",e,", average ", average, ":\n")

                    if(useSingleRepo == 1)
                        repoDirTmp <- paste(repoDir, "/", M, "/single/", e - 1, sep="")
                    else if(useSingleRepo == 2)
                        repoDirTmp <- paste(repoDir, "/", M, "/single-allAgents/", e - 1, sep="")                    
                    else if(average)
                        repoDirTmp <- paste(repoDir, "/", M, "/average/", e - 1, sep="")
                    else
                        repoDirTmp <- paste(repoDir, "/", M, "/voting/", e - 1, sep="")

                    ret <- readValues(repoDirTmp, indicesMat[e,] - 1, filePrefix, M, t)

                    if(average)
                        retVA[[e]] <- ret
                    else
                        retVV[[e]] <- ret
                } # e
            } # average
        }
        
        evaluateThreshold <- FALSE
        
        if((length(thresholdEvalSeq) > 1 || thresholdEvalSeq[1] != 0) && length(C2Seq) == 1)
        {
            evaluateThreshold <- TRUE
        
            # Get benchmark results
            
            ret <- readBenchmarkResults(max(vals), t)
            
            resvals <- ret$vals
            
            cat("Benchmark results:\n")
            print(resvals)
            
            thresholdMatVA <- rep(0, nUniqueEnsembles)
            thresholdMatVV <- rep(0, nUniqueEnsembles)

            corMatVA <- rep(0, nUniqueEnsembles)
            corMatVV <- rep(0, nUniqueEnsembles)
            
            for(average in averageSeq)
            {
                for(e in eSeq)
                {
                    cat("Evaluate threshold for committee",e,", average ", average, ":\n")
                
                    for(thresholdEval in thresholdEvalSeq)
                    {
                        if(verbose)
                            cat("Threshold: ", thresholdEval, "\n")

                        if(average)
                            ret2 <- retVA[[e]]
                        else
                            ret2 <- retVV[[e]]                            
                            
                        ret <- calculateBellmanErrors(thresholdEval, C2Seq[1], filePrefix, M, ret2$yPredicted, ret2$yReal, ret2$stateProbabilities, ret2$numberActions, 1)

                        errors <- ret$errors
                        errorsOther <- ret$errorsOther
                        ind <- indicesMat[e,]
                                                
                        tt <- cor.test(errors[,1], resvals[ind], method="kendall")
                        
                        if(average)
                        {
                            if(abs(tt$estimate) > corMatVA[e])
                            {
                                corMatVA[e] <- abs(tt$estimate)
                                thresholdMatVA[e] <- thresholdEval
                            }
                        }
                        else
                        {
                            if(abs(tt$estimate) > corMatVV[e])
                            {
                                corMatVV[e] <- abs(tt$estimate)
                                thresholdMatVV[e] <- thresholdEval
                            }
                        }
                        
                        if(verbose)
                        {
                            cat("Correlation between total reward and bellman error and p-value:\n")
                            print(tt$estimate)
                            print(tt$p.value)
                        }                        
                    } # thresholdEval
                    
                    if(average)
                        cat("Best threshold: ", thresholdMatVA[e], " best correlation: ", corMatVA[e], "\n")
                    else
                        cat("Best threshold: ", thresholdMatVV[e], " best correlation: ", corMatVV[e], "\n")
                } # e
            } # average
            
            cat("Correlations average:\n")
            print(corMatVA)
            cat("Best thresholds average:\n")
            print(thresholdMatVA)
            
            cat("Correlations voting:\n")
            print(corMatVV)
            cat("Best thresholds voting:\n")
            print(thresholdMatVV)            
        }

        for(averagedYReal in averagedYRealSeq)
        {
            for(trueQuantileThreshold in trueQuantileThresholdSeq)
            {
                for(quantileThreshold in quantileThresholdSeq)
                {
                    for(quantileThresholdInv in quantileThresholdInvSeq)
                    {
                        for(C in CSeq)
                        {
                            for(C2 in C2Seq)
                            {
                                cat("averagedYReal = ", averagedYReal, "\n")
                                cat("quantileThreshold = ", quantileThreshold, " quantileThresholdInv = ", quantileThresholdInv, " C = ", C, " C2 = ", C2, "\n", sep="")

                                for(method in methodSeq)
                                {
                                    for(C2Method in C2MethodSeq)
                                    {
                                        for(CMethod in CMethodSeq)
                                        {
                                            for(actionsMax in actionsMaxSeq)
                                            {
                                                for(alphaMin in alphaMinSeq)
                                                {
                                                    cat("method = ", method, " CMethod = ", CMethod, " C2Method = ", C2Method, " actionsMax = ", actionsMax, " alphaMin = ", alphaMin, "\n", sep="")

                                                    for(average in averageSeq)
                                                    {
                                                        ensembleSizesVec <- NULL
                                                        ensembleIndicesVec <- NULL
                                                        ensembleWeightsVec <- NULL
                                                        agentsVec <- NULL
                                                        mse2Vec <- NULL
                                                        mse2EqualWeightsVec <- NULL

                                                        for(e in eSeq)
                                                        {
                                                            T <- T + 1
                                                            if(T >= maxT)
                                                                return ()
                                                        
                                                            if(greedySelectiveEnsemble)
                                                            {
                                                                ind <- indicesMat[e,]

                                                                o <- order(resvals[ind], decreasing = TRUE)[1:CMethod]

                                                                cat("Best indices for Greedy Ensemble:\n")
                                                                print(ind[o])
                                                                
                                                                ensembleIndicesVec <- c(ensembleIndicesVec, ind[o])
                                                                ensembleWeightsVec <- c(ensembleWeightsVec, rep(1, CMethod))
                                                                ensembleSizesVec <- c(ensembleSizesVec, CMethod)

                                                                agentsVec <- c(agentsVec, CMethod)

                                                                mse2Vec <- c(mse2Vec, 0)
                                                                mse2EqualWeightsVec <- c(mse2EqualWeightsVec, 0)                                                            
                                                                nStates <- 0                                                             
                                                            }
                                                            else
                                                            {
                                                                if(average)
                                                                    retV <- retVA[[e]]
                                                                else
                                                                    retV <- retVV[[e]]
                                                                                                                                            
                                                                yPredicted <- retV$yPredicted
                                                                yReal <- retV$yReal
                                                                nStates <- retV$nStates
                                                                stateProb <- retV$stateProbabilities
                                                                nActions <- retV$numberActions
                                                                actionIndices <- retV$actionIndices

                                                                threshold <- 0
                                                                if(evaluateThreshold)
                                                                {
                                                                    if(average)
                                                                        threshold <- thresholdMatVA[e]
                                                                    else
                                                                        threshold <- thresholdMatVV[e]                                                                
                                                                }
                                                                
                                                                cat("Calculate errors for committee",e,":\n")

                                                                retE <- calculateErrors(yPredicted, yReal, M, nStates, absoluteError = TRUE, zeroMeanErrors = TRUE, unityVarianceErrors = TRUE, averagedYReal = averagedYReal, C = C, C2 = C2, verbose = verbose, thresholdCust = threshold, quantileThreshold = quantileThreshold, quantileThresholdInv = quantileThresholdInv, trueQuantileThreshold = trueQuantileThreshold)
                                                                                    
                                                                error <- retE$error
                                                                errorUnmodified <- retE$errorUnmodified
                                                                
                                                                # all M indices, 1 run
                                                                indicesMat2 <- matrix(1:M, 1, M)

                                                                cat("Perform selective ensemble learning for committee",e,":\n")

                                                                ret <- getBestIndices(method, M, M, nUniqueEnsembles, indicesMat2, error, errorUnmodified, yPredicted, yReal, stateProb, actionIndices, nActions, actionsMax, CMethod, C2Method, alphaMin, noDuplicates = TRUE, verbose = verbose)

                                                                o <- order(indicesMat[e,ret$ensembleIndicesVec])
                                                                ensembleIndicesVec <- c(ensembleIndicesVec, (indicesMat[e,ret$ensembleIndicesVec])[o])
                                                                ensembleWeightsVec <- c(ensembleWeightsVec, ret$ensembleWeightsVec[o])
                                                                
                                                                ensembleSizesVec <- c(ensembleSizesVec, ret$ensembleSizesVec)

                                                                agentsVec <- c(agentsVec, ret$averageAgents)

                                                                mse2Vec <- c(mse2Vec, ret$averageMse2)
                                                                mse2EqualWeightsVec <- c(mse2EqualWeightsVec, ret$averageMse2EqualWeights)
                                                            }                                                            
                                                        } # eSeq

                                                        averageMse2 <- mean(mse2Vec)
                                                        averageMse2EqualWeights <- mean(mse2EqualWeightsVec)

                                                        cat("Average number of agents in ensemble = ", mean(agentsVec), "\n")
                                                        
                                                        path <- paste(fileToSave, "-", ind, sep="")
                                                        
                                                        write.table(as.vector(t(ensembleSizesVec)), "selEnsembleSizes", col.names=FALSE,row.names=FALSE)
                                                        write.table(as.vector(t(ensembleIndicesVec)), "selEnsembleIndices", col.names=FALSE,row.names=FALSE)

                                                        for(equalWeights in equalWeightsSeq)
                                                        {
                                                            # Voting decisions need equal weights
                                                            if(!equalWeights && !average)
                                                                next # for
                                                        
                                                            if(equalWeights)
                                                            {
                                                                # WARNING: Overwrites the weights
                                                                write.table(round(as.vector(t(rep(1,length(ensembleWeightsVec)))),9), "selEnsembleWeights",col.names=FALSE,row.names=FALSE)
                                                            }
                                                            else
                                                            {
                                                                write.table(round(as.vector(t(ensembleWeightsVec)),9), "selEnsembleWeights",col.names=FALSE,row.names=FALSE)                
                                                            }
                                                                                                    
                                                            nRuns <- length(ensembleSizesVec)

                                                            cat("equalWeights = ", equalWeights, " average = ", average, "\n")
                                                            
                                                            if(average)
                                                            {
                                                                if(file.exists("selEnsembleAverageResults"))
                                                                {
                                                                    file.remove("selEnsembleAverageResults")
                                                                }
                                                            }
                                                            else
                                                            {
                                                                if(file.exists("selEnsembleVotingResults"))
                                                                {
                                                                    file.remove("selEnsembleVotingResults")
                                                                }                                            
                                                            }
                                                            
                                                            # perform benchmark

                                                            cat("Start benchmark:\n")

                                                            ## only test the last
                                                            cmdLine <- paste(cmdSelective, " ", t, " ", t, " 1 ", as.numeric(average), " ", path, sep="")
                                                                
                                                            if(verbose)
                                                                system(cmdLine, intern=FALSE)
                                                            else
                                                                out <- system(cmdLine, intern=TRUE)

                                                            if(average)
                                                            {
                                                                if(!file.exists("selEnsembleAverageResults"))
                                                                {
                                                                    cat("Benchmark failed (result file does not exist)\n")
                                                                
                                                                    return()
                                                                }
                                                            }
                                                            else
                                                            {
                                                                if(!file.exists("selEnsembleVotingResults"))
                                                                {
                                                                    cat("Benchmark failed (result file does not exist)\n")
                                                                
                                                                    return()
                                                                }                                            
                                                            }
                                                            
                                                            # read results

                                                            cat("Benchmark results:\n")
                                                            if(average)
                                                            {
                                                                vals <- as.vector(t(read.table("selEnsembleAverageResults")))
                                                                cat("Average:\n")
                                                                print(vals)
                                                            }
                                                            else
                                                            {
                                                                vals <- as.vector(t(read.table("selEnsembleVotingResults")))
                                                                cat("Voting:\n")
                                                                print(vals)                                            
                                                            }
                                                            
                                                            results <- rbind( results, list(M = M, repo = repoDir, t = t, nStates = nStates, averageDecision = average, method = method, CMethod = CMethod, C2Method = C2Method, actionsMax = actionsMax, equalWeights = equalWeights, alphaMin = alphaMin, trueQuantileThreshold = trueQuantileThreshold, quantileThreshold = quantileThreshold, quantileThresholdInv = quantileThresholdInv, C = C, C2 = C2, T = nRuns, committeeSizeAvg = mean(agentsVec), committeeSizeSd = sd(agentsVec), mse2 = averageMse2, mse2Eq = averageMse2EqualWeights, benchmarkMean = vals[1], benchmarkSd = vals[2], pValue1=vals[3], pValue2=vals[4], pValue3=vals[5]))

                                                            write.table(results, file=fileToSave)
                                                            
                                                            ind <- ind + 1
                                                        } # equalWeights
                                                    } # average
                                                } # alphaMin
                                            } # actionsMaxSeq
                                        } # CMethod
                                    } # C2Method
                                } # method
                            } # C2
                        } # C
                    } # quantileThresholdInv
                } # quantileThreshold
            } # trueQuantileThreshold
        } # averagedYReal
    } # M
}

selectiveEnsembleLearning_fixedSize <- function(Dmat, K)
{
    # finds >= K indices which minimize the correlation of the errors

    M <- nrow(Dmat)
    
    # (1, left hand) sum of weights
    Amat <- matrix(1, M, 1)

    # (2, left hand)  w_1, ..., w_M
    for (i in 1:M)
    {
        nMat <- rep(0, M)
        nMat[i] <- 1
        Amat <- cbind(Amat, nMat)
    }

    # (3, left hand)  w_1, ..., w_M
    for (i in 1:M)
    {
        nMat <- rep(0, M)
        nMat[i] <- -1
        Amat <- cbind(Amat, nMat)
    }

    dvec <- rep(0,M)

    # (1, right hand) = M
    bvec <- 1

    # (2, right hand) >= 0
    bvec <- c(bvec, rep(0, length=M))

    # (3, right hand) <= 1 / K
    bvec <- c(bvec, rep(- 1 / K, length=M))

    # solve quadratic programming problem (by setting alpha)
    solv <- solve.QP(Dmat, dvec, Amat, bvec=bvec, meq=1)
    # get alphas
    alpha <- solv$solution

    return(list(weights = alpha))
}

selectiveEnsembleLearning <- function(error, yProb, actionIndices, nActionsVec, nActionsMax, yPredicted, C, C2, alphaMin, build = 1)
{
    M <- nrow(error)
    N <- ncol(error)
    
    if(C2 > 0)
        yProb <- yProb^C2 / sum(yProb^C2)
    else
        yProb <- yProb / sum(yProb)
            
    Dmat <- matrix(0, M, M)

    if(build == 1 || build == 3 || build == 5)
    {
        # Voting:
        ## Warning: weighted_mean = 0 is NOT equivalent to weighted_mean = 1
        weighted_mean <- 1
        mode <- 1 # best action gets 'weight_best_action', all others share '1 - weight_best_action'
        weight_best_action <- 1
        theta <- -1 # ignored
        err <- abs(error)

        values <- .C("cost_matrix", Dmat=as.numeric(Dmat), error=as.numeric(err), vals=as.numeric(yPredicted), actionindices=as.integer(actionIndices), state_weights=as.numeric(yProb), actions=as.integer(nActionsVec), actionsMax=as.integer(nActionsMax), M=as.integer(M), N=as.integer(length(nActionsVec)), weighted_mean=as.integer(weighted_mean), mode=as.integer(mode), theta=as.numeric(theta), weight_best_action=as.numeric(weight_best_action))
    }
    else if(build == 2 || build == 4 || build == 6)
    {
        # Average:
        weighted_mean <- 1
        mode <- 2 # ranked weighted
        weight_best_action <- 1 # ignored
        theta <- -1 # ignored
        err <- abs(error)

        values <- .C("cost_matrix", Dmat=as.numeric(Dmat), error=as.numeric(err), vals=as.numeric(yPredicted), actionindices=as.integer(actionIndices), state_weights=as.numeric(yProb), actions=as.integer(nActionsVec), actionsMax=as.integer(nActionsMax), M=as.integer(M), N=as.integer(length(nActionsVec)), weighted_mean=as.integer(weighted_mean), mode=as.integer(mode), theta=as.numeric(theta), weight_best_action=as.numeric(weight_best_action))
    }

    ret <- list(Dmat = matrix(values$Dmat, M, M))

    Dmat <- ret$Dmat
    
    fitness <- function(x, Dmat)
    {
        ind <- which(x == 1)

        if(length(ind) == 0)
            return(-Inf)

        totalcost <- 0
        for(i in 1:length(ind))
        {
            for(j in 1:length(ind))
                totalcost <- totalcost + Dmat[ind[i],ind[j]] / length(ind)^2
        }

        return(-totalcost)
    }
    
    if(build == 5 || build == 6)
    {
        # GA, fixed number of agents
        
        gabin_spCrossover_keepNumberOfOnes <- function (object, parents, ...)
        {
            fitness <- object@fitness[parents]
            parents <- object@population[parents, , drop = FALSE]
            n <- ncol(parents)
            children <- matrix(NA, nrow = 2, ncol = n)
            fitnessChildren <- rep(NA, 2)
            crossOverPoint <- sample(0:n, size = 1)
            if (crossOverPoint == 0) {
                children[1:2, ] <- parents[2:1, ]
                fitnessChildren[1:2] <- fitness[2:1]
            }
            else if (crossOverPoint == n) {
                children <- parents
                fitnessChildren <- fitness
            }
            else {
                children <- parents
                fitnessChildren <- fitness
                ind1 <- which(parents[1,] == 1)
                ind2 <- which(parents[2,] == 1)

                if(setequal(ind1, ind2))
                {
                    # no need for doing a crossover operation, return childs identical to the parents ('as father as son')
                    children <- parents
                    fitnessChildren <- fitness
                }
                else
                {
                    indDup <- intersect(ind1,ind2)

                    ind1Unique <- setdiff(ind1, indDup)
                    ind2Unique <- setdiff(ind2, indDup)

                    ind1New <- c(indDup, ind1Unique)
                    ind2New <- c(indDup, ind2Unique)
            
                    len <- length(ind1New)

                    # get new crossover point
                    crossOverPoint <- sample(1:(len-1), size = 1)

                    indChild1 <- c(ind1New[1:crossOverPoint], ind2New[
                        (crossOverPoint + 1):len])

                    indChild2 <- c(ind2New[1:crossOverPoint], ind1New[
                        (crossOverPoint + 1):len])

                    children[1, ] <- rep(0, n)
                    children[2, ] <- rep(0, n)

                    children[1, indChild1] <- 1
                    children[2, indChild2] <- 1

                    fitnessChildren <- NA
                }
            }
            out <- list(children = children, fitness = fitnessChildren)
            return(out)
        }

        gabin_raMutation_keepNumberOnes <- function (object, parent, ...)
        {
            mutate <- parent <- as.vector(object@population[parent, ])

            indOnes <- which(parent == 1)
            indZeroes <- which(parent == 0)

            # mutate two chromosomes, keeping the number of ones

            i <- sample(indOnes, size = 1)
            j <- sample(indZeroes, size = 1)

            mutate[i] <- 0
            mutate[j] <- 1

            return(mutate)
        }

        gabin_Population_fixedNumberOnes <- function (object, ...)
        {
            population <- matrix(NA, nrow = object@popSize, ncol = object@nBits)
            for (i in 1:object@popSize)
            {
                # attention: C is defined outside the function ('global' variable)
                ind <- sample(1:object@nBits, C)
                population[i, ] <- rep(0, object@nBits)
                population[i, ind] <- 1
            }
            return(population)
        }
        
        # Linear rank selection
        GA <- ga(type="binary", fitness = fitness, nBits = M, maxiter = 200, popSize = 100, pcrossover = 0.8, pmutation = 0.2, selection = gabin_lrSelection, crossover = gabin_spCrossover_keepNumberOfOnes, population = gabin_Population_fixedNumberOnes, mutation = gabin_raMutation_keepNumberOnes, seed = 1, Dmat = Dmat)
        print(summary(GA))
        
        ind <- which(GA@solution == 1)
        
        totalcost <- 0
        for(i in 1:length(ind))
        {
            for(j in 1:length(ind))
                totalcost <- totalcost + Dmat[ind[i],ind[j]] / length(ind)^2
        }        
    }
    else if(build == 1 || build == 2 || build == 3 || build == 4)
    {
        ret <- selectiveEnsembleLearning_fixedSize(Dmat, C)
        
        if(build == 1 || build == 2)
        {
            # choose first C with highest weights
            
            o <- order(ret$weights, decreasing = TRUE)

            ind <- (1:M)[o]
            weights <- ret$weights[o]

            ind <- ind[1:C]
            weights <- weights[1:C]
        }
        else
        {
            o <- ret$weights >= 0.7 * 1 / C
            ind <- (1:M)[o]
            weights <- ret$weights[o]        
        }

        totalcost <- 0
        for(i in 1:length(ind))
        {
            for(j in 1:length(ind))
            {
                totalcost <- totalcost + Dmat[ind[i],ind[j]] / length(ind)^2
            }
        }        
    }
    else
    {
        cat("### Wrong build number\n")
        return(NULL)
    }

    cat("Build = ", build, ", found ", length(ind), " agents in ensemble\n")
    cat("totalcost = ", totalcost, "\n")    
    print(ind)
    
    return(list(indices = ind, weights = rep(1,length(ind))))
}    

