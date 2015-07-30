library(data.table)

myttest <- function(sampleMean, sampleSd, sampleSize)
{
    # Welch's t-test (unpaired t-test with unequal variances)
    
    t <- (sampleMean[1] - sampleMean[2]) / sqrt(sampleSd[1]^2 / sampleSize[1] + sampleSd[2]^2 / sampleSize[2])
    
    df <- (sampleSd[1]^2 / sampleSize[1] + sampleSd[2]^2 / sampleSize[2])^2 / (sampleSd[1]^4 / (sampleSize[1]^2 * (sampleSize[1] - 1)) + sampleSd[2]^4 / (sampleSize[2]^2 * (sampleSize[2] - 1)))
    
    p <- 2 * pt(-abs(t), df=df)
    
    return(list(t=t, df=df, p.value=p))
}

calculateStatSignificance <- function(e1, e2, nTestruns1, nTestruns2, t1, t2, vals, verbose = FALSE)
{
    pValue <- 0

    eVec <- NULL

    rewardVec <- NULL

    s1 <- 1:nTestruns1

    for(i in s1)
    {
        eVec <- c(eVec, e1)

        v1 <- as.vector(vals[e1, t1, i])
        rewardVec <- c(rewardVec, v1)
    }

    s2 <- 1:nTestruns2

    for(i in s2)
    {
        eVec <- c(eVec, e2)

        v2 <- as.vector(vals[e2, t2, i])
        rewardVec <- c(rewardVec, v2)
    }
    
    d <- data.frame(eVec, rewardVec)
    colnames(d) <- c("Experiment","Reward")
    a <- aov(Reward ~ Experiment, d)

    pValue <- summary(a)[[1]][1,5]

    return(list(pValue = pValue))
}

getEnsembleParams <- function(e, nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2)
{
    if(e == 1)
    {
        lenTmp <- len
        nTestrunsTmp <- nTestrunsSingle
    }
    else if((e >= 2 && e <= 3) || (e >= 10 && e <= 13) || (e >= 16 && e <= 21))
    {
        lenTmp <- len
        nTestrunsTmp <- nTestrunsEnsemble
    }
    else if(e == 14 || e == 15)
    {
        lenTmp <- len
        nTestrunsTmp <- nTestrunsSelectiveEnsemble
    }
    else if(e == 4 || e == 5 || e == 6 || e == 7 || e == 8 || e == 9 || e == 22 || e == 23)
    {
        lenTmp <- len2
        nTestrunsTmp <- nTestrunsEnsemblePolicyEnsemble
    }
    else
    {
        # Error
    }

    return(list(len=lenTmp, nTestruns=nTestrunsTmp))
}

getResultsMain <- function(file_suffix, experimentSeq, experimentSeq2, experimentStart, experimentBy, tSeq, nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2, isSelEnsemble)
{
    if(len < len2)
    {
        # Error
    }

    if(sum(is.element(experimentSeq2, experimentSeq)) != length(experimentSeq2))
    {
        # Error
    }

    if(!isSelEnsemble)
    {
        # check tSeq
        if(length(tSeq) != 4)
        {
            # Error
        }        
    }

    nTestrunsMax <- max(nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble)

    maxExperiments <- max(experimentSeq)
    
    vals <- array(0, dim=c(maxExperiments, len, nTestrunsMax))

    if(!isSelEnsemble)
    {
        totalReward <- matrix(0, length(tSeq), maxExperiments)
        totalRewardSd <- matrix(0, length(tSeq), maxExperiments)

        meanVals <- matrix(0, maxExperiments, len)
        maxVals <- matrix(0, maxExperiments, len)    
    }
    
    # read values from the agent(s)
    for(e in experimentSeq)
    {
        cat("e = ", e, "\n", sep="")
     
        ret <- getEnsembleParams(e, nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2)
       
        lenTmp <- ret$len
        nTestrunsTmp <- ret$nTestruns

        if(isSelEnsemble)
            iterationsSeq <- len
        else
            iterationsSeq <- seq(experimentStart, lenTmp, experimentBy)

        if(e == 2 || e == 3 || e == 10 || e == 11 || e == 12 || e == 13 || e == 16 || e == 17 || e == 18 || e == 19 || e == 20 || e == 21)
            path_prefix <- "experiment-sampledindices"
        else
            path_prefix <- "experiment"

        for (i in 0:(nTestrunsTmp - 1))
        {
            for(t in iterationsSeq)
            {
                logfile <- paste(path_prefix, e, "/log", i, "_", t, file_suffix, sep="")

                if(!file.exists(logfile))
                {
                    cat("file ", logfile, " does not exist\n")
                }

#                val <- as.numeric(read.table(logfile))
                val <- as.numeric(fread(logfile))

                vals[e, t, i + 1] <- val
            } # t
        } # i

        if(isSelEnsemble)
        {
            if(e == 14 || e == 15)
            {
                val <- as.vector(vals[e,len,1:nTestrunsTmp])

                totalReward <- mean(val)
                totalRewardSd <- sd(val)            

                meanVals <- totalReward
                maxVals <- totalReward
            }
        }
        else
        {
            for(i in 1:length(tSeq))
            {
                val <- as.vector(vals[e, tSeq[i], 1:nTestrunsTmp])

                totalReward[i, e] <- mean(val)
                totalRewardSd[i, e] <- sd(val)
            }

            for(t in iterationsSeq)
            {
                val <- as.vector(vals[e, t, 1:nTestrunsTmp])

                meanVals[e,t] <- mean(val)
                maxVals[e,t] <- max(val)
            }
        }
    } # e

    if(isSelEnsemble)
        pValues <- rep(0, length(experimentSeq2) - 1)
    else
        pValues <- matrix(0, length(experimentSeq2) - 1, 6)

    e1 <- experimentSeq2[1]
    
    ret <- getEnsembleParams(e1, nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2)    
    
    nTestruns1 <- ret$nTestruns

    for(ind in 2:length(experimentSeq2))
    {
        e2 <- experimentSeq2[ind]

        ret <- getEnsembleParams(e2, nTestrunsSingle, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2)
       
        nTestrunsTmp <- ret$nTestruns

        if(isSelEnsemble)
            pValues[ind-1] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, len, len, vals, verbose = FALSE)$pValue
        else
        {
            pValues[ind-1, 1] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[1], tSeq[1], vals, verbose = FALSE)$pValue
            pValues[ind-1, 2] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[2], tSeq[2], vals, verbose = FALSE)$pValue
            pValues[ind-1, 3] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[3], tSeq[3], vals, verbose = FALSE)$pValue
            pValues[ind-1, 4] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[4], tSeq[4], vals, verbose = FALSE)$pValue
            pValues[ind-1, 5] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[3], tSeq[2], vals, verbose = FALSE)$pValue
            pValues[ind-1, 6] <- calculateStatSignificance(e1, e2, nTestruns1, nTestrunsTmp, tSeq[4], tSeq[2], vals, verbose = FALSE)$pValue
        }
    }

    return(list(vals=vals, meanVals=meanVals, maxVals=maxVals, totalReward=totalReward, totalRewardSd=totalRewardSd, pValues=pValues))
}
