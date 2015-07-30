source("startElEval_env.R")

averageSeq <- c(TRUE, FALSE)
useSingleRepoSeq <- c(FALSE, TRUE)
singleThreshold <- FALSE

bestScoresLast <- NULL

for(average in averageSeq)
{
    for(useSingleRepo in useSingleRepoSeq)
    {
        iterations <- len
        repoDir <- paste("repo_it", iterations, sep="")

        # Get the benchmark results for the randomly-selective ensemble

        runs <- 50
        randomCommitteeSizes <- c(5, 10, 20, 30, 50)

        scoresRC <- rep(0, length(randomCommitteeSizes))

        for(i in 1:length(randomCommitteeSizes))
        {
            if(randomCommitteeSizes[i] == 3 && average)
            {
                numberDir <- 2
            } else if(randomCommitteeSizes[i] == 3 && !average)
            {
                numberDir <- 3
            } else if(randomCommitteeSizes[i] == 5 && average)
            {
                numberDir <- 10
            } else if(randomCommitteeSizes[i] == 5 && !average)
            {
                numberDir <- 11
            } else if(randomCommitteeSizes[i] == 10 && average)
            {
                numberDir <- 12
            } else if(randomCommitteeSizes[i] == 10 && !average)
            {
                numberDir <- 13
            } else if(randomCommitteeSizes[i] == 20 && average)
            {
                numberDir <- 16
            } else if(randomCommitteeSizes[i] == 20 && !average)
            {
                numberDir <- 17
            } else if(randomCommitteeSizes[i] == 30 && average)
            {
                numberDir <- 18
            } else if(randomCommitteeSizes[i] == 30 && !average)
            {
                numberDir <- 19
            } else if(randomCommitteeSizes[i] == 50 && average)
            {
                numberDir <- 20
            } else if(randomCommitteeSizes[i] == 50 && !average)
            {
                numberDir <- 21
            }

            vals <- NULL
            
            for(j in 1:runs)
            {
                filename2 <- paste("experiment-sampledindices", numberDir, "/log", j - 1, "_", iterations, file_suffix, sep="")
                vals <- c(vals, as.numeric(read.table(filename2)))
            }
            
            scoresRC[i] <- mean(vals)
        }

        if(average)
        {
            if(useSingleRepo)
                file <- paste("results1-singleDecisions-allAgents-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-", repoDir, sep="")
            else
                file <- paste("results1-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-", repoDir, sep="")
        } else
        {
            if(useSingleRepo)
                file <- paste("results1-singleDecisions-allAgents-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-", repoDir, sep="")
            else
                file <- paste("results1-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-", repoDir, sep="")
        }

        tab <- read.table(file)

        trueQuantileThresholdSeq <- unique(tab$trueQuantileThreshold)
        committeeSizesSeq <- unique(tab$committeeSizeAvg)

        scores <- tab$benchmarkMean
        
        # Get best threshold for each committee size

        bestScores <- rep(0, length(committeeSizesSeq))
        bestThresholds <- rep(0, length(committeeSizesSeq))

        for(i in 1:length(committeeSizesSeq))
        {
            L <- committeeSizesSeq[i]
            ind <- which.max(scores[tab$committeeSizeAvg == L])
            bestScores[i] <- (scores[tab$committeeSizeAvg == L])[ind]
            bestThresholds[i] <- trueQuantileThresholdSeq[ind]
        }

        if(singleThreshold)
        {
            bestThreshold <- bestThresholds[which.max(bestScores)]
            bestThresholdInd <- which(trueQuantileThresholdSeq == bestThreshold)

            bestScores <- rep(0, length(committeeSizesSeq))
            
            for(i in 1:length(committeeSizesSeq))
            {
                L <- committeeSizesSeq[i]
                bestScores[i] <- (scores[tab$committeeSizeAvg == L])[bestThresholdInd]
                bestThresholds[i] <- bestThreshold
            }
        }
        
        cat("Best thresholds:\n")
        print(bestThresholds)        
        
        # get RL method and exploration method from working directory
        
        rlMethod <- toupper(strsplit(basename(getwd()),"-")[[1]][3])
        explorationMethod <- strsplit(basename(getwd()),"-")[[1]][4]
        
        if(explorationMethod != "softmax")
            explorationMethod <- "epsilon-greedy"
        
        if(average)
        {
            mainlab <- paste(rlMethod, ", ",  explorationMethod, ", Average", sep="")
        } else
        {
            mainlab <- paste(rlMethod, ", ",  explorationMethod, ", Voting", sep="")
        }

        ylim <- c(min(c(bestScores, bestScoresLast, scoresRC)), max(c(bestScores, bestScoresLast, scoresRC)))
        xlim <- c(min(c(committeeSizesSeq, randomCommitteeSizes)), max(c(committeeSizesSeq, randomCommitteeSizes)))

        nVals <- 5        
        yStep <- round((max(ylim) - min(ylim)) / (nVals - 1),digits=parameterPlotNDigits)        
        yMax <- round(max(ylim),digits=parameterPlotNDigits)
        yMin <- yMax - (nVals - 1) * yStep        
        yLabelSeq <- seq(yMax, yMin, -yStep)

        setEPS()
        
        addStr <- NULL
        if(singleThreshold)
            addStr <- "_singleThreshold"
        
        if(average)
        {
            if(useSingleRepo)
                fileOut <- paste(confName, addStr, "_committeeSizes_single_average_it", iterations, ".eps", sep="")
            else
                fileOut <- paste(confName, addStr, "_committeeSizes_average_it", iterations, ".eps", sep="")
        }
        else
        {
            if(useSingleRepo)
                fileOut <- paste(confName, addStr, "_committeeSizes_single_voting_it", iterations, ".eps", sep="")
            else
                fileOut <- paste(confName, addStr, "_committeeSizes_voting_it", iterations, ".eps", sep="")
        }

        postscript(fileOut)
        if(useSingleRepo)
        {
            plot(committeeSizesSeq, bestScores, type='o', lwd=1.5, , lty = "dotted", col='black', ylab=plotYLab, xlab="committee size", main=mainlab, pch=21, cex.lab=2.0, cex.axis=2.0, cex.main=2.0, cex.sub=1.7, cex=2.0, ylim=ylim, xlim=xlim, xaxt="n", yaxt="n")
            axis(1, at=parameterPlotXAxisAt, labels=parameterPlotXAxisLabels, cex.axis=2.0)
            axis(2, at=yLabelSeq, labels=yLabelSeq, cex.axis=2.0)
#            text(committeeSizesSeq, bestScores, as.character(bestThresholds),cex=1.7,adj=c(0.5,1.5))
            lines(committeeSizesSeq[1:length(bestScoresLast)], bestScoresLast, type='o', lwd=1.5, lty = "dashed", col='black', pch=18, cex=2.0)
#            text(committeeSizesSeq, bestScoresLast, as.character(bestThresholdsLast),cex=1.7,adj=c(0.5,1.5))            
            lines(randomCommitteeSizes, scoresRC, type='o', lwd=1.5, lty = "solid", col='grey', pch=24, cex=2.0)
            legend("bottomright", lwd=c(1.5,1.5,1.5), pch=c(21,18,24), col=c('black', 'black', 'grey'), lty=c("dotted","dashed","solid"), legend=c("sel. ens. (single)", "sel. ens. (full)", "large ensemble"), cex=2.0)
        }        
        else
        {        
            plot(committeeSizesSeq, bestScores, type='o', lwd=1.5, , lty = "dotted", col='black', ylab=plotYLab, xlab="committee size", main=mainlab, pch=21, cex.lab=2.0, cex.axis=2.0, cex.main=2.0, cex.sub=1.7, cex=2.0, ylim=ylim, xlim=xlim, xaxt="n", yaxt="n")
            axis(1, at=parameterPlotXAxisAt, labels=parameterPlotXAxisLabels, cex.axis=2.0)
            axis(2, at=yLabelSeq, labels=yLabelSeq, cex.axis=2.0)
            text(committeeSizesSeq, bestScores, as.character(bestThresholds),cex=1.7,adj=c(0.5,1.5))
            lines(randomCommitteeSizes, scoresRC, type='o', lwd=1.5, lty = "solid", col='grey', pch=24, cex=2.0)
            legend("bottomright", lwd=c(1.5,1.5), pch=c(21,24), col=c('black', 'grey'), lty=c("dotted","solid"), legend=c("selective ensemble", "large ensemble"), cex=2.0)
        }        
        dev.off()
        
        bestScoresLast <- bestScores
        bestThresholdsLast <- bestThresholds
    } # useSingleRepo
} # average
