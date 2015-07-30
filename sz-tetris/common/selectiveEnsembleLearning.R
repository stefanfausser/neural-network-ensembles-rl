source("../../common/selectiveEnsembleLearningLib.R")
dyn.load("../../common/cost_matrix.so")
dyn.load("../../common/calcMSE.so")
dyn.load("../../common/calcEnsembleMSE.so")
dyn.load("../../common/readFileAsVector.so")
dyn.load("../../common/getFileLines.so")

performTest <- function(logfile, t, voting = FALSE, useSingleRepo = FALSE, greedySelectiveEnsemble = FALSE, maxT = Inf, firstFour = FALSE)
{
    ensembleIndicesFile <- "../../common/ensembleIndices2_" # 50 runs, 100 agents

    repoDir <- paste("repo_it", t, sep="")

    cmdSelective <- "bash startElEval_selective.sh"

    MSeq <- 50
    averagedYRealSeq <- c(FALSE)
    if(voting)
    {
        methodSeq <- c(1) # voting
        averageSeq <- FALSE
    }
    else
    {
        methodSeq <- c(2) # average
        averageSeq <- TRUE
    }

    if(greedySelectiveEnsemble)
        trueQuantileThresholdSeq <- c(0)
    else
    {
        if(useSingleRepo)
            trueQuantileThresholdSeq <- c(0.1, 0.3, 0.5, 0.7, 0.9)                
        else
        {
            if(length(grep("tdlambda",basename(getwd()))))
            {
                # TD-lambda
                if(voting)
                    trueQuantileThresholdSeq <- c(0.2,0.3,0.4,0.5,0.6,0.7,0.8)
                else
                    trueQuantileThresholdSeq <- c(0.4,0.5,0.6,0.7,0.8)
            }
            else
            {
                # TDC            
                trueQuantileThresholdSeq <- c(0.3,0.4,0.5,0.6,0.7,0.8)
            }
        }
    }

    if(useSingleRepo)
        CMethodSeq <- c(5,10,15,20,25,30) # number of agents
    else if(length(grep("tdlambda",basename(getwd()))) && !voting && t == 20)
        CMethodSeq <- c(5,10,15,20,25,30,35,40) # number of agents
    else
        CMethodSeq <- c(5,10,15,20,25,30,35) # number of agents
        
    C2MethodSeq <- c(0.75) # probability exponent
    actionsMaxSeq <- c(0) # limit the number of best actions        
    thresholdEvalSeq <- c(0)
    quantileThresholdSeq <- c(0)
    quantileThresholdInvSeq <- c(0)
    CSeq <- 0 # unit variances
    C2Seq <- 0 # mean errors
    alphaMinSeq <- c(0.000000001)
    equalWeightsSeq <- TRUE

    if(firstFour)
    {
        trueQuantileThresholdSeq <- trueQuantileThresholdSeq[1]
        CMethodSeq <- CMethodSeq[1:4]
    }
    
    verbose <- FALSE

    test(paste(logfile, repoDir, sep=""), "tetris", cmdSelective, ensembleIndicesFile, MSeq, repoDir, averagedYRealSeq, averageSeq, methodSeq, t, equalWeightsSeq, alphaMinSeq, CMethodSeq, C2MethodSeq, actionsMaxSeq, thresholdEvalSeq, trueQuantileThresholdSeq, quantileThresholdSeq, quantileThresholdInvSeq, CSeq, C2Seq, useSingleRepo, greedySelectiveEnsemble, verbose, maxT)
}

performTests <- function(tSeq = c(10,20,60,100), greedy = FALSE, maxT = Inf, firstFour = FALSE)
{
    # States were collected with a large ensemble
    
    for (t in tSeq)
        performTest(voting = FALSE, t, logfile = "results1-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)

    for (t in tSeq)
        performTest(voting = TRUE, t, logfile = "results1-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)
    
    # States were collected by all agents with independent / single decisions
    
    for (t in tSeq)
    {
        performTest(voting = FALSE, useSingleRepo = 2, t, logfile = "results1-singleDecisions-allAgents-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)
        performTest(voting = TRUE, useSingleRepo = 2, t, logfile = "results1-singleDecisions-allAgents-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)
    }
    
    if(greedy)
    {
        # Greedy Selective Ensemble
    
        for (t in tSeq)
        {
            performTest(voting = FALSE, greedySelectiveEnsemble = TRUE, t, logfile = "results1-greedy-noRandomStarts-average-weightedMean-rankedWeighted-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)
            performTest(voting = TRUE, greedySelectiveEnsemble = TRUE, t, logfile = "results1-greedy-noRandomStarts-voting-weightedMean-trueQuantileThreshold-policy-", maxT = maxT, firstFour = firstFour)
        }
    }
}
