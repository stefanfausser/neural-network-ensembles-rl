source("../../common/startElEval_process.R")
source("startElEval_env.R")
pathEnsembleIndices <- "../../common/ensembleIndices2_"

experimentSeq2 <- s2

nAgents <- 3
nAgents2 <- 5
nAgents3 <- 10

nTestrunsSingleReal <- 100 # runs for single agent
nTestrunsEnsemble <- 50 # runs for non-selective non-in-ensemble learning ensemble
nTestrunsEnsemblePolicyEnsemble <- 20 # runs for in-ensemble learning ensemble
nTestrunsSelectiveEnsemble <- 0

isSelEnsemble <- 0

tSeq <- c(len / 5 / 2, len / 5, len / 5 * 3, len)

ret <- getResultsMain(file_suffix, experimentSeq, experimentSeq2, experimentStart, experimentBy, tSeq, nTestrunsSingleReal, nTestrunsEnsemble, nTestrunsSelectiveEnsemble, nTestrunsEnsemblePolicyEnsemble, len, len2, isSelEnsemble)

# reihenfolge wie in paper
s <- s2

# results for 2.5M episodes
cat("Results for 2.5M episodes:\n");
print(round(ret$totalReward[1,s],digits=nDigits))
print(round(ret$totalRewardSd[1,s],digits=nDigits))

# results for 5M episodes
cat("Results for 5M episodes:\n");
print(round(ret$totalReward[2,s],digits=nDigits))
print(round(ret$totalRewardSd[2,s],digits=nDigits))

# results for 15M episodes
cat("Results for 15M episodes:\n");
print(round(ret$totalReward[3,s],digits=nDigits))
print(round(ret$totalRewardSd[3,s],digits=nDigits))

# results for 25M episodes
cat("Results for 25M episodes:\n");
print(round(ret$totalReward[4,s],digits=nDigits))
print(round(ret$totalRewardSd[4,s],digits=nDigits))

cat("p-values, comparison of 5M to 15M values (fair):\n")
print(ret$pValues[,5])
print(ret$pValues[,5] < 0.001)
print(ret$pValues[,5] < 0.01)

cat("p-values, comparison of 5M to 25M values (fair):\n")
print(ret$pValues[,6])
print(ret$pValues[,6] < 0.001)
print(ret$pValues[,6] < 0.01)

# build latex table entries
str <- NULL
for(i in 1:length(s))
{
    if(s[i] == 1)
    {
        M <- 1
        piTrain <- "S"
        piBench <- "S"
    }
    else if(s[i] == 2)
    {
        M <- 3
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 3)
    {
        M <- 3
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 4)
    {
        M <- 5
        piTrain <- "A"
        piBench <- "A"
    }
    else if(s[i] == 5)
    {
        M <- 5
        piTrain <- "V"
        piBench <- "V"
    }
    else if(s[i] == 6)
    {
        M <- 5
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 8)
    {
        M <- 5
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 10)
    {
        M <- 5
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 11)
    {
        M <- 5
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 12)
    {
        M <- 10
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 13)
    {
        M <- 10
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 16)
    {
        M <- 20
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 17)
    {
        M <- 20
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 18)
    {
        M <- 30
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 19)
    {
        M <- 30
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 20)
    {
        M <- 50
        piTrain <- "S"
        piBench <- "A"
    }
    else if(s[i] == 21)
    {
        M <- 50
        piTrain <- "S"
        piBench <- "V"
    }
    else if(s[i] == 22)
    {
        M <- 3
        piTrain <- "A"
        piBench <- "A"
    }
    else if(s[i] == 23)
    {
        M <- 3
        piTrain <- "V"
        piBench <- "V"
    }

    str <- paste(str, paste("TD & ", "$", M, "$ & ", piTrain, " & ", piBench, 
    " & $", round(ret$totalReward[1,s[i]],digits=nDigits), " \\pm ", round(ret$totalRewardSd[1,s[i]],digits=nDigits), "$",
    " & $", round(ret$totalReward[2,s[i]],digits=nDigits), " \\pm ", round(ret$totalRewardSd[2,s[i]],digits=nDigits), "$",
    " & $", round(ret$totalReward[3,s[i]],digits=nDigits), " \\pm ", round(ret$totalRewardSd[3,s[i]],digits=nDigits), "$",
    " & $", round(ret$totalReward[4,s[i]],digits=nDigits), " \\pm ", round(ret$totalRewardSd[4,s[i]],digits=nDigits), "$",
    "\\\\ \n", sep=""))    
}

# Get the results when one selects the best (1-3) agent(s) out of 50 available

M_forBest <- 5
nTop <- 1
v <- read.table(paste(pathEnsembleIndices, M_forBest, sep=""))
nUniqueEnsembles <- nrow(v) / M_forBest
indicesMatSingleTop <- t(matrix(t(v),M_forBest,nUniqueEnsembles))
repeats <- nUniqueEnsembles

resTopMat <- matrix(0, length(tSeq), repeats)

for(r in 1:repeats)
{
    i <- 1
    for(t in tSeq)
    {
        ind <- indicesMatSingleTop[r,]

        resTopMat[i,r] <- mean(sort(ret$vals[1, t, ind], decreasing=TRUE)[1:nTop])
        
        i <- i + 1
    }
}

resultsSingleTop <- rep(0, length(tSeq))
resultsSingleTopSd <- rep(0, length(tSeq))

for(i in 1:length(tSeq))
{
    resultsSingleTop[i] <- mean(resTopMat[i,])
    resultsSingleTopSd[i] <- sd(resTopMat[i,])
}

piTrain <- "S"
piBench <- "S*"
M <- 1

str <- paste(str, paste("TD & ", "$", M, "$ & ", piTrain, " & ", piBench, 
" & $", round(resultsSingleTop[1],digits=nDigits), " \\pm ", round(resultsSingleTopSd[1],digits=nDigits), "$",
" & $", round(resultsSingleTop[2],digits=nDigits), " \\pm ", round(resultsSingleTopSd[2],digits=nDigits), "$",
" & $", round(resultsSingleTop[3],digits=nDigits), " \\pm ", round(resultsSingleTopSd[3],digits=nDigits), "$",
" & $", round(resultsSingleTop[4],digits=nDigits), " \\pm ", round(resultsSingleTopSd[4],digits=nDigits), "$",
"\\\\ \n", sep=""))    

cat("Latex table entries:\n", str,"\n", sep="")

v2 <-  ret$meanVals[c(1:3,10:13),seq(from=off, to=len, by=experimentBy)]
ylim2 <- c(min(v2),max(v2))

setEPS()

ylabels <- seq(from=min(ylim2),to=max(ylim2),by=0.1)

s <- seq(off,len,10)

if(isTetris)
{
    yLabelSeq <- seq(round(max(v2),digits=0), round(min(v2),digits=0), -10)
} else
{
    nVals <- 6
    yStep <- round((max(v2) - min(v2)) / (nVals - 1),digits=2)        
    yMax <- round(max(v2),digits=1)
    yMin <- yMax - (nVals - 1) * yStep        
    yLabelSeq <- seq(yMax, yMin, -yStep)
}


# get RL method and exploration method from working directory

rlMethod <- toupper(strsplit(basename(getwd()),"-")[[1]][3])

if(isTetris)
{
    explorationMethod = ""
} else
{
    explorationMethod <- strsplit(basename(getwd()),"-")[[1]][4]

    if(explorationMethod != "softmax")
        explorationMethod <- ", epsilon-greedy"
    else
        explorationMethod <- ", softmax"
}

if(rlMethod == "TDLAMBDA")
{
    myMainLab <- bquote(bold(.(plotMainLab)*',' ~ TD(lambda)* .(explorationMethod)))
} else
{
    myMainLab <- bquote(bold(.(plotMainLab)*',' ~ .(rlMethod)* .(explorationMethod)))
}
    
postscript(paste(plotFile, "1.eps", sep=""))
plot(s, ret$meanVals[1,s], type='o', lwd=1.5, lty = "solid", col='grey', xlab=plotXLab, ylab=plotYLab, main=myMainLab, ylim=ylim2, xaxt="n", pch=21, cex.lab=1.7, cex.axis=1.7, cex.main=1.7, cex.sub=1.5,cex=2.0, yaxt="n")
lines(s, ret$meanVals[2,s], type='o', lwd=1.5, lty = "dotted", col='black', pch=24,cex=2.0)
lines(s, ret$meanVals[10,s], type='o', lwd=1.5, lty = "longdash", col='grey', pch=25,cex=2.0)
lines(s, ret$meanVals[12,s], type='o', lwd=1, lty = "solid", col='black', pch=23,cex=2.0)
axis(1, at=plotXAxisAt, labels=plotXAxisLabels, cex.axis=1.7)
axis(2, at=yLabelSeq, labels=yLabelSeq, cex.axis=1.6)
legend("bottomright", lwd=c(1.5,1.5,1.5,1), pch=c(21,24,25,23), col=c('grey', 'black', 'grey', 'black'), lty=c("solid","dotted","longdash","solid"), legend=c("single agent", sprintf("%i agents, Average", nAgents), sprintf("%i agents, Average", nAgents2), sprintf("%i agents, Average", nAgents3)), cex=1.7)
dev.off()

postscript(paste(plotFile, "2.eps", sep=""))
plot(s, ret$meanVals[1,s], type='o', lwd=1.5, lty = "solid", col='grey', xlab=plotXLab, ylab=plotYLab, main=myMainLab, ylim=ylim2, xaxt="n", pch=21, cex.lab=1.7, cex.axis=1.7, cex.main=1.7, cex.sub=1.5,cex=2.0, yaxt="n")
lines(s, ret$meanVals[3,s], type='o', lwd=1.5, lty = "dotted", col='black', pch=24,cex=2.0)
lines(s, ret$meanVals[11,s], type='o', lwd=1.5, lty = "longdash", col='grey', pch=25,cex=2.0)
lines(s, ret$meanVals[13,s], type='o', lwd=1, lty = "solid", col='black', pch=23,cex=2.0)
axis(1, at=plotXAxisAt, labels=plotXAxisLabels, cex.axis=1.7)
axis(2, at=yLabelSeq, labels=yLabelSeq, cex.axis=1.6)
legend("bottomright", lwd=c(1.5,1.5,1.5,1), pch=c(21,24,25,23), col=c('grey', 'black', 'grey', 'black'), lty=c("solid","dotted","longdash","solid"), legend=c("single agent", sprintf("%i agents, Majority Voting", nAgents), sprintf("%i agents, Majority Voting", nAgents2), sprintf("%i agents, Majority Voting", nAgents3)), cex=1.7)
dev.off()

# Write best results (will be used for getSelectiveEnsembleResults.R)

valsAverage20 <- rbind(ret$totalReward[,16], ret$totalRewardSd[,16], rep(nTestrunsEnsemble, length(ret$totalReward[,16])))
valsVoting20 <- rbind(ret$totalReward[,17], ret$totalRewardSd[,17], rep(nTestrunsEnsemble, length(ret$totalReward[,17])))

valsAverage30 <- rbind(ret$totalReward[,18], ret$totalRewardSd[,18], rep(nTestrunsEnsemble, length(ret$totalReward[,18])))
valsVoting30 <- rbind(ret$totalReward[,19], ret$totalRewardSd[,19], rep(nTestrunsEnsemble, length(ret$totalReward[,19])))

valsAverage50 <- rbind(ret$totalReward[,20], ret$totalRewardSd[,20], rep(nTestrunsEnsemble, length(ret$totalReward[,20])))
valsVoting50 <- rbind(ret$totalReward[,21], ret$totalRewardSd[,21], rep(nTestrunsEnsemble, length(ret$totalReward[,21])))

write.table(valsAverage20, "bestValsAverage20")
write.table(valsVoting20, "bestValsVoting20")

write.table(valsAverage30, "bestValsAverage30")
write.table(valsVoting30, "bestValsVoting30")

write.table(valsAverage50, "bestValsAverage50")
write.table(valsVoting50, "bestValsVoting50")

