testruns <- 50
Mseq <- c(3,5,10,20,30,50)
nAgents <- 100

j <- 1
while(1)
{
    if(j > length(Mseq))
        break

    M <- Mseq[j]

    ensembleIndices <- matrix(0, testruns, M)

    e <- 1
    while(1)
    {
        if(e > testruns)
            break

        ind <- sample(1:nAgents, M)

        foundDouble <- 0
        for(e2 in 1:e)
        {
            if(sum(sort(ensembleIndices[e2,]) == sort(ind)) >= M)
                foundDouble <- 1
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
            ensembleIndices[e,] <- sort(ind)
            e <- e + 1
        }
    }

    filename <- paste("ensembleIndices2_", M, sep="")
#    filename <- paste("ensembleIndicesPartialParallel_", M, sep="")
    write.table(as.vector(t(ensembleIndices)), filename, col.names=FALSE,row.names=FALSE)
    j <- j + 1
}
