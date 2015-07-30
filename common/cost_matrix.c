#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

// R CMD SHLIB cost_matrix.c

/* calculates the cost matrix */

void cost_matrix(double *Dmat, double *error, double *vals, int *actionindices, double *state_weights, int *actions, int *actionsMax, int *M, int *N, int *weighted_mean, int *mode, double *theta, double *weight_best_action)
{
    int u;
        
    for(u = 0; u < *M; u++)
    {
        int v;
        for(v = 0; v < *M; v++)
        {
            Dmat[v + u * *M] = 0;
        }
    }
    
    int i;
    int lastInd = 0;
                
    for(i = 0; i < *N; i++)
    {
        int nActions = actions[i];
     
        if(*actionsMax > 0)
        {
            if(nActions > *actionsMax)
                nActions = *actionsMax;
        }
        
        int j;
        
        double weights[*M][nActions];

        for(u = 0; u < *M; u++)
        {
            double sum1 = 0;
            
            if(*mode == 0)
            {
                // Equally weighted
                
                for(j = 0; j < nActions; j++)
                {
                    weights[u][j] = 1.0 / (double) nActions;
                }
            }
            else if(*mode == 1)
            {
                /// Best action strongest weighted, rest of actions equally weighted
                if(nActions == 1)
                {
                    weights[u][0] = 1.0;
                }
                else
                {
                    double bestVal2 = -999999;
                    int bestJ2 = 0;
                    for(j = 0; j < nActions; j++)
                    {
                        if(vals[u + *M * (j + lastInd)] > bestVal2)
                        {
                            bestVal2 = vals[u + *M * (j + lastInd)];
                            bestJ2 = j;
                        }
                    }
                    
                    for(j = 0; j < nActions; j++)
                    {
                        if(j == bestJ2)
                            weights[u][j] = *weight_best_action;
                        else
                            weights[u][j] = 1.0 / ((double) nActions - 1.0) * (1.0 - *weight_best_action);
                    }
                }                    
            }
            else if(*mode == 2)
            {
                // Ranked weighted (within each agent)
                                
                int rank[*M][nActions];
                int rankNew[*M][nActions];
                
                for(j = 0; j < nActions; j++)
                {
                    rank[u][j] = j;
                }
                
                /* bubble sort the array, descending */
                int swapped;
                do
                {
                    swapped = 0;
                    for(j = 1; j < nActions; j++)
                    {
                        if (vals[u + *M * (rank[u][j] + lastInd)] < vals[u + *M * (rank[u][j - 1] + lastInd)])
                        {
                            int tmp = rank[u][j];
                            rank[u][j] = rank[u][j - 1];
                            rank[u][j - 1] = tmp;
                            swapped = 1;
                        }
                    }
                } while(swapped);
                
                for(j = 0; j < nActions; j++)
                {
                    // rank1[0] = index with lowest value,
                    // rank1[nActions - 1] = index with highest value
                    rankNew[u][rank[u][j]] = j + 1;
                }
                
                if(*theta < 0)
                {
                    // Linear
                    for(j = 0; j < nActions; j++)
                    {
                        rank[u][j] = rankNew[u][j];

                        sum1 += rank[u][j];
                    }                       
                }
                else
                {
                    // Exp
                    
                    for(j = 0; j < nActions; j++)
                    {
                        rank[u][j] = exp(rankNew[u][j] / (double) nActions * *theta);

                        sum1 += rank[u][j];
                    }
                }
                
                for(j = 0; j < nActions; j++)
                {
                    weights[u][j] = rank[u][j] / sum1;
                }
            }

            // Do some weight verifications
            sum1 = 0;
            for(j = 0; j < nActions; j++)
            {
                sum1 += weights[u][j];
                
                if(weights[u][j] < 0)
                    printf("### Wrong weights #3 (%lf)\n", weights[u][j]);
            }
            
            if((sum1 < 0.99999) || (sum1 > 1.00001))
                printf("### Wrong weights #1 (%lf)\n", sum1);
            
        } /* for u */
        
        for(u = 0; u < *M; u++)
        {
            int v;
            for(v = 0; v < *M; v++)
            {
                if(*weighted_mean)
                {
                    // Weighted Mean
                    
                    /* In case of Voting (mode = 1, weight_best_action = 1):
                     * The totalcosts will be minimized for selecting agents
                     * 1. which have the lowest errors (= high consistency) with their decisions */

                    double mean1 = 0, mean2 = 0;
                    for(j = 0; j < nActions; j++)
                    {
                        mean1 += weights[v][j] * error[v + *M * (j + lastInd)];
                        mean2 += weights[u][j] * error[u + *M * (j + lastInd)];                    
                    }
                    
                    Dmat[v + u * *M] += state_weights[i] * mean1 * mean2;
                }
                else
                {
                    // Weighted

                    /* In case of Voting (mode = 1, weight_best_action = 1):
                     * The totalcosts will be minimized for selecting agents
                     * 1. which do different decisions
                     * 2. which have the lowest errors (= high consistency) with the same decisions */
                    
                    double sum = 0;
                    for(j = 0; j < nActions; j++)
                    {
                        int k = actionindices[u + *M * (j + lastInd)];
                        
                        int found = 0, j2;
                        for(j2 = 0; j2 < nActions; j2++)
                        {
                            int k2 = actionindices[v + *M * (j2 + lastInd)];
                            if(k2 == k)
                            {
                                found = 1;
                                break;
                            }
                        }
                        
                        if(found)
                            sum += weights[v][j2] * weights[u][j] * error[v + *M * (j2 + lastInd)] * error[u + *M * (j + lastInd)];
                    }
                    
                    Dmat[v + u * *M] += state_weights[i] * sum;
                }
                                
            } /* for v */
        } /* for u */
        
        lastInd += actions[i];
    } /* for i */
}
