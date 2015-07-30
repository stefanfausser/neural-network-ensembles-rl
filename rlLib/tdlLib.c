// The MIT License (MIT)
// 
// Copyright (c) 2010 - 2015 Stefan Faußer
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * \file tdlLib.c
 * \brief Reinforcement learning library
 *
 * \author Stefan Faußer
 *
 * Modification history:
 * 
 * 2010-07-01, S. Fausser - written
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* overview of the RL methods with function approximation:
 * diff = (V_theta1(y_i) - (reward + gamma * V_theta1(y'_i)))
 * w_prime = all second order weights
 * summed_W_Gradient = sum of all (gradient_y_i multiplied by w_prime)
 * h = (diff - summed_W_Gradient) * gradient2_y_i * w_prime
 *
 * TD: Delta theta2 = diff * gradient_y_i
 * RG: Delta theta2 = diff * (gradient_y_i - gamma * gradient_y'_i)
 * TDC: Delta theta2 = diff * gradient_y_i - gamma * gradient_y'_i * summed_W_Gradient - h
 * GTD: Delta theta2 = (gradient_y_i - gamma * gradient_y'_i) * summed_W_Gradient - h
 *
 */
 
#include "tdlLib.h"

#ifndef INFINITY
#define INFINITY        9999999
#endif

typedef struct
{
    int I;
    int s;
    double Vs;
    double Vsreal;
} tPossiblePosition;

static bool netInitialized[MAX_MLPS];
static bool tdlInitialized = false;

static struct train_data testD;
static signed long micro_max_test = -1;

static bool lInternalTestDataBusy = false;
static void *lInternalTestDataHandle = NULL;

typedef struct
{
    struct train_data trainD;
    int mlpfd;
    unsigned long micro;
    unsigned long micro_max;
    unsigned long m;
    double alpha;
    double beta;
    double gamma_internal;
    double lambda;
    unsigned long mseIterations;
    unsigned long nLearnedStates;
    double summedMse;
    double *sampleAlphaDiscount;
    double gradientPrimeFactor;
    eStateValueFuncApprox stateValueFuncApprox;
    double *pWeights;
    int trainingMode;
    double *pV;
    unsigned long nStates;
    bool normalizeLearningRate;
    bool tdcOwnSummedGradient;
    bool replacingTraces;
} tNet;

static tNet lNet[MAX_MLPS];

typedef struct
{
    int nAgents;
    int nAgentsLearningFromAverageSV;
    bool weightedAverage;
    double weightCurrentAgent;
    bool haveDecisionWeights;
    double decisionWeight;
    double decisionNoise;
    double stateValueImprecision;
} tNetAll;

static tNetAll lNetAll;

static double V(int n,
                struct train_data *pData,
                int micro,
                bool usePrimeX,
                double *pVs)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        return -4;
    
    if(pVs == NULL)
        return -5;
    
    double val = 0;

    int i;

    if(usePrimeX && !pData->hasXPrime[micro])
    {
        *pVs = 0;
        return 0;
    }
    
    for(i = 0; i < lNet[n].m; i++)
    {
        if(usePrimeX)
            val += lNet[n].pWeights[i] * pData->x_prime[micro][i];
        else
            val += lNet[n].pWeights[i] * pData->x[micro][i];
    }

    *pVs = val;

    return 0;
}

static int tdlOutput(int n,
                     struct train_data *pData,
                     unsigned long micro_max)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;
    
    if(pData == NULL)
        return -4;
    
    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
    {
        int ret = mlpOutput (lNet[n].mlpfd, micro_max, 0, pData);
        if (ret)
        {
            printf ("### tdlOutput: mlpOutput function returned error (%i).\n",ret);
            return -5;
        }
    }
    else if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear)
    {
        unsigned long micro;
        for(micro = 0; micro < micro_max; micro++)
        {
            double Vs = 0;
            int ret = V(n, pData, micro, false, &Vs);
            if(ret)
            {
                printf("### tdlOutput: V function returned error (%i)\n", ret);
                return -6;
            }
                
            pData->y[micro][0] = Vs;
        }
    }
    else
    {
        // eStateValueFuncApprox_none
        
        unsigned long micro;
        for(micro = 0; micro < micro_max; micro++)
        {
            long s = (long) pData->x[micro][0];
            if(s < 0 || s > lNet[n].nStates)
            {
                printf("### tdlOutput: s (%li) < 0 || s > nStates (%li)\n", s, lNet[n].nStates);
                return -7;
            }
            
            pData->y[micro][0] = lNet[n].pV[s];
        }        
    }

    return 0;
}

static int TD_RG(
    int n,
    bool doTdUpdate,
    struct train_data *pData,
    int micro,
    double **pDeltaWeight)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(pDeltaWeight == NULL)
        return -4;
    
    if(*pDeltaWeight == NULL)
        return -4;

    double Vs_from = 0;
    int ret = V(n, pData, micro, false, &Vs_from);
    if(ret)
    {
        printf("### TD: V function returned error (%i)\n", ret);
        return -5;
    }

    double normFactor = 0;
    if(lNetAll.nAgentsLearningFromAverageSV <= 1)
        normFactor = 1.0;
    else
    {
        if(lNetAll.weightedAverage)
            normFactor = lNetAll.weightCurrentAgent;
        else
            normFactor = 1.0 / lNetAll.nAgentsLearningFromAverageSV;            
    }            
                    
    double r = pData->reward[micro];
    
    int i;
    double delta = 0;
    double gamma = lNet[n].gamma_internal;
    double Vs_to2 = pData->y2[micro][0];
    if(doTdUpdate)
        delta = pData->y[micro][0] + Vs_to2 - Vs_from;
    else
    {
        double Vs_to = 0;
        ret = V(n, pData, micro, true, &Vs_to);
        if(ret)
        {
            printf("### TD: V function returned error (%i)\n", ret);
            return -6;
        }

        // RG update
        if(pData->hasXPrime[micro])
        {
            delta = r + gamma * normFactor * Vs_to + Vs_to2 - Vs_from;                
        }
        else
            delta = r - Vs_from;
    }
    
    for(i = 0; i < lNet[n].m; i++)
    {
        if(doTdUpdate)
            (*pDeltaWeight)[i] = delta * pData->x[micro][i];
        else
        {
            // RG update
            if(pData->hasXPrime[micro])
                (*pDeltaWeight)[i] = delta * (pData->x[micro][i] - lNet[n].gradientPrimeFactor * gamma * pData->x_prime[micro][i]);
            else
                (*pDeltaWeight)[i] = delta * pData->x[micro][i];
        }
    }
    
    return 0;
}

static int tdlTrainWeights(int n,
                           struct train_data *pData,
                           int micro_max,
                           double alpha,
                           double beta,
                           bool updateFirstLayer,
                           bool updateSecondLayer,
                           double *pMse)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;
    
    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
    {
        double normFactor = 0;
        if(lNetAll.nAgentsLearningFromAverageSV <= 1)
            normFactor = 1.0;
        else
        {
            if(lNetAll.weightedAverage)
                normFactor = lNetAll.weightCurrentAgent;        
            else
                normFactor = 1.0 / lNetAll.nAgentsLearningFromAverageSV;            
        }            
        
        double mse = mlp (lNet[n].mlpfd, micro_max, pData, alpha, beta, lNet[n].gamma_internal, lNet[n].lambda, updateFirstLayer, updateSecondLayer, lNet[n].sampleAlphaDiscount, lNet[n].gradientPrimeFactor, normFactor, false, lNet[n].tdcOwnSummedGradient, lNet[n].replacingTraces);

        if (mse < 0)
        {
            printf ("### tdlTrainWeights: mlp function returned error (%i).\n", (int) mse);
            return -4;
        }
        
        if(pMse != NULL)
            *pMse = mse;
    }
    else if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear)
    {
        bool doTdUpdate = false;
        if(lNet[n].trainingMode == TD_MODE)
            doTdUpdate = true;
        
        int micro;
        int i;
        double deltaWeight[lNet[n].m];
        for(i = 0; i < lNet[n].m; i++)
            deltaWeight[i] = 0;
        
        double *pDeltaWeightTmp = (double *) malloc (sizeof(double) * lNet[n].m);
        for(micro = 0; micro < micro_max; micro++)
        {
            int ret = TD_RG(n, doTdUpdate, pData, micro, &pDeltaWeightTmp);
            if(ret)
            {
                printf("### tdlTrainWeights: TD function returned error\n");
                return -5;
            }

            for(i = 0; i < lNet[n].m; i++)
                deltaWeight[i] += pDeltaWeightTmp[i];
        }
        free(pDeltaWeightTmp);
        
        for(i = 0; i < lNet[n].m; i++)
        {
            lNet[n].pWeights[i] += alpha * deltaWeight[i] / micro_max;
        }
    }
    else
    {
        // eStateValueFuncApprox_none
        
        int micro;
        double mse = 0;
        for(micro = 0; micro < micro_max; micro++)
        {
            long s_from = (long) pData->x[micro][0];
            if(s_from < 0 || s_from > lNet[n].nStates)
            {
                printf("### tdlOutput: s_from (%li) < 0 || s_from > nStates (%li)\n", s_from, lNet[n].nStates);
                return -7;
            }
            
            long s_to = 0;
            double V_to = 0;

            if(pData->hasXPrime[micro])
            {
                s_to = (long) pData->x_prime[micro][0];
                if(s_to < 0 || s_to > lNet[n].nStates)
                {
                    printf("### tdlOutput: s_to (%li) < 0 || s_to > nStates (%li)\n", s_to, lNet[n].nStates);
                    return -7;
                }
                V_to = lNet[n].pV[s_to];
            }
            
            double V_from = lNet[n].pV[s_from];
            
            double r = pData->reward[micro];
            double gamma = lNet[n].gamma_internal;

            double delta = r + gamma * V_to - V_from;
            mse += pow(delta, 2.0);

            lNet[n].pV[s_from] += alpha * delta;
        } /* for micro */

        if(pMse != NULL)
            *pMse = mse;
    }

    return 0;
}

int tdlCleanup(int n)
{
    if(!tdlInitialized)
        return -1;
    
    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;
    
    if(!netInitialized[n])
        return -3;
    
    if (freeTrainData (&lNet[n].trainD))
        printf ("### tdlCleanup: unable to free mlp data.\n");

    free(lNet[n].sampleAlphaDiscount);
    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear)
        free(lNet[n].pWeights);
    else if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_none)
        free(lNet[n].pV);
    else
        cleanupMlpNet (lNet[n].mlpfd);

    netInitialized[n] = false;

    bool netLeft = false;
    int i;
    for(i = 0; i < MAX_MLPS; i++)
    {
        if(netInitialized[i])
        {
            netLeft = true;
            break;
        }
    }
    
    if(!netLeft)
    {
        printf("tdlCleanup: Last net cleaned up, cleanup up the library completely\n");
        if (freeTrainData (&testD))
            printf ("### tdlCleanup: unable to free mlp data.\n");
        if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
            mlpLibDeinit ();
        tdlInitialized = false;
    }

    return 0;
}

int tdlInit(int n,
            int nAgentsLearningFromAverageSV,
            int nAgents,
            int m_request,
            int micro_max_request,
            int micro_max_test_request,
            char *configfile,
            char *netfile,
            unsigned int seed,
            eStateValueFuncApprox stateValueFuncApprox,
            bool needs_x_prime,
            double gradientPrimeFactor,
            double weightInterval,
            int trainingMode,
            double minStateValue,
            double maxStateValue,
            bool weightedAverage,
            double weightCurrentAgent,
            bool haveDecisionWeights,
            double decisionWeight,
            double decisionNoise,
            double stateValueImprecision,
            bool tdcOwnSummedGradient,
            bool replacingTraces
           )
{
    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -1;
        
    if(netInitialized[n])
        return -2;
    
    struct mlp_init_values mlpInitVals;
    double a;
    double aOut = 0;
    struct mlp_param mlpP;
    int weightNormalizedInitialization = 0;
    int thresholdZeroInitialization = 0;
    
    
    if(stateValueFuncApprox == eStateValueFuncApprox_none)
    {
        // Force the number of neurons to be 1 (single dimension state coding = state tables)
        m_request = 1;
    }
    
    /* Start of important Parameters */
    mlpInitVals.m = m_request;
    mlpInitVals.n = 1;
    if(stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        a = 2.0 / mlpInitVals.m;
    else
    {
        a = weightInterval;
        mlpInitVals.nrHiddenLayers = 0;
        mlpP.trainingMode = trainingMode;
    }
    /* End of important parameters */
    
    if(stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
    {
        if (openMlpParameterFile (configfile, &mlpInitVals, &mlpP, &seed, &a, &aOut, &weightNormalizedInitialization, &thresholdZeroInitialization))
        {
            printf ("### tdlInit: could not load mlp parameters from file %s.\n",configfile);
            return -1;
        }
        else
        {
            if(!n)        
                printf ("tdlInit: successfully loaded mlp parameters from file.\n");
        }
        
        // Overwrite input + output dimension
        mlpInitVals.m = m_request;
        mlpInitVals.n = 1;
    }
    
    if(!n)        
    {
        printf("tdlInit: nAgentsLearningFromAverageSV (%i)\n", nAgentsLearningFromAverageSV);
    
        printf ("tdlInit: m = %i, n = %i\n", mlpInitVals.m, mlpInitVals.n);
    }

    if (allocateTrainData (&lNet[n].trainD, micro_max_request, mlpInitVals.m, mlpInitVals.n, needs_x_prime))
    {
        printf ("### tdlInit: unable to allocate mlp data.\n");
        return -2;
    }

    if(!tdlInitialized)
    {
        if (allocateTrainData (&testD, micro_max_test_request, mlpInitVals.m, mlpInitVals.n, false))
        {
            printf ("### tdlInit: unable to allocate mlp data.\n");
            return -3;
        }
        micro_max_test = micro_max_test_request;
        if(stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
            mlpLibInit (MAX_MLPS);
        int i;
        for(i = 0; i < MAX_MLPS; i++)
            netInitialized[i] = false;
    }

    unsigned long m_max;
    if(micro_max_request > micro_max_test_request)
        m_max = micro_max_request;
    else
        m_max = micro_max_test_request;
    
    if(stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
    {
        if(netfile != NULL)
            lNet[n].mlpfd = mlpRestore (&mlpP, netfile, micro_max_request, seed);
        else
            lNet[n].mlpfd = -1;
        
        if (lNet[n].mlpfd >= 0)
        {
            printf ("tdlInit: old mlp state for net %i restored.\n", n);
            printf ("Starting statistics for restored neural network...\n");
            outputWeightsStatistics (lNet[n].mlpfd);
            
            getMlpInitValues(lNet[n].mlpfd, &mlpInitVals);
        }
        else
        {
            lNet[n].mlpfd = initializeMlpNet (&mlpInitVals, &mlpP, a, aOut, weightNormalizedInitialization, thresholdZeroInitialization, m_max, seed);

            if (lNet[n].mlpfd < 0)
            {
                printf ("### tdlInit: Unable to initialize mlp weights (%i).\n", lNet[n].mlpfd);
                return -4;
            }
        }
    }

    lNet[n].micro_max = micro_max_request;
    lNet[n].m = m_request;
    lNet[n].micro = 0;
    lNet[n].alpha = 0.01;
    lNet[n].tdcOwnSummedGradient = tdcOwnSummedGradient;
    lNet[n].replacingTraces = replacingTraces;
    lNet[n].beta = 0.01;
    lNet[n].gamma_internal = 0.9;
    lNet[n].lambda = 0;
    lNet[n].mseIterations = 1000;
    lNet[n].nLearnedStates = 0;
    lNet[n].summedMse = 0;
    lNet[n].sampleAlphaDiscount = (double *) malloc(micro_max_request * sizeof(double));
    lNet[n].gradientPrimeFactor = gradientPrimeFactor;
    lNet[n].stateValueFuncApprox = stateValueFuncApprox;
    lNetAll.nAgents = nAgents;
    lNetAll.nAgentsLearningFromAverageSV = nAgentsLearningFromAverageSV;
    lNetAll.weightedAverage = weightedAverage;
    lNetAll.weightCurrentAgent = weightCurrentAgent;
    lNetAll.haveDecisionWeights = haveDecisionWeights;
    lNetAll.decisionNoise = decisionNoise;
    lNetAll.stateValueImprecision = stateValueImprecision;
    if(haveDecisionWeights)
        lNetAll.decisionWeight = decisionWeight;
    
    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear)
    {
        lNet[n].trainingMode = mlpP.trainingMode;
        lNet[n].pWeights = (double *) malloc(lNet[n].m * sizeof(double));
        int i;
#if 0
const double biFeatures[][22] = { 
    {-3.9449, -9.3878, -3.2907, -9.4902, -15.1198, -2.5428, -7.2624, -3.6481, -24.8931, 8.2739, -23.8090, 3.9224, -21.8455, 2.5704, -25.9445, 5.4411, -10.6643, 1.5848, -27.1720, 3.6956, -68.3404, -1.1340},
    {11.0572, -31.3095, -1.94, -11.3902, -7.4452, -11.4852, -9.1258, 0.4228, -30.0521, 11.1677, -8.4923, 5.9814, -14.7293, 5.8015, -45.5280, 6.1028, -19.9065, 5.1730, -5.3666, 12.5566, -82.6131, -16.3142},
};
#endif
        for(i = 0; i < m_request; i++)
        {
//            lNet[n].pWeights[i] = biFeatures[0][i];
            lNet[n].pWeights[i] = randValDouble(-a,a);
//            lNet[n].pWeights[i] = randValDouble(0,a);
        }
    }
    else if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_none)
    {
        lNet[n].trainingMode = trainingMode;
        lNet[n].pV = (double *) malloc(m_max * sizeof(double));
        int i;
        for(i = 0; i < m_max; i++)
            lNet[n].pV[i] = randValDouble(minStateValue, maxStateValue);
        lNet[n].nStates = m_max;        
    }

    netInitialized[n] = true;
    tdlInitialized = true;
    
    if(!n)        
    {
        printf ("m = %i\n", m_request);
        if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        {
            printf ("h0 = %i\n", mlpInitVals.h[0]);
            printf ("nrHiddenLayers = %i\n", mlpInitVals.nrHiddenLayers);
            if (mlpInitVals.nrHiddenLayers > 1)
                printf ("h1 = %i\n", mlpInitVals.h[1]);
        }
        
        if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear || lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        {
            if (mlpP.trainingMode == TD_MODE)
                printf ("trainingMode = TD Learning\n");
            else if (mlpP.trainingMode == RG_MODE)
                printf ("trainingMode = RG Learning\n");
            else if (mlpP.trainingMode == TDC_MODE)
                printf ("trainingMode = TDC Learning\n");
            else
                printf ("trainingMode = UNKNOWN ???\n");
        }

        if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        {
            printf("Nonlinear function approximation by a MLP\n");
            if (mlpInitVals.transFktTypeHidden == 0)
                printf ("transFktType hidden layer = Logistic function\n");
            else if (mlpInitVals.transFktTypeHidden == 1)
            {
                printf ("transFktType hidden layer = Fermi function\n");
                printf ("beta = %f\n", mlpP.beta);
            }
            else if (mlpInitVals.transFktTypeHidden == 2)
                printf ("transFktType hidden layer = Tangens hyperbolicus\n");
            else
                printf ("transFktType hidden layer = Unknown ? Plz check error backpropagation library.\n");
            if (mlpInitVals.transFktTypeOutput == 0)
                printf ("transFktType output layer = Logistic function\n");
            else if (mlpInitVals.transFktTypeOutput == 1)
            {
                printf ("transFktType output layer = Fermi function\n");
                printf ("beta = %f\n", mlpP.beta);
            }
            else if (mlpInitVals.transFktTypeOutput == 2)
                printf ("transFktType output layer = Tangens hyperbolicus\n");
            else if (mlpInitVals.transFktTypeOutput == 3)
                printf ("transFktType output layer = Linear function\n");
            else
                printf ("transFktType output layer = Unknown ? Plz check error backpropagation library.\n");

            if(mlpInitVals.hasThresholdOutput)
                printf("Output layer has threshold\n");
            else
                printf("Output layer has no threshold\n");
            
            printf ("maxIterations = %i\n", mlpP.maxIterations);
            printf ("epsilon = %f\n", mlpP.epsilon);
        }
        else if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_linear)
        {
            printf("Linear function approximation\n");
        }
        else
        {
            // eStateValueFuncApprox_none
            printf("No function approximation of the state-values\n");
        }

        printf ("seed = %i\n", seed);
        printf ("a = %f\n", a);
        printf ("aOut = %f\n", aOut);
        printf ("weightNormalizedInitialization = %i\n", weightNormalizedInitialization);
        
        if(thresholdZeroInitialization)
            printf("Initialized zero thresholds\n");
        else
            printf("Initialized random thresholds\n");
        
        printf("tdlInit: can save maximum %li states\n",lNet[n].micro_max);
    }
    
    return 0;
}

int tdlGetParam(
    int n,
    bool *pNormalizeLearningRate,
    double *pAlpha,
    double *pBeta,
    double *pGamma,
    double *pLambda)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(pAlpha == NULL ||
       pBeta == NULL ||
       pGamma == NULL)
        return -4;

    *pNormalizeLearningRate = lNet[n].normalizeLearningRate;
    *pAlpha = lNet[n].alpha;
    *pBeta = lNet[n].beta;
    *pGamma = lNet[n].gamma_internal;
    *pLambda = lNet[n].lambda;

    return 0;
}

int tdlSetParam(
    int n,
    bool normalizeLearningRate,
    double alpha_request,
    double beta_request,
    double gamma_request,
    double lambda_request)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;
    
    lNet[n].normalizeLearningRate = normalizeLearningRate;
    lNet[n].alpha = alpha_request;
    lNet[n].beta = beta_request;
    lNet[n].gamma_internal = gamma_request;
    lNet[n].lambda = lambda_request;

    return 0;
}

int tdlSetDecisionWeightCurrentAgent(
    double weightCurrentAgent)
{
    if(!tdlInitialized)
        return -1;

    lNetAll.weightCurrentAgent = weightCurrentAgent;
    
    return 0;
}

int tdlSetMseIterations(
    int n,
    unsigned long mseIterations_request)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    lNet[n].mseIterations = mseIterations_request;

    return 0;
}

int tdlAddStateDone(
    int n,
    bool autoLearning)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;
        
    if(lNet[n].micro >= lNet[n].micro_max)
    {
        if(autoLearning)
        {
            lNet[n].nLearnedStates += lNet[n].micro;

            double mse = 0;
            if(tdlLearn(n, &mse))
            {
                printf("### tdlAddState: tdlLearn function returned error.\n");
                return -2;
            }

            lNet[n].summedMse += mse;

            if(lNet[n].nLearnedStates >= lNet[n].mseIterations)
            {
                lNet[n].summedMse /= lNet[n].nLearnedStates;

                printf("tdlAddState: Learned states (%li), mse (%lf)\n",lNet[n].nLearnedStates,lNet[n].summedMse);

                lNet[n].nLearnedStates = 0;
                lNet[n].summedMse = 0;
            }
        }
        else
            return -3;
    }

    return 0;
}

int tdlGetNumberStates(
    int n,
    unsigned long *pNStates)
{
    if(!tdlInitialized)
        return -1;
    
    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(pNStates == NULL)
        return -4;
    
    *pNStates = lNet[n].micro;

    return 0;
}

int tdlUpdateAlphaDiscount(
    int n,
    int micro,
    double alphaDiscount)
{
    if(!tdlInitialized)
        return -1;
    
    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(micro >= lNet[n].micro_max)
        return -4;

    
    lNet[n].sampleAlphaDiscount[micro] = alphaDiscount;

    return 0;
}

int tdlAddState(
        int n,
        tState *pS,
        tState *pS_prime,
        tState *pS_prime_ensemble,
        double reward,
        double alphaDiscount)
{
    if(!tdlInitialized)
        return -1;
    
    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(lNet[n].micro >= lNet[n].micro_max)
        return -4;

    double normFactor1 = 0, normFactor2 = 0;

    if(lNetAll.nAgentsLearningFromAverageSV <= 1)
    {
        normFactor1 = 1.0;
        normFactor2 = 1.0;
    }
    else
    {
        if(lNetAll.weightedAverage)
        {
            normFactor1 = lNetAll.weightCurrentAgent;
            normFactor2 = (1.0 - lNetAll.weightCurrentAgent) * 1.0 / ((double) lNetAll.nAgentsLearningFromAverageSV - 1.0);
        }
        else
        {
            normFactor1 = 1.0 / lNetAll.nAgentsLearningFromAverageSV;
            normFactor2 = 1.0 / lNetAll.nAgentsLearningFromAverageSV;
        }
    }
    
    pS->s = lNet[n].trainD.x[lNet[n].micro];
    if(pS_prime == NULL)
    {
        lNet[n].trainD.hasXPrime[lNet[n].micro] = 0;
        lNet[n].trainD.y[lNet[n].micro][0] = reward;
        lNet[n].trainD.y2[lNet[n].micro][0] = 0;
    }
    else
    {
        lNet[n].trainD.hasXPrime[lNet[n].micro] = 1;
        pS_prime->s = lNet[n].trainD.x_prime[lNet[n].micro];
        lNet[n].trainD.y[lNet[n].micro][0] = reward + normFactor1 * lNet[n].gamma_internal * pS_prime->Vs;
        if(pS_prime_ensemble == NULL)
            lNet[n].trainD.y2[lNet[n].micro][0] = 0;
        else
            lNet[n].trainD.y2[lNet[n].micro][0] = normFactor2 * lNet[n].gamma_internal * pS_prime_ensemble->Vs;
    }

    lNet[n].trainD.reward[lNet[n].micro] = reward;
    lNet[n].trainD.gamma[lNet[n].micro] = lNet[n].gamma_internal;
    lNet[n].trainD.delta[lNet[n].micro] = pow(lNet[n].trainD.y[lNet[n].micro][0] + lNet[n].trainD.y2[lNet[n].micro][0] - pS->Vs, 2.0);

    lNet[n].sampleAlphaDiscount[lNet[n].micro] = alphaDiscount;

    lNet[n].micro++;

    return 0;
}

int tdlClearGradientTrace(
    int n)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    // TODO: Implement for linear FA
    
    if(lNet[n].stateValueFuncApprox == eStateValueFuncApprox_nonlinear_mlp)
        clearGradientTrace(lNet[n].mlpfd);
    else
        return -4;
    
    return 0;
}

int tdlCancelEpisode(
    int n)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;
    
    lNet[n].micro = 0;
    
    return 0;
}

int tdlLearn(
    int n,
    double *pMse)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    unsigned long _micro = lNet[n].micro;
    
    if(!_micro)
        return 0;
    
    struct train_data _trainD;

    _trainD = lNet[n].trainD;
    
    bool updateFirstLayer = true;
    bool updateSecondLayer = true;
    
    double alpha = lNet[n].alpha;
    double beta = lNet[n].beta;
    
    if(lNet[n].normalizeLearningRate)
    {
        alpha /= (double) _micro;
        beta /= (double) _micro;
    }
    
    double mse = 0;
    
    int ret = tdlTrainWeights(n, &_trainD, _micro, alpha, beta, updateFirstLayer, updateSecondLayer, &mse);

    if (ret)
    {
        printf ("### tdlLearn: mlp function returned error.\n");
        return -2;
    }
    
    if(pMse != NULL)
       *pMse = mse; 
    
    lNet[n].micro = 0;

    return 0;
}

int tdlFreeStateValues(
    void *pHandle)
{
    if(!tdlInitialized)
        return -1;

    if(pHandle == NULL)
        return -2;
    
    if(pHandle == &testD)
    {
        if(lInternalTestDataBusy)
            lInternalTestDataBusy = false;
        else
            return -3;
    }
    else
    {
        if (freeTrainData (pHandle))
        {
            free(pHandle);
            
            return -4;
        }

        free(pHandle);
    }
    
    return 0;
}

int tdlGetStateValuesPrepare(
    bool useInternalMem,
    void **pHandle,                             
    tState *pS,
    unsigned long micro)
{
    if(!tdlInitialized)
        return -1;

    if(pS == NULL)
        return -4;
    
    if(useInternalMem)
    {
        if(micro > micro_max_test)
            return -5;

        if(lInternalTestDataBusy)
            return -6;
                
        lInternalTestDataBusy = true;
        
        lInternalTestDataHandle = (void *) &testD;
        
        *pHandle = (void *) &testD;

        /* Prepare pointer for state coding s */
        unsigned long i;
        for(i = 0; i < micro; i++)
            pS[i].s = testD.x[i];
    }
    else
    {
        struct train_data *pTestD = (struct train_data *) malloc(sizeof(struct train_data));
        if(pTestD == NULL)
        {
            return -8;
        }        
        
        /* WARNING: Only allocates data for a single output neuron (but this should be always enough)
         * and uses the number of input neurons for the first agent (but other agents should have the same number)*/
        if (allocateTrainData (pTestD, micro, lNet[0].m, 1, false))
        {
            free(pTestD);

            return -7;
        }
        
        *pHandle = pTestD;
        
        /* Prepare pointer for state coding s */
        unsigned long i;
        for(i = 0; i < micro; i++)
            pS[i].s = pTestD->x[i];
    }
    
    return 0;
}

int tdlGetStateValues(
    void *pHandle,
    int n,
    tState *pS,
    unsigned long micro)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(pS == NULL)
        return -4;

    if(pHandle == &testD)
    {        
        if(micro > micro_max_test)
            return -5;
    }
    else
    {
        // TODO: Verify sizes
    }

    /* Test trained mlp */
    int ret = tdlOutput (n, pHandle, micro);
    if (ret)
    {
        printf ("### tdlGetStateValue: tdlOutput function returned error (%i).\n",ret);
        return -4;
    }

    /* Copy over state values V(s) */
    unsigned long i;
    for(i = 0; i < micro; i++)
        pS[i].Vs = ((struct train_data *) pHandle)->y[i][0];

    return 0;
}

int tdlSaveNet(
    int n,
    char *netfile)
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(mlpSave(lNet[n].mlpfd, netfile))
    {
        printf("### tdlSaveNet: mlpSave function returned error.\n");
        return -4;
    }

    return 0;
}

int tdlGetBestState2(
    tState *pS,
    double *pSTTV,
    unsigned long micro,
    eStateDecision stateDecision,
    double decisionNoise,
    double stateValueImprecision,
    double tau,
    unsigned long *pI,
    double *pVs,
    double *pVsR,
    unsigned long *pNStates,
    double *pMinDiff,
    double *pMaxDiff
    )
{
    tPossiblePosition pos[micro];
    int nPossiblePositions = 0;
    double bestval = -INFINITY;
    unsigned long i;

    static unsigned long calls = 0;
    
    static double averageProbSoftmax = 0;
    static double averageLowestProbSoftmax = 0;
    static double averageHighestProbSoftmax = 0;
    
    calls++;
    
    bool forceBestDecision = false;
    if(pSTTV != NULL)
        forceBestDecision = true;

    // Fill the pos array with the states to choose from
        
    if(stateDecision == eStateDecision_softmax ||
       stateDecision == eStateDecision_linear ||
       stateDecision == eStateDecision_exploitation_partial ||
       stateDecision == eStateDecision_softmax_statistics
    )
    {
        // Add all
        for(i = 0; i < micro; i++)
        {
            double Vcurr, VcurrReal;
            Vcurr = pS[i].Vs;

            VcurrReal = Vcurr;

            Vcurr += randValDouble(-decisionNoise, decisionNoise);

            if(forceBestDecision)
                Vcurr = pSTTV[i];
            
            pos[nPossiblePositions].I = i;
            pos[nPossiblePositions].Vs = Vcurr;
            pos[nPossiblePositions].Vsreal = VcurrReal;
            nPossiblePositions++;
        }        
    }
    else if(stateDecision == eStateDecision_exploitation ||
       stateDecision == eStateDecision_epsilon_greedy)
    {
        // Only add the best
        for(i = 0; i < micro; i++)
        {
            double Vcurr, VcurrReal;
            Vcurr = pS[i].Vs;

            VcurrReal = Vcurr;

            Vcurr += randValDouble(-decisionNoise, decisionNoise);

            if(forceBestDecision)
                Vcurr = pSTTV[i];
            
            if ((Vcurr >= (bestval - stateValueImprecision)) && (Vcurr <= (bestval + stateValueImprecision)))
            {
                pos[nPossiblePositions].I = i;
                pos[nPossiblePositions].Vs = Vcurr;
                pos[nPossiblePositions].Vsreal = VcurrReal;
                nPossiblePositions++;
                
                /* Choose new bestval and possibly remove values from list that no longer
                * fullfil the required criteria */
                int ind;
                bestval = -INFINITY;
                for(ind = 0; ind < nPossiblePositions; ind++)
                {
                    if(pos[ind].Vs > bestval)
                    {
                        bestval = pos[ind].Vs;
                    }
                }

                for(ind = 0; ind < nPossiblePositions; ind++)
                {
                    if ((pos[ind].Vs > (bestval + stateValueImprecision)) || (pos[ind].Vs < (bestval - stateValueImprecision)))
                    {
                        // Remove index ind from pos list
                        pos[ind].I = pos[nPossiblePositions-1].I;
                        pos[ind].Vs = pos[nPossiblePositions-1].Vs;
                        pos[ind].Vsreal = pos[nPossiblePositions-1].Vsreal;
                        nPossiblePositions--;
                        // Forces a validation of the current index (former last index)
                        ind--;
                    }
                }
            }
            else if (Vcurr > bestval)
            {
                pos[0].I = i;
                pos[0].Vs = Vcurr;
                pos[0].Vsreal = VcurrReal;
                nPossiblePositions = 1;
                bestval = Vcurr;
            }
        }        
    }

    unsigned long myrand2 = 0;

    if(stateDecision == eStateDecision_softmax)
    {
        int J;
        double sum = 0;
        for (J = 0; J < nPossiblePositions; J++)
            sum += exp (pos[J].Vs / tau);
        double probsum = 0;
        double randval = randValDouble (0, 1);
        myrand2 = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            probsum += (exp (pos[J].Vs / tau) / sum);
            if (randval <= probsum)
            {
                myrand2 = J;
                break;
            }
        }
        pI[0] = pos[myrand2].I;
        pVs[0] = pos[myrand2].Vs;
        if(pVsR != NULL)
            pVsR[0] = pos[myrand2].Vsreal;
        
        *pNStates = 1;
    }
    else if(stateDecision == eStateDecision_softmax_statistics)
    {
        int J;
        double sum = 0;
        double prob[nPossiblePositions];
                
        for (J = 0; J < nPossiblePositions; J++)
        {
            prob[J] = exp (pos[J].Vs / tau);                        
            sum += prob[J];
        }

        for (J = 0; J < nPossiblePositions; J++)
            prob[J] /= sum;
        
        double avProb = 0;
        double lowestProb = 1.0;
        double highestProb = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            avProb += prob[J] / (double) nPossiblePositions;

            if(prob[J] < lowestProb)
                lowestProb = prob[J];

            if(prob[J] > highestProb)
                highestProb = prob[J];
        }
        
        averageProbSoftmax = ((calls - 1.0) * averageProbSoftmax + avProb) / (double) calls;
        averageLowestProbSoftmax = ((calls - 1.0) * averageLowestProbSoftmax + lowestProb) / (double) calls;
        averageHighestProbSoftmax = ((calls - 1.0) * averageHighestProbSoftmax + highestProb) / (double) calls;
        
        if(calls > 10000)
        {
            printf("softmax statistics: average prob (%lf), average lowest prob (%lf), average highest prob (%lf)\n", averageProbSoftmax, averageLowestProbSoftmax, averageHighestProbSoftmax);
            calls = 0;
        }
        
        double probsum = 0;
        double randval = randValDouble (0, 1);
        myrand2 = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            probsum += prob[J];
            if (randval <= probsum)
            {
                myrand2 = J;
                break;
            }
        }
        pI[0] = pos[myrand2].I;
        pVs[0] = pos[myrand2].Vs;
        if(pVsR != NULL)
            pVsR[0] = pos[myrand2].Vsreal;
        
        *pNStates = 1;        
    }
    else if(stateDecision == eStateDecision_linear)
    {
        // Choose states with probability linear to their state-values
        int J;
        double sum = 0;
        double prob[nPossiblePositions];

        for (J = 0; J < nPossiblePositions; J++)
        {
            prob[J] = pos[J].Vs;
            sum += prob[J];
        }

        for (J = 0; J < nPossiblePositions; J++)
            prob[J] /= sum;
        
        double avProb = 0;
        double lowestProb = 1.0;
        double highestProb = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            avProb += prob[J] / (double) nPossiblePositions;

            if(prob[J] < lowestProb)
                lowestProb = prob[J];

            if(prob[J] > highestProb)
                highestProb = prob[J];
        }
        
        averageProbSoftmax = ((calls - 1.0) * averageProbSoftmax + avProb) / (double) calls;
        averageLowestProbSoftmax = ((calls - 1.0) * averageLowestProbSoftmax + lowestProb) / (double) calls;
        averageHighestProbSoftmax = ((calls - 1.0) * averageHighestProbSoftmax + highestProb) / (double) calls;
        
        if(calls > 10000)
        {
            printf("linear statistics: average prob (%lf), average lowest prob (%lf), average highest prob (%lf)\n", averageProbSoftmax, averageLowestProbSoftmax, averageHighestProbSoftmax);
            calls = 0;
        }
        
        double probsum = 0;
        double randval = randValDouble (0, 1);
        myrand2 = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            probsum += prob[J];
            if (randval <= probsum)
            {
                myrand2 = J;
                break;
            }
        }
        pI[0] = pos[myrand2].I;
        pVs[0] = pos[myrand2].Vs;
        if(pVsR != NULL)
            pVsR[0] = pos[myrand2].Vsreal;
        
        *pNStates = 1;        
    }
    else if(stateDecision == eStateDecision_exploitation_partial)
    {
        // Choose the best state with probability impactOfBestState (0.7) and the rest states with 1.0 - impactOfBestState (0.3) / number of rest states
        double bestval = -INFINITY;
        int bestJ = -1;
        int J;
        for (J = 0; J < nPossiblePositions; J++)
        {
            if(pos[J].Vs > bestval)
            {
                bestval = pos[J].Vs;
                bestJ = J;
            }
            else if(pos[J].Vs == bestval)
            {
                if(randValLong (0, 1))
                {
                    bestJ = J;
                }
            }            
        }

        const double impactOfBestState = 0.7;
        double sum = (nPossiblePositions - 1);
        double probsum = 0;
        double randval = randValDouble (0, 1);
        myrand2 = 0;
        for (J = 0; J < nPossiblePositions; J++)
        {
            if(J != bestJ)
                probsum += (1.0 - impactOfBestState) / sum;
            else
                probsum += impactOfBestState;
            if (randval <= probsum)
            {
                myrand2 = J;
                break;
            }
        }
        pI[0] = pos[myrand2].I;
        pVs[0] = pos[myrand2].Vs;
        if(pVsR != NULL)
            pVsR[0] = pos[myrand2].Vsreal;
        
        *pNStates = 1;
    }
    else if(stateDecision == eStateDecision_epsilon_greedy || 
        stateDecision == eStateDecision_exploitation)
    {
        for(i = 0; i < nPossiblePositions; i++)
        {
            pI[i] = pos[i].I;
            pVs[i] = pos[i].Vs;
            if(pVsR != NULL)
                pVsR[i] = pos[i].Vsreal;
        }
        *pNStates = nPossiblePositions;
    }

    return 0;
}

int tdlGetBestStateEnsemble(
    void *pHandle,
    int n,
    eEnsembleDecision ensembleDecision,
    tState *pS,
    double *pSTTV,
    unsigned long micro,
    eStateDecision stateDecision,
    double tau,
    double epsilon,
    bool *pExploited,
    unsigned long *pBestStatesIndices,
    unsigned long *pBestStatesLen,
    double *pVs,
    double *pVs2,
    double *pVsR,
    double *pMinDiff,
    double *pMaxDiff,
    bool haveCachedVals
    )
{
    if(!tdlInitialized)
        return -1;

    if((n > MAX_MLPS - 1)||
       (n < 0))
        return -2;

    if(!netInitialized[n])
        return -3;

    if(pS == NULL)
        return -4;
    
    if(!micro)
    {
        printf("tdlGetBestStateEnsemble: micro = 0 ???\n");
        return -5;
    }

    // Decide if the following game-state should be explored or exploited
    *pExploited = true;
    if(stateDecision == eStateDecision_exploration)
        *pExploited = false;
    else if(stateDecision == eStateDecision_exploitation)
        *pExploited = true;
    else if(stateDecision == eStateDecision_epsilon_greedy)
    {
        double myrand = randValDouble (0, 1);
        if (myrand >= epsilon)
            *pExploited = false;
        else
            *pExploited = true;
    }

    if(!(*pExploited) && pVs2 == NULL)
    {
        int i;
        // Return all states
        for(i = 0; i < micro; i++)
            pBestStatesIndices[i] = i;

        *pBestStatesLen = micro;

        return 0;
    }

    // Directly choose the best state, an ensemble decision is highly unnecessary
    if(pSTTV != NULL)
        ensembleDecision = eEnsembleDecision_no_ensemble;
        
    tState state[lNetAll.nAgents][micro];

    double sumVs_tmp[micro];
    double *pSumVs = NULL;
    
    if(haveCachedVals && ensembleDecision != eEnsembleDecision_no_ensemble)
    {
        printf("### tdlGetBestStateEnsemble: Cached vals not supported for ensembles\n");
        return -6;
    }
    
    if(!haveCachedVals)
    {
        // Get the state-values of our agent n
        if(tdlGetStateValues(pHandle, n, pS, micro))
        {
            printf("### tdlGetStateValues function returned error.\n");
            return -10;
        }
    }
    
    double valuesThisAgent[micro];

    if((ensembleDecision == eEnsembleDecision_average_state_values_decision) ||
       (ensembleDecision == eEnsembleDecision_average_state_values_decision_weighted)
    )
    {
        unsigned long i;
        for(i = 0; i < micro; i++)
        {
            valuesThisAgent[i] = pS[i].Vs;
            
            if(ensembleDecision == eEnsembleDecision_average_state_values_decision_weighted)
                pS[i].Vs *= lNetAll.weightCurrentAgent;
            else
            {
                if(lNetAll.haveDecisionWeights)
                    pS[i].Vs *= lNetAll.decisionWeight;
                else
                    pS[i].Vs *= (1.0 / lNetAll.nAgents);
            }            
        }
    }
    
    // Now get the state-values of all other agents
    if(ensembleDecision != eEnsembleDecision_no_ensemble)
    {
        pSumVs = sumVs_tmp;

        unsigned long i;
        for(i = 0; i < micro; i++)
            pSumVs[i] = 0;

        int z;
        for(z = 0; z < lNetAll.nAgents; z++)
        {
            if(z == n)
                continue;
            
            if(tdlGetStateValues(pHandle, z, state[z], micro))
            {
                printf("### tdlGetStateValues function returned error.\n");
                return -10;
            }
                            
            for(i = 0; i < micro; i++)
            {
                pSumVs[i] += state[z][i].Vs;
                if(ensembleDecision == eEnsembleDecision_average_state_values_decision)
                {
                    if(lNetAll.haveDecisionWeights)
                        pS[i].Vs += lNetAll.decisionWeight * state[z][i].Vs;
                    else
                        pS[i].Vs += 1.0 / lNetAll.nAgents * state[z][i].Vs;
                }
                else if(ensembleDecision == eEnsembleDecision_average_state_values_decision_weighted)
                    pS[i].Vs += (1.0 - lNetAll.weightCurrentAgent) * 1.0 / ((double) lNetAll.nAgents - 1.0) * state[z][i].Vs;
            }
        }
    }

    if(pMinDiff != NULL && pMaxDiff != NULL)
    {
        double mindiff = INFINITY;
        double maxdiff = 0;
        int i;
        for(i = 0; i < micro; i++)
        {
            int j;
            for(j = 0; j < micro; j++)
            {
                if(i == j)
                    continue;
                
                double val1, val2;
                val1 = pS[i].Vs;
                val2 = pS[j].Vs;

                double diff = fabs(val1 - val2);
                if(diff < mindiff)
                    mindiff = diff;
                if(diff > maxdiff)
                    maxdiff = diff;
            }
        }
    }

    if(!(*pExploited))
    {
        int i;
        // Return all states
        for(i = 0; i < micro; i++)
            pBestStatesIndices[i] = i;

        *pBestStatesLen = micro;

        if(pSumVs != NULL && pVs2 != NULL)
        {
            int i;
            for(i = 0; i < *pBestStatesLen; i++)
                pVs2[i] = pSumVs[pBestStatesIndices[i]];
        }
        
        return 0;
    }
    
    if(ensembleDecision == eEnsembleDecision_no_ensemble ||
       ensembleDecision == eEnsembleDecision_single_agent_decision
    )
    {
        int ret = tdlGetBestState2(pS, pSTTV, micro, stateDecision, lNetAll.decisionNoise, lNetAll.stateValueImprecision, tau, pBestStatesIndices, pVs, pVsR, pBestStatesLen, pMinDiff, pMaxDiff);

        if(pSumVs != NULL && pVs2 != NULL)
        {
            int i;
            for(i = 0; i < *pBestStatesLen; i++)
                pVs2[i] = pSumVs[pBestStatesIndices[i]];
        }

        if(ret)
        {
            printf("### tdlGetBestState returned error (%i)\n", ret);
            return -1;
        }
    }
    else if((ensembleDecision == eEnsembleDecision_average_state_values_decision) ||
            (ensembleDecision == eEnsembleDecision_average_state_values_decision_weighted)
    )
    {
        int ret = tdlGetBestState2(pS, NULL, micro, stateDecision, lNetAll.decisionNoise, lNetAll.stateValueImprecision, tau, pBestStatesIndices, pVs, NULL, pBestStatesLen, pMinDiff, pMaxDiff);

        int i;
        if(pSumVs != NULL && pVs2 != NULL)
        {
            for(i = 0; i < *pBestStatesLen; i++)
                pVs2[i] = pSumVs[pBestStatesIndices[i]];
        }

        for(i = 0; i < *pBestStatesLen; i++)
            pVsR[i] = valuesThisAgent[pBestStatesIndices[i]];

        if(ret)
        {
            printf("### tdlGetBestState returned error (%i)\n", ret);
            return -2;
        }
    }
    else if(ensembleDecision == eEnsembleDecision_voting_decision ||
            ensembleDecision == eEnsembleDecision_voting_decision_weighted
    )
    {
        int z;
        double votingCount[micro];
        unsigned long i;

        for(i = 0; i < micro; i++)
            votingCount[i] = 0;

        // Determine the individual decision of each agent
        for(z = 0; z < lNetAll.nAgents; z++)
        {
            unsigned long nStates = 0;
            unsigned long IArray[micro];
            double VsArray[micro];

            int ret;
            if(z == n)
                ret = tdlGetBestState2(pS, NULL, micro, stateDecision, lNetAll.decisionNoise, lNetAll.stateValueImprecision, tau, IArray, VsArray, NULL, &nStates, pMinDiff, pMaxDiff);
            else
                ret = tdlGetBestState2(state[z], NULL, micro, stateDecision, lNetAll.decisionNoise, lNetAll.stateValueImprecision, tau, IArray, VsArray, NULL, &nStates, NULL, NULL);

            if(ret)
            {
                printf("### tdlGetBestState returned error (%i)\n", ret);
                return -3;
            }

            int j;
            for(j = 0; j < nStates; j++)
            {
                if(IArray[j] < 0 || IArray[j] >= micro)
                {
                    printf("### IArray[j] (%li)\n", IArray[j]);
                    return -4;
                }
                
                if(ensembleDecision == eEnsembleDecision_voting_decision_weighted)
                {
                    if(z == n)
                    {
                        // Example with agents = 5:
                        // weightCurrentAgent = 1, 4 votes (breaks ties)
                        // weightCurrentAgent = 0.7, 3 votes (breaks ties)
                        // weightCurrentAgent = 0.5, 2 votes (breaks ties)
                        // weightCurrentAgent = 0.2, 1 votes (breaks ties)
                        // weightCurrentAgent = 0, 0 votes (breaks ties)
                        
                        // Add 0.01 so the acting agent breaks ties
                        votingCount[IArray[j]] += ceil((lNetAll.nAgents - 1) * lNetAll.weightCurrentAgent) + 0.01;
                    }
                    else
                        votingCount[IArray[j]] ++;
                }
                else
                    votingCount[IArray[j]]++;
            }
        }

        unsigned long bestI[micro];
        unsigned long nBest = 0;
        double bestVotingCount = -1;
        for(i = 0; i < micro; i++)
        {
            if(votingCount[i] > bestVotingCount)
            {
                bestVotingCount = votingCount[i];
                bestI[0] = i;
                nBest = 1;
            }
            else if(votingCount[i] == bestVotingCount)
            {
                bestI[nBest] = i;
                nBest++;
            }
        }

        if(nBest <= 0)
        {
            printf("### nBest <= 0, micro (%li), (%lf)\n", micro, bestVotingCount);
            return -4;
        }

        for(i = 0; i < nBest; i++)
            pBestStatesIndices[i] = bestI[i];
        *pBestStatesLen = nBest;

        if(pSumVs != NULL && pVs2 != NULL)
        {
            for(i = 0; i < *pBestStatesLen; i++)
                pVs2[i] = pSumVs[pBestStatesIndices[i]];
        }

        for(i = 0; i < *pBestStatesLen; i++)
        {
            pVs[i] = pS[pBestStatesIndices[i]].Vs;
            pVsR[i] = pS[pBestStatesIndices[i]].Vs;
        }
    }

    return 0;
}

double randValDouble (
    double min,
    double max)
{
    /*Zufallszahl zwischen min und max */
    if(min==0.0 && max==0.0)
        return 0.0;
    double val = min + ((max - min) * random () / (RAND_MAX + 1.0));
    return val;
}

int32_t randValLong (
    int32_t min,
    int32_t max)
{
    /*Zufallszahl zwischen min und max */
    if(min==0 && max==0)
      return 0;
    int32_t val = random () % (max - min + 1) + min;
    return val;
}

double signf(double v)
{
    return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0);
}

int bubbleRank(
    double *pV,
    int N,
    int descending,
    int *pIndices,
    int *pRank)
{
    int j;
    
    for(j = 0; j < N; j++)
        pIndices[j] = j;
    
    /* bubble sort the array */
    int swapped;
    do
    {
        swapped = 0;
        for(j = 1; j < N; j++)
        {                        
            if(!descending)
            {
                if (pV[pIndices[j]] < pV[pIndices[j - 1]])
                    swapped = 1;                            
            }
            else
            {
                if (pV[pIndices[j]] > pV[pIndices[j - 1]])
                    swapped = 1;                            
            }
            
            if (swapped)
            {
                int tmp = pIndices[j];
                pIndices[j] = pIndices[j - 1];
                pIndices[j - 1] = tmp;
            }
        }
    } while(swapped);
    
    for(j = 0; j < N; j++)
        pRank[pIndices[j]] = j + 1;
    
    return 0;
}

int tdlParseConfig(tConfigParam *pParam, char *filename)
{
    FILE *fp;
    char var[512], value[512], line[512];

    fp = fopen (filename, "r");
    if(fp == NULL)
    {
        printf("### tdlParseConfig: Unable to open file (%s)\n", filename);
        return -1;
    }

    pParam->decisionNoise = 0;
    pParam->agents = 1;
    pParam->updateWeightsImmediate = false;
    pParam->weightedAverage = false;
    pParam->weightedDecisions = false;
    pParam->weightCurrentAgent = 0;
    pParam->learnFromAverageStateValues = false;
    pParam->averageDecision = false;
    pParam->votingDecision = false;
    pParam->averageDecisionBenchmark = false;
    pParam->votingDecisionBenchmark = false;
    pParam->normalizeLearningRate = false;
    pParam->epsilonBase = 1.0;
    pParam->epsilonBaseOpp = 1.0;
    pParam->tdcOwnSummedGradient = false;
    pParam->replacingTraces = false;
    
    while (fgets (line, sizeof (line), fp))
    {
        memset (var, 0, sizeof (var));
        memset (value, 0, sizeof (value));
        if (sscanf (line, "%[^ \t=]%*[\t ]=%*[\t ]%[^\n]", var, value) == 2)
        {
            if (strcmp (var, "reward_won") == 0)
                pParam->reward_won = atof (value);
            if (strcmp (var, "reward_lost") == 0)
                pParam->reward_lost = atof (value);
            if (strcmp (var, "reward_draw") == 0)
                pParam->reward_draw = atof (value);
            if (strcmp (var, "gamma") == 0)
                pParam->gamma = atof (value);
            if (strcmp (var, "lambda") == 0)
                pParam->lambda = atof (value);
            if (strcmp (var, "alpha") == 0)
                pParam->alpha = atof (value);
            if (strcmp (var, "iterations") == 0)
                pParam->iterations = atoi (value);
            if (strcmp (var, "epsilon") == 0)
                pParam->epsilon = atof (value);
            if (strcmp (var, "epsilonBase") == 0)
                pParam->epsilonBase = atof (value);
            if (strcmp (var, "tau") == 0)
                pParam->tau = atof (value);
            if (strcmp (var, "epsilonOpp") == 0)
                pParam->epsilonOpp = atof (value);
            if (strcmp (var, "epsilonBaseOpp") == 0)
                pParam->epsilonBaseOpp = atof (value);
            if (strcmp (var, "tauOpp") == 0)
                pParam->tauOpp = atof (value);
            if (strcmp (var, "stateValueImprecision") == 0)
                pParam->stateValueImprecision = atof (value);
            if (strcmp (var, "errorStatistics") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->errorStatistics = true;
                else
                    pParam->errorStatistics = false;
            }
            if (strcmp (var, "errorEpsilon") == 0)
                pParam->errorEpsilon = atof (value);
            if (strcmp (var, "totalErrorIterations") == 0)
                pParam->totalErrorIterations = atoi (value);
            if (strcmp (var, "groupErrorIterations") == 0)
                pParam->groupErrorIterations = atoi (value);
            if (strcmp (var, "expectedTotalRewardIterations") == 0)
                pParam->expectedTotalRewardIterations = atoi (value);
            if (strcmp (var, "bellmanErrorIterations") == 0)
                pParam->bellmanErrorIterations = atoi (value);
            if (strcmp (var, "stateValueFunctionApproximation") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->stateValueFunctionApproximation = true;
                else
                    pParam->stateValueFunctionApproximation = false;
            }
            if (strcmp (var, "minA") == 0)
                pParam->minA = atof (value);
            if (strcmp (var, "maxA") == 0)
                pParam->maxA = atof (value);
            if (strcmp (var, "linearApproximation") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->linearApproximation = true;
                else
                    pParam->linearApproximation = false;
            }
            if (strcmp (var, "a") == 0)
                pParam->a = atof (value);
            if (strcmp (var, "trainingMode") == 0)
                pParam->trainingMode = atoi (value);
            if (strcmp (var, "mlpConfigFile") == 0)
                strcpy(pParam->conffile, value);
            if (strcmp (var, "mlpSaveFile") == 0)
                strcpy(pParam->savfile, value);
            if (strcmp (var, "beta") == 0)
            {
                if(strcmp(value,"alpha") == 0)
                    pParam->beta = pParam->alpha;
                else
                    pParam->beta = atof (value);
            }
            if (strcmp (var, "gradientPrimeFactor") == 0)
                pParam->gradientPrimeFactor = atof (value);
            if (strcmp (var, "normalizeGradientByAgents") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->normalizeGradientByAgents = true;
                else
                    pParam->normalizeGradientByAgents = false;
            }
            if (strcmp (var, "batchSize") == 0)
                pParam->batchSize = atoi (value);
            if (strcmp (var, "agents") == 0)
                pParam->agents = atoi (value);
            if (strcmp (var, "updateWeightsImmediate") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->updateWeightsImmediate = true;
                else
                    pParam->updateWeightsImmediate = false;
            }
            if (strcmp (var, "weightedAverage") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->weightedAverage = true;
                else
                    pParam->weightedAverage = false;
            }
            if (strcmp (var, "weightedDecisions") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->weightedDecisions = true;
                else
                    pParam->weightedDecisions = false;
            }
            if (strcmp (var, "weightCurrentAgent") == 0)
                pParam->weightCurrentAgent = atof (value);
            if (strcmp (var, "learnFromAverageStateValues") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->learnFromAverageStateValues = true;
                else
                    pParam->learnFromAverageStateValues = false;
            }
            if (strcmp (var, "averageDecision") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->averageDecision = true;
                else
                    pParam->averageDecision = false;
            }
            if (strcmp (var, "votingDecision") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->votingDecision = true;
                else
                    pParam->votingDecision = false;
            }
            if (strcmp (var, "averageDecisionBenchmark") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->averageDecisionBenchmark = true;
                else
                    pParam->averageDecisionBenchmark = false;
            }
            if (strcmp (var, "votingDecisionBenchmark") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->votingDecisionBenchmark = true;
                else
                    pParam->votingDecisionBenchmark = false;
            }
            if (strcmp (var, "decisionNoise") == 0)
                pParam->decisionNoise = atof (value);
            int j;
            for(j = 1; j <= 100; j++)
            {
                char cmpstr[100];
                snprintf(cmpstr, 100, "seed%i", j);
                if (strcmp (var, cmpstr) == 0)
                    pParam->seed[j - 1] = atoi (value);
                else
                    pParam->seed[j - 1] = 0;
            }
            if (strcmp (var, "normalizeLearningRate") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->normalizeLearningRate = true;
                else
                    pParam->normalizeLearningRate = false;
            }
            
            if (strcmp (var, "tdcOwnSummedGradient") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->tdcOwnSummedGradient = true;
                else
                    pParam->tdcOwnSummedGradient = false;
            }            
            
            if (strcmp (var, "replacingTraces") == 0)
            {
                if(strcmp(value,"y") == 0)
                    pParam->replacingTraces = true;
                else
                    pParam->replacingTraces = false;
            }            
        }
    } /*while */

    if(pParam->agents <= 1)
        pParam->agents = 1;
    
    if(pParam->weightCurrentAgent == 0)
        pParam->weightCurrentAgent = 1.0 / pParam->agents;
    
    if(pParam->normalizeGradientByAgents)
        pParam->gradientPrimeFactor /= (double) pParam->agents;
    
    printf("tdlParseConfig: Following parameters have been set:\n");
    printf("reward_won (%lf), reward_lost (%lf), reward_draw(%lf), gamma (%lf)\n", pParam->reward_won, pParam->reward_lost, pParam->reward_draw, pParam->gamma);
    printf("alpha (%lf), iterations (%li)\n", pParam->alpha, pParam->iterations);
    printf("epsilon (%lf), epsilonBase (%1.16lf), epsilonOpp (%lf), epsilonBaseOpp (%1.16lf), tau (%lf), tauOpp (%lf)\n", pParam->epsilon, pParam->epsilonBase, pParam->epsilonOpp, pParam->epsilonBaseOpp, pParam->tau, pParam->tauOpp);
    printf("stateValueImprecision (%lf), decisionNoise (%lf)\n", pParam->stateValueImprecision, pParam->decisionNoise);
    printf("errorStatistics (%i), errorEpsilon (%lf), totalErrorIterations (%i), groupErrorIterations (%i), bellmanErrorIterations (%i)\n", pParam->errorStatistics, pParam->errorEpsilon, pParam->totalErrorIterations, pParam->groupErrorIterations, pParam->bellmanErrorIterations);
    printf("expectedTotalRewardIterations (%i)\n", pParam->expectedTotalRewardIterations);
    printf("batchSize (%i), normalizeLearningRate (%i)\n", pParam->batchSize, pParam->normalizeLearningRate);
    printf("tdcOwnSummedGradient (%i), replacingTraces (%i)\n", pParam->tdcOwnSummedGradient, pParam->replacingTraces);
    if(!pParam->stateValueFunctionApproximation)
        printf("Learning in state-tables, minA (%lf), maxA (%lf)\n", pParam->minA, pParam->maxA);
    else
    {
        if(pParam->linearApproximation)
            printf("Linear function approximation, a (%lf), trainingMode (%i), gradientPrimeFactor (%lf), normalizeGradientByAgents (%i), lambda (%lf)\n", pParam->a, pParam->trainingMode, pParam->gradientPrimeFactor, pParam->normalizeGradientByAgents, pParam->lambda);
        else
            printf("Nonlinear (MLP) function approximation, mlpConfigFile (%s), mlpSaveFile (%s), beta (%lf), gradientPrimeFactor (%lf), normalizeGradientByAgents (%i), lambda (%lf)\n", pParam->conffile, pParam->savfile, pParam->beta, pParam->gradientPrimeFactor, pParam->normalizeGradientByAgents, pParam->lambda);
    }
    if(pParam->agents == 1)
        printf("1 agent, single decision\n");
    else
    {
        printf("%i agents, updateWeightsImmediate (%i), weightedAverage (%i), weightedDecisions (%i), weightCurrentAgent (%lf), learnFromAverageStateValues (%i)\n", pParam->agents, pParam->updateWeightsImmediate, pParam->weightedAverage, pParam->weightedDecisions, pParam->weightCurrentAgent, pParam->learnFromAverageStateValues);
        printf("averageDecision (%i), votingDecision (%i)\n", pParam->averageDecision, pParam->votingDecision);
        printf("averageDecisionBenchmark (%i), votingDecisionBenchmark (%i)\n", pParam->averageDecisionBenchmark, pParam->votingDecisionBenchmark);
    }

    printf("seed1: %i\n", pParam->seed[0]);
    
    fclose (fp);
    
    return 0;
}
