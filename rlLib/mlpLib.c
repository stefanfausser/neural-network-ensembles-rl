// The MIT License (MIT)
// 
// Copyright (c) 2008 - 2015 Stefan Faußer
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
 * \file mlpLib.c
 * \brief Multi-layer perceptron (MLP) libary
 *
 * This MLP library supports Online, Batch, Batch with Momentumterm, RPROP-, RPROP+ and Quickprop learning.
 * Further, it can be used for learning state-action value functions (reinforcement learning)
 * when used in combination with the tdlLib.
 * 
 * \author Stefan Fausser
 * 
 * Modification history:
 * 
 * 2008-04-01, S. Fausser - written
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <string.h>             /*memset */
#include "mlpLib.h"
#include "matrixLib.h"
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_blas.h>

#define MAX_WEIGHT_ONLINE_MODE  99999999
#define ZERO_PRECISION          0.001    

/**
 * structure including all mlp parameters for one mlp network
 */
struct mlp_vars
{
    /**
     * weights for neurons in hidden layer
     */
    double ***w;
    double ***w_prime;
    
    /**
     * bias for neurons in hidden layer
     */
    double **theta;
    double **theta_prime;
    
    /**
     * modifyable eta values for supersab / resilient weight udpate mode in hidden layer
     */
    double ***eta_w;

    /**
     * last gradient (t-1) in hidden layer
     */
    double ***desc_w;

    /**
     * last weight-delta in hidden layer
     */
    double ***delta_w1;

    /**
     * dendrit potential in hidden layer
     */
    double ***u;

    /**
     * axon potential in hidden layer
     */
    double ***y;
    double ***y_tsignal;

    /**
     * delta values in hidden layer
     */
    double ***delta;
    double ****gradvec;
    double ****gradvec_tsignal;
    double ****grad2vec;
    double ****gradientTrace;
    /**
     * weights for neurons in output layer
     */
    double **w2;
    double **w2_prime;
    
    /**
     * bias for neurons in output layer
     */
    double *theta2;
    double *theta2_prime;
    
    /**
     * modifyable eta values for supersab / resilient weight udpate mode in output layer
     */
    double **eta_w2;

    /**
     * last gradient in output layer
     */
    double **desc_w2;

    /**
     * last weight-delta in output layer
     */
    double **delta_w2;

    /**
     * dendrit potential in output layer
     */
    double *u2;

    /**
     * axon potential in output layer
     */
    double **y2;
    double **y2_prime;

    /**
     * delta values in output layer
     */
    double **delta2;
    double **gradvec2;
    double **gradvec2_tsignal;
    double **grad2vec2;
    double **gradientTrace2;
    
    /**
     * mlp_param structure
     */
    struct mlp_param mlpP;

    /**
     * mlp_init_values structure
     */
    struct mlp_init_values mlpIv;

    /**
     * maximum number of training samples in one epoche
     */
    uint32_t micro_max;

    /**
     * floating point value that will be used to random initialize the weights
     */
    double a;

    double aOut;
    
    int weightNormalizedInitialization;
    
    int thresholdZeroInitialization;
};

/**
 * Gradient factor
 */
#define GRADIENT_FACTOR (-1.0)

/*Calculate S = summed gradient information (weight specific) over all patterns of the pattern set*/
#define CALC_GRADIENT_WEIGHTS_OUTPUT(_S) \
    _S = 0; \
    for (muster=0;muster<micro_max;muster++) { \
        _S += pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers-1][muster][i] * pMlpVA[mlpfd]->delta2[muster][j]; \
    } \
    _S *= GRADIENT_FACTOR;

#define CALC_GRADIENT_BIAS_OUTPUT(_S) \
    _S = 0; \
    for (muster=0;muster<micro_max;muster++) { \
        _S += pMlpVA[mlpfd]->delta2[muster][j]; \
    }

#define CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER(_S) \
    _S = 0; \
    for (muster=0;muster<micro_max;muster++) { \
        _S += pData->x[muster][k] * pMlpVA[mlpfd]->delta[l][muster][i]; \
    } \
    _S *= GRADIENT_FACTOR;

#define CALC_GRADIENT_WEIGHTS_HIDDEN(_S) \
    _S = 0; \
    for (muster=0;muster<micro_max;muster++) { \
        _S += pMlpVA[mlpfd]->y[l-1][muster][i2] * pMlpVA[mlpfd]->delta[l][muster][i]; \
    } \
    _S *= GRADIENT_FACTOR;

#define CALC_GRADIENT_BIAS_HIDDEN(_S) \
    _S = 0; \
    for (muster=0;muster<micro_max;muster++) { \
        _S += pMlpVA[mlpfd]->delta[l][muster][i]; \
    }

#define CALC_RESILIENT_UPDATE_WEIGHT(_S,_weight,_delta) \
    /*Berechnung der aktuellen Gewichtsaenderung*/ \
    if ( _S > 0 ) { \
        /*S(t)>0*/ \
        _weight = _weight - _delta; \
    } \
    else if ( _S < 0 ) { \
        /*S(t-1)<0*/ \
        _weight = _weight + _delta; \
    } \
    else { \
        /*S(t-1)*S(t)=0*/ \
    }

#define CALC_RESILIENT_DELTA_WEIGHT(_S,_delta_weight,_delta) \
    /*Berechnung der aktuellen Gewichtsaenderung*/ \
    if ( _S > 0 ) { \
        /*S(t)>0*/ \
        _delta_weight = - _delta; \
    } \
    else if ( _S < 0 ) { \
        /*S(t-1)<0*/ \
        _delta_weight = + _delta; \
    } \
    else { \
        _delta_weight = 0; \
    }

#define CALC_RESILIENT_DELTA_INCREASE(_delta,_eta_pos,_max_delta) \
    _delta = _eta_pos * _delta; \
    if ( _delta > _max_delta ) \
        _delta = _max_delta;

#define CALC_RESILIENT_DELTA_DECREASE(_delta,_eta_neg,_min_delta) \
    _delta = _eta_neg * _delta; \
    if ( _delta < _min_delta ) \
        _delta = _min_delta;

#define CALC_RESILIENT_DS(_dS,_S,_desc_w) \
    _dS = sign(_S) * sign(_desc_w);

/* locals */
static int maxMlps = -1;
static struct mlp_vars **pMlpVA = NULL;
static int nRegisteredMlps = -1;
static int *mlpRegistered = NULL;


int mlpLibInit (
    uint16_t nrMlps)
{
    int i;

    maxMlps = nrMlps;

    mlpRegistered = malloc (sizeof (*mlpRegistered) * maxMlps);
    pMlpVA = malloc (sizeof (**pMlpVA) * maxMlps);

    nRegisteredMlps = 0;
    for (i = 0; i < maxMlps; i++)
    {
        mlpRegistered[i] = 0;
    }

    return 0;
}


int mlpLibDeinit (
    )
{
    free (mlpRegistered);
    free (pMlpVA);

    return 0;
}

/*local function prototypes*/
static int allocateWeights (
    struct mlp_vars *pMlpV);
static int initializeWeights (
    struct mlp_vars *pMlpV,
    int verbose,
    unsigned int seed);
static double randomVal (
    double min,
    double max);

/*implementation*/

#define TRANSFKT(Y,X,TYPE,BETA) \
    /*Transferfunktion*/ \
    if (TYPE==0) { \
        /*Logistische Funktion*/ \
        Y = 1.0 / (1.0 + exp(-X)); \
    } \
    else if (TYPE==1) { \
        /*Fermi Funktion*/ \
        Y = 1.0 / (1.0 + exp(-BETA*X)); \
    } \
    else if (TYPE==2) { \
        /*Tangens hyperbolicus*/ \
        Y = tanh(X); \
    } \
    else { \
        Y = X; \
    }

#define TRANSFKT_DERIVATIVE(Y,FX,TYPE,BETA) \
    /*1. Ableitung der Transferfunktion*/ \
    if (TYPE==0) { \
        /*1. Ableitung Logistische Funktion*/ \
        Y = FX * (1.0 - FX); \
    } \
    else if (TYPE==1) { \
        /*1. Ableitung Fermi Funktion*/ \
        Y = BETA * FX * (1.0 - FX); \
    } \
    else if (TYPE==2) { \
        /*1. Ableitung Tangens hyperbolicus*/ \
        Y = 1.0 - (FX * FX); \
    } \
    else { \
        Y = 1; \
    }

#define TRANSFKT_DERIVATIVE_2(Y,FX,TYPE,BETA) \
    /*2. Ableitung der Transferfunktion*/ \
    if (TYPE==0) { \
        /*2. Ableitung Logistische Funktion*/ \
        Y = FX * ((1.0 - FX) * ((1.0 - FX) - FX)); \
    } \
    else if (TYPE==1) { \
        /*2. Ableitung Fermi Funktion*/ \
        Y = BETA * FX * ((1.0 - FX) * ((1.0 - FX) - FX)); \
    } \
    else if (TYPE==2) { \
        /*2. Ableitung Tangens hyperbolicus*/ \
        Y = -2.0 * (FX - pow(FX, 3.0)); \
    } \
    else { \
        Y = 0; \
    }

static int allocateWeights (
    struct mlp_vars *pMlpV)
{
    uint16_t l;
    int ret;

    if (pMlpV == NULL)
        return -1;

    /*Speicher allozieren fuer w[pMlpV->mlpIv.nrHiddenLayers][pMlpV->mlpIv.m][pMlpV->mlpIv.h] */
    pMlpV->w = (double ***) malloc (pMlpV->mlpIv.nrHiddenLayers * sizeof (double **));
    if (pMlpV->w == NULL)
    {
        printf ("### not enough memory to allocate w.\n");
        return -2;
    }

    pMlpV->w_prime = (double ***) malloc (pMlpV->mlpIv.nrHiddenLayers * sizeof (double **));
    if (pMlpV->w_prime == NULL)
    {
        printf ("### not enough memory to allocate w_prime.\n");
        return -2;
    }

    /*Speicher allozieren fuer theta[pMlpV->mlpIv.nrHiddenLayers][zeile] */
    pMlpV->theta = (double **) malloc (pMlpV->mlpIv.nrHiddenLayers * sizeof (double *));
    if (pMlpV->theta == NULL)
    {
        printf ("### not enough memory to allocate theta1.\n");
        return -3;
    }

    pMlpV->theta_prime = (double **) malloc (pMlpV->mlpIv.nrHiddenLayers * sizeof (double *));
    if (pMlpV->theta_prime == NULL)
    {
        printf ("### not enough memory to allocate theta1_prime.\n");
        return -3;
    }

    for (l = 0; l < pMlpV->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            ret = allocateMatrix2 (&pMlpV->w[l], pMlpV->mlpIv.m, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->w_prime[l], pMlpV->mlpIv.m, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
        }
        else
        {
            ret = allocateMatrix2 (&pMlpV->w[l], pMlpV->mlpIv.h[l - 1], pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->w_prime[l], pMlpV->mlpIv.h[l - 1], pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;            
        }
        ret = allocateVector (&pMlpV->theta[l], pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
        ret = allocateVector (&pMlpV->theta_prime[l], pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
    }

    /*Speicher allozieren fuer w2[zeile][spalte] */
    ret = allocateMatrix2 (&pMlpV->w2, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1], pMlpV->mlpIv.n);
    if (ret)
        return -2;

    ret = allocateMatrix2 (&pMlpV->w2_prime, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1], pMlpV->mlpIv.n);
    if (ret)
        return -2;

    /*Speicher allozieren fuer theta2[zeile] */
    ret = allocateVector (&pMlpV->theta2, pMlpV->mlpIv.n);
    if (ret)
        return -2;

    ret = allocateVector (&pMlpV->theta2_prime, pMlpV->mlpIv.n);
    if (ret)
        return -2;

    return 0;
}


static int initializeWeights (
    struct mlp_vars *pMlpV,
    int verbose,
    unsigned int seed)
{
    uint16_t k, i, i2, j, l;
    int ret;
    
    if (pMlpV == NULL)
        return -1;

    /*Allocate weights first */
    ret = allocateWeights (pMlpV);
    if (ret)
        return -2;

    /*Schritt 1: Initialisierung */
    srandom (seed);

    // Attention: The _prime weights are only used for TDC_MODE and they are just set to zero
    
    /* Hidden Layer */

    for (l = 0; l < pMlpV->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            for (k = 0; k < pMlpV->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpV->mlpIv.h[l]; i++)
                {
                    double val = pMlpV->a;
                    if(pMlpV->weightNormalizedInitialization == 1)
                        val *= 1.0 / sqrt(pMlpV->mlpIv.m);
                    else if(pMlpV->weightNormalizedInitialization == 2)
                        val *= sqrt(6.0 / (pMlpV->mlpIv.m + pMlpV->mlpIv.h[l]));
                                        
                    pMlpV->w[l][k][i] = randomVal (-val, val);                    
                    pMlpV->w_prime[l][k][i] = 0;
                }
            }
        }
        else
        {
            for (i = 0; i < pMlpV->mlpIv.h[l - 1]; i++)
            {
                for (i2 = 0; i2 < pMlpV->mlpIv.h[l]; i2++)
                {
                    double val = pMlpV->a;
                    if(pMlpV->weightNormalizedInitialization == 1)
                        val *= 1.0 / sqrt(pMlpV->mlpIv.h[l - 1]);
                    else if(pMlpV->weightNormalizedInitialization == 2)
                        val *= sqrt(6.0 / (pMlpV->mlpIv.h[l - 1] + pMlpV->mlpIv.h[l]));

                    pMlpV->w[l][i][i2] = randomVal (-val, val);                    
                    pMlpV->w_prime[l][i][i2] = 0;                    
                }
            }
        }
        
        for (i = 0; i < pMlpV->mlpIv.h[l]; i++)
        {
            if(pMlpV->thresholdZeroInitialization)
                pMlpV->theta[l][i] = 0;
            else
            {
                double val = pMlpV->a;
                if(pMlpV->weightNormalizedInitialization == 1)
                {
                    if(l == 0)
                        val *= 1.0 / sqrt(pMlpV->mlpIv.m);
                    else
                        val *= 1.0 / sqrt(pMlpV->mlpIv.h[l - 1]);
                }
                else if(pMlpV->weightNormalizedInitialization == 2)
                {
                    if(l == 0)
                        val *= sqrt(6.0 / (pMlpV->mlpIv.m + pMlpV->mlpIv.h[l]));
                    else
                        val *= sqrt(6.0 / (pMlpV->mlpIv.h[l - 1] + pMlpV->mlpIv.h[l]));
                }
                
                pMlpV->theta[l][i] = randomVal (-val, val);
            }
            
            pMlpV->theta_prime[l][i] = 0;
        }
    }

    /* output layer */
    for (i = 0; i < pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1]; i++)
    {
        for (j = 0; j < pMlpV->mlpIv.n; j++)
        {
            double val = pMlpV->a;
            if(pMlpV->aOut > 0)
                val = pMlpV->aOut;            
            
            if(pMlpV->weightNormalizedInitialization == 1)
                val *= 1.0 / sqrt(pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1]);
            else if(pMlpV->weightNormalizedInitialization == 2)
                val *= sqrt(6.0 / (pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + pMlpV->mlpIv.n));

            pMlpV->w2[i][j] = randomVal (-val, val);
            pMlpV->w2_prime[i][j] = 0;            
        }
    }
    for (j = 0; j < pMlpV->mlpIv.n; j++)
    {
        if(pMlpV->thresholdZeroInitialization)
            pMlpV->theta2[j] = 0;
        else
        {
            double val = pMlpV->a;
            if(pMlpV->aOut > 0)
                val = pMlpV->aOut;
            
            if(pMlpV->weightNormalizedInitialization == 1)
                val *= 1.0 / sqrt(pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1]);
            else if(pMlpV->weightNormalizedInitialization == 2)
                val *= sqrt(6.0 / (pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + pMlpV->mlpIv.n));

            pMlpV->theta2[j] = randomVal (-val, val);        
        }
        
        pMlpV->theta2_prime[j] = 0;
    }

    return 0;
}

static int mlp_reset_specialvars (
    int mlpfd)
{
    uint16_t k, i, i2, j;
    int32_t l;                     /*wegen for Schleife => Fehlerrückvermittlung... */

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    double eta = pMlpVA[mlpfd]->mlpP.eta_start;
    if(pMlpVA[mlpfd]->mlpP.etaNormalize)
        eta /= (double) pMlpVA[mlpfd]->micro_max;
    
    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    pMlpVA[mlpfd]->delta_w1[l][k][i] = 0;
                    pMlpVA[mlpfd]->eta_w[l][k][i] = eta;
                    pMlpVA[mlpfd]->desc_w[l][k][i] = 0;
                }
            }
        }
        else
        {
            for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    pMlpVA[mlpfd]->delta_w1[l][i2][i] = 0;
                    pMlpVA[mlpfd]->eta_w[l][i2][i] = eta;
                    pMlpVA[mlpfd]->desc_w[l][i2][i] = 0;
                }
            }
        }
    }

    for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
    {                           /*h+1 */
        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
        {
            pMlpVA[mlpfd]->delta_w2[i][j] = 0;
            pMlpVA[mlpfd]->eta_w2[i][j] = eta;
            pMlpVA[mlpfd]->desc_w2[i][j] = 0;
        }
    }
    return 0;
}

static double randomVal (
    double min,
    double max)
{
    if(min == 0 && max == 0)
        return 0;

    double val = min + ((max - min) * random () / (RAND_MAX + 1.0));
    
    return val;
}

int openMlpParameterFile (
    const char *filename,
    struct mlp_init_values *pMlpInitVals,
    struct mlp_param *pMlpP,
    uint32_t *seed,
    double *a,
    double *aOut,
    int *weightNormalizedInitialization,
    int *thresholdZeroInitialization)
{
    FILE *fp;
    char var[512], value[512], line[512];
    int i = 0;
    char str[1000];

    if (pMlpP == NULL)
        return -1;

    if (pMlpInitVals == NULL)
        return -2;

    /* Set defaults (all zero) */
    memset(pMlpP, 0, sizeof(struct mlp_param));
    
    fp = fopen (filename, "r");
    if (fp)
    {
        while (fgets (line, sizeof (line), fp))
        {
            memset (var, 0, sizeof (var));
            memset (value, 0, sizeof (value));
            if (sscanf (line, "%[^ \t=]%*[\t ]=%*[\t ]%[^\n]", var, value) == 2)
            {
                if (strcmp (var, "seed") == 0)
                    *seed = atoi (value);
                if (strcmp (var, "a") == 0)
                    *a = atof (value);
                if (strcmp (var, "aOut") == 0)
                    *aOut = atof (value);
                if (strcmp (var, "weightNormalizedInitialization") == 0)
                    *weightNormalizedInitialization = atoi (value);
                if (strcmp (var, "thresholdZeroInitialization") == 0)
                    *thresholdZeroInitialization = atoi (value);
                if (strcmp (var, "nrHiddenLayers") == 0)
                    pMlpInitVals->nrHiddenLayers = atoi (value);
                for (i = 0; i < pMlpInitVals->nrHiddenLayers; i++)
                {
                    sprintf (str, "h%i", i);
                    if (strcmp (var, str) == 0)
                        pMlpInitVals->h[i] = atoi (value);
                }
                if (strcmp (var, "m") == 0)
                    pMlpInitVals->m = atoi (value);
                if (strcmp (var, "n") == 0)
                    pMlpInitVals->n = atoi (value);
                if (strcmp (var, "maxIterations") == 0)
                    pMlpP->maxIterations = atoi (value);
                if (strcmp (var, "eta1") == 0)
                    pMlpP->eta1 = atof (value);
                if (strcmp (var, "eta2") == 0)
                    pMlpP->eta2 = atof (value);
                if (strcmp (var, "etaNormalize") == 0)
                    pMlpP->etaNormalize = atoi (value);
                if (strcmp (var, "epsilon") == 0)
                    pMlpP->epsilon = atof (value);
                if (strcmp (var, "transFktTypeHidden") == 0)
                    pMlpInitVals->transFktTypeHidden = atoi (value);
                if (strcmp (var, "transFktTypeOutput") == 0)
                    pMlpInitVals->transFktTypeOutput = atoi (value);
                if (strcmp (var, "hasThresholdOutput") == 0)
                    pMlpInitVals->hasThresholdOutput = atoi (value);
                if (strcmp (var, "beta") == 0)
                    pMlpP->beta = atof (value);
                if (strcmp (var, "eta_pos") == 0)
                    pMlpP->eta_pos = atof (value);
                if (strcmp (var, "eta_neg") == 0)
                    pMlpP->eta_neg = atof (value);
                if (strcmp (var, "eta_start") == 0)
                    pMlpP->eta_start = atof (value);
                if (strcmp (var, "eta_max") == 0)
                    pMlpP->eta_max = atof (value);
                if (strcmp (var, "eta_min") == 0)
                    pMlpP->eta_min = atof (value);
                if (strcmp (var, "beta_max") == 0)
                    pMlpP->beta_max = atof (value);
                if (strcmp (var, "alpha") == 0)
                    pMlpP->alpha = atof (value);
                if (strcmp (var, "trainingMode") == 0)
                    pMlpP->trainingMode = atoi (value);
                if (strcmp (var, "verboseOutput") == 0)
                    pMlpP->verboseOutput = atoi (value);
            }
        }                       /*while */
        fclose (fp);
    }                           /*if */
    else
    {
        return -2;
    }
    return 0;
}

int initializeMlpNet (
    struct mlp_init_values *pMlpInitVals,
    struct mlp_param *pMlpParam,                      
    double a,
    double aOut,
    int weightNormalizedInitialization,
    int thresholdZeroInitialization,
    uint32_t micro_max,
    unsigned int seed)
{
    int ret;
    uint16_t i, l;
    struct mlp_vars *pMlpV = NULL;

    if ((!pMlpInitVals->m) || (!pMlpInitVals->h) || (!pMlpInitVals->n) || (!pMlpInitVals->nrHiddenLayers))
        return -2;

    pMlpV = malloc (sizeof (*pMlpV));
    if (pMlpV == NULL)
    {
        printf ("### not enough memory to allocate pMlpV.\n");
        return -3;
    }

    pMlpV->mlpIv.m = pMlpInitVals->m;
    pMlpV->mlpIv.nrHiddenLayers = pMlpInitVals->nrHiddenLayers;
    
    for (l = 0; l < pMlpV->mlpIv.nrHiddenLayers; l++)
        pMlpV->mlpIv.h[l] = pMlpInitVals->h[l];

    pMlpV->mlpIv.n = pMlpInitVals->n;

    pMlpV->mlpIv.transFktTypeHidden = pMlpInitVals->transFktTypeHidden;
    pMlpV->mlpIv.transFktTypeOutput = pMlpInitVals->transFktTypeOutput;
    pMlpV->mlpIv.hasThresholdOutput = pMlpInitVals->hasThresholdOutput;
    pMlpV->a = a;
    pMlpV->aOut = aOut;
    pMlpV->weightNormalizedInitialization = weightNormalizedInitialization;
    pMlpV->thresholdZeroInitialization = thresholdZeroInitialization;
    pMlpV->micro_max = micro_max;

    ret = initializeWeights (pMlpV, 0, seed);
    if (ret)
    {
        free (pMlpV);
        return -4;
    }

    /*Speicher allozieren fuer eta[pMlpV->mlpIv.nrHiddenLayers][pMlpV->mlpIv.m+1][pMlpV->mlpIv.h[l]] */
    pMlpV->eta_w = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->eta_w == NULL)
    {
        printf ("### not enough memory to allocate eta_w.\n");
        return -5;
    }
    /*Speicher allozieren fuer desc_w[pMlpV->mlpIv.nrHiddenLayers][pMlpV->mlpIv.m+1][pMlpV->mlpIv.h[l]] */
    pMlpV->desc_w = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->desc_w == NULL)
    {
        printf ("### not enough memory to allocate desc_w.\n");
        return -6;
    }
    /*Speicher allozieren fuer delta_w1[pMlpV->mlpIv.nrHiddenLayers][pMlpV->mlpIv.m+1][pMlpV->mlpIv.h[l]] */
    pMlpV->delta_w1 = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->delta_w1 == NULL)
    {
        printf ("### not enough memory to allocate delta_w1.\n");
        return -7;
    }
    /*Speicher allozieeren fuer u[pMlpV->mlpIv.nrHiddenLayers][pMlpV->mlpIv.h[l]] */
    pMlpV->u = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->u == NULL)
    {
        printf ("### not enough memory to allocate u.\n");
        return -8;
    }
    /*Speicher allozieren fuer pData->y[pMlpV->mlpIv.nrHiddenLayers][micro_max][pMlpV->mlpIv.h[l]] */
    pMlpV->y = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->y == NULL)
    {
        printf ("### not enough memory to allocate y.\n");
        return -9;
    }
    pMlpV->y_tsignal = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->y_tsignal == NULL)
    {
        printf ("### not enough memory to allocate y_tsignal.\n");
        return -9;
    }
    /*Speicher allozieeren fuer delta[pMlpV->mlpIv.nrHiddenLayers][micro_max][pMlpV->mlpIv.h[l]] */
    pMlpV->delta = (double ***) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double **));
    if (pMlpV->delta == NULL)
    {
        printf ("### not enough memory to allocate delta.\n");
        return -10;
    }
    /*Speicher allozieeren fuer delta[pMlpV->mlpIv.nrHiddenLayers][micro_max][pMlpV->mlpIv.h[l]] */
    pMlpV->gradvec = (double ****) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double ***));
    if (pMlpV->gradvec == NULL)
    {
        printf ("### not enough memory to allocate gradvec.\n");
        return -10;
    }
    pMlpV->gradvec_tsignal = (double ****) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double ***));
    if (pMlpV->gradvec_tsignal == NULL)
    {
        printf ("### not enough memory to allocate gradvec_tsignal.\n");
        return -10;
    }
    pMlpV->grad2vec = (double ****) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double ***));
    if (pMlpV->grad2vec == NULL)
    {
        printf ("### not enough memory to allocate grad2vec.\n");
        return -10;
    }
    
    pMlpV->gradientTrace = (double ****) malloc ((pMlpV->mlpIv.nrHiddenLayers) * sizeof (double ***));
    if (pMlpV->gradientTrace == NULL)
    {
        printf ("### not enough memory to allocate gradientTrace.\n");
        return -10;
    }    

    for (l = 0; l < pMlpV->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            ret = allocateMatrix2 (&pMlpV->eta_w[l], pMlpV->mlpIv.m + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->desc_w[l], pMlpV->mlpIv.m + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->delta_w1[l], pMlpV->mlpIv.m + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
        }
        else
        {
            ret = allocateMatrix2 (&pMlpV->eta_w[l], pMlpV->mlpIv.h[l - 1] + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->desc_w[l], pMlpV->mlpIv.h[l - 1] + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
            ret = allocateMatrix2 (&pMlpV->delta_w1[l], pMlpV->mlpIv.h[l - 1] + 1, pMlpV->mlpIv.h[l]);
            if (ret)
                return -2;
        }

        if(l == 0)
        {
            if(l == pMlpV->mlpIv.nrHiddenLayers - 1)
            {
                // Naechste Schicht ist die Ausgabeschicht
                
                ret = allocateArray3 (&pMlpV->gradvec[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.n);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradvec_tsignal[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.n);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradientTrace[l], pMlpV->mlpIv.h[l], pMlpV->mlpIv.n, pMlpV->mlpIv.m + 1);
                if (ret)
                    return -2;
            }
            else
            {
                // Naechste Schicht ist die naechste Zwischenschicht

                ret = allocateArray3 (&pMlpV->gradvec[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1]);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradvec_tsignal[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1]);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradientTrace[l], pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1], pMlpV->mlpIv.m + 1);
                if (ret)
                    return -2;                
            }
        }
        else
        {
            if(l == pMlpV->mlpIv.nrHiddenLayers - 1)
            {
                // Naechste Schicht ist die Ausgabeschicht
                
                ret = allocateArray3 (&pMlpV->gradvec[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.n);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradvec_tsignal[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.n);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradientTrace[l], pMlpV->mlpIv.h[l], pMlpV->mlpIv.n, pMlpV->mlpIv.h[l - 1] + 1);
                if (ret)
                    return -2;                
            }
            else
            {
                // Naechste Schicht ist die naechste Zwischenschicht

                ret = allocateArray3 (&pMlpV->gradvec[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1]);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradvec_tsignal[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1]);
                if (ret)
                    return -2;
                ret = allocateArray3 (&pMlpV->gradientTrace[l], pMlpV->mlpIv.h[l], pMlpV->mlpIv.h[l + 1], pMlpV->mlpIv.h[l - 1] + 1);
                if (ret)
                    return -2;                
            }
        }

        ret = allocateMatrix2 (&pMlpV->u[l], micro_max, pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
        ret = allocateMatrix2 (&pMlpV->y[l], micro_max, pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
        ret = allocateMatrix2 (&pMlpV->y_tsignal[l], micro_max, pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
        ret = allocateMatrix2 (&pMlpV->delta[l], micro_max, pMlpV->mlpIv.h[l]);
        if (ret)
            return -2;
        ret = allocateArray3 (&pMlpV->grad2vec[l], micro_max, pMlpV->mlpIv.h[l], pMlpV->mlpIv.n);
        if (ret)
            return -2;        
    } /* for l */

    ret = allocateMatrix2 (&pMlpV->gradientTrace2, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + 1, pMlpV->mlpIv.n);
    if (ret)
        return -2;        
    ret = allocateMatrix2 (&pMlpV->eta_w2, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + 1, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->desc_w2, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + 1, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->delta_w2, pMlpV->mlpIv.h[pMlpV->mlpIv.nrHiddenLayers - 1] + 1, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateVector (&pMlpV->u2, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->y2, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->y2_prime, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->delta2, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->gradvec2, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->gradvec2_tsignal, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;
    ret = allocateMatrix2 (&pMlpV->grad2vec2, micro_max, pMlpV->mlpIv.n);
    if (ret)
        return -2;

    if (nRegisteredMlps < maxMlps)
    {
        nRegisteredMlps++;
        for (i = 0; i < maxMlps; i++)
        {
            if (mlpRegistered[i] == 0)
            {
                pMlpVA[i] = pMlpV;
                mlpRegistered[i] = 1;
                break;
            }
        }
    }
    else
    {
        printf ("### maximum number of mlps exceeded.\n");
        return -33;
    }

    setMlpParam(i, pMlpParam);
    mlp_reset_specialvars (i);
    clearGradientTrace(i);
    
    return i;
}


int cleanupMlpNet (
    int mlpfd)
{
    uint16_t l;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    /*Free everything for the hidden layer... */
    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            freeMatrix2 (pMlpVA[mlpfd]->w[l], pMlpVA[mlpfd]->mlpIv.m);
            freeMatrix2 (pMlpVA[mlpfd]->w_prime[l], pMlpVA[mlpfd]->mlpIv.m);
        }
        else
        {
            freeMatrix2 (pMlpVA[mlpfd]->w[l], pMlpVA[mlpfd]->mlpIv.h[l - 1]);
            freeMatrix2 (pMlpVA[mlpfd]->w_prime[l], pMlpVA[mlpfd]->mlpIv.h[l - 1]);
        }
        freeVector (pMlpVA[mlpfd]->theta[l]);
        freeVector (pMlpVA[mlpfd]->theta_prime[l]);
    }

    freeMatrix2 (pMlpVA[mlpfd]->w2, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]);
    freeMatrix2 (pMlpVA[mlpfd]->w2_prime, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]);
    
    freeVector (pMlpVA[mlpfd]->theta2);
    freeVector (pMlpVA[mlpfd]->theta2_prime);

    free (pMlpVA[mlpfd]->w);
    free (pMlpVA[mlpfd]->w_prime);

    free (pMlpVA[mlpfd]->theta);
    free (pMlpVA[mlpfd]->theta_prime);

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            freeMatrix2 (pMlpVA[mlpfd]->eta_w[l], pMlpVA[mlpfd]->mlpIv.m + 1);
            freeMatrix2 (pMlpVA[mlpfd]->desc_w[l], pMlpVA[mlpfd]->mlpIv.m + 1);
            freeMatrix2 (pMlpVA[mlpfd]->delta_w1[l], pMlpVA[mlpfd]->mlpIv.m + 1);
        }
        else
        {
            freeMatrix2 (pMlpVA[mlpfd]->eta_w[l], pMlpVA[mlpfd]->mlpIv.h[l - 1] + 1);
            freeMatrix2 (pMlpVA[mlpfd]->desc_w[l], pMlpVA[mlpfd]->mlpIv.h[l - 1] + 1);
            freeMatrix2 (pMlpVA[mlpfd]->delta_w1[l], pMlpVA[mlpfd]->mlpIv.h[l - 1] + 1);
        }

        if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1)
        {
            freeArray3 (pMlpVA[mlpfd]->gradientTrace[l], pMlpVA[mlpfd]->mlpIv.h[l], pMlpVA[mlpfd]->mlpIv.n);
        }
        else
        {
            freeArray3 (pMlpVA[mlpfd]->gradientTrace[l], pMlpVA[mlpfd]->mlpIv.h[l], pMlpVA[mlpfd]->mlpIv.h[l + 1]);            
        }

        freeMatrix2 (pMlpVA[mlpfd]->u[l], pMlpVA[mlpfd]->micro_max);
        freeMatrix2 (pMlpVA[mlpfd]->y[l], pMlpVA[mlpfd]->micro_max);
        freeMatrix2 (pMlpVA[mlpfd]->y_tsignal[l], pMlpVA[mlpfd]->micro_max);
        freeMatrix2 (pMlpVA[mlpfd]->delta[l], pMlpVA[mlpfd]->micro_max);
        freeArray3 (pMlpVA[mlpfd]->gradvec[l], pMlpVA[mlpfd]->micro_max, pMlpVA[mlpfd]->mlpIv.h[l]);
        freeArray3 (pMlpVA[mlpfd]->gradvec_tsignal[l], pMlpVA[mlpfd]->micro_max, pMlpVA[mlpfd]->mlpIv.h[l]);
        freeArray3 (pMlpVA[mlpfd]->grad2vec[l], pMlpVA[mlpfd]->micro_max, pMlpVA[mlpfd]->mlpIv.h[l]);    
    }

    freeMatrix2 (pMlpVA[mlpfd]->gradientTrace2, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1] + 1);
    freeMatrix2 (pMlpVA[mlpfd]->eta_w2, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1] + 1);
    freeMatrix2 (pMlpVA[mlpfd]->desc_w2, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1] + 1);
    freeMatrix2 (pMlpVA[mlpfd]->delta_w2, pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1] + 1);
    freeVector (pMlpVA[mlpfd]->u2);
    freeMatrix2 (pMlpVA[mlpfd]->y2, pMlpVA[mlpfd]->micro_max);
    freeMatrix2 (pMlpVA[mlpfd]->y2_prime, pMlpVA[mlpfd]->micro_max);
    freeMatrix2 (pMlpVA[mlpfd]->delta2, pMlpVA[mlpfd]->micro_max);
    freeMatrix2 (pMlpVA[mlpfd]->gradvec2, pMlpVA[mlpfd]->micro_max);
    freeMatrix2 (pMlpVA[mlpfd]->gradvec2_tsignal, pMlpVA[mlpfd]->micro_max);
    freeMatrix2 (pMlpVA[mlpfd]->grad2vec2, pMlpVA[mlpfd]->micro_max);

    free (pMlpVA[mlpfd]->eta_w);
    free (pMlpVA[mlpfd]->desc_w);
    free (pMlpVA[mlpfd]->delta_w1);
    free (pMlpVA[mlpfd]->u);
    free (pMlpVA[mlpfd]->y);
    free (pMlpVA[mlpfd]->y_tsignal);
    free (pMlpVA[mlpfd]->delta);
    free (pMlpVA[mlpfd]->gradvec);
    free (pMlpVA[mlpfd]->gradvec_tsignal);
    free (pMlpVA[mlpfd]->grad2vec);
    free (pMlpVA[mlpfd]->gradientTrace);
    
    mlpRegistered[mlpfd] = 0;
    nRegisteredMlps--;

    free (pMlpVA[mlpfd]);

    return 0;
}

int outputWeightsStatistics (
    int mlpfd)
{
    uint16_t k, i, i2, j, l;
    uint32_t index;
    double *vals;
    double mean, variance;
    
    if (pMlpVA[mlpfd] == NULL)
        return -1;

    printf("'Zero' weights are in interval [-%lf,%lf]\n", ZERO_PRECISION, ZERO_PRECISION);
    
    unsigned long nZeroWeights;

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        printf ("weights hidden layer (%i):\n", l);
        if (l == 0)
        {
            index = 0;
            vals = malloc (sizeof (double) * pMlpVA[mlpfd]->mlpIv.m * pMlpVA[mlpfd]->mlpIv.h[l]);
            nZeroWeights = 0;
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                {
                    vals[index] = pMlpVA[mlpfd]->w[l][k][i];
                    index++;
                    if(pMlpVA[mlpfd]->w[l][k][i] >= - ZERO_PRECISION && pMlpVA[mlpfd]->w[l][k][i] <= ZERO_PRECISION)
                        nZeroWeights++;
                }
            }
            mean = gsl_stats_mean (vals, 1, index);
            variance = gsl_stats_variance (vals, 1, index);

            printf ("The sample mean is %g\n", mean);
            printf ("The estimated variance is %g\n", variance);

            free (vals);
            index = 0;
        }
        else
        {
            vals = malloc (sizeof (double) * pMlpVA[mlpfd]->mlpIv.h[l - 1] * pMlpVA[mlpfd]->mlpIv.h[l]);
            nZeroWeights = 0;
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                {
                    vals[index] = pMlpVA[mlpfd]->w[l][i2][i];
                    index++;
                    if(pMlpVA[mlpfd]->w[l][i2][i] >= - ZERO_PRECISION && pMlpVA[mlpfd]->w[l][i2][i] <= ZERO_PRECISION)
                        nZeroWeights++;                
                }
            }
            mean = gsl_stats_mean (vals, 1, index);
            variance = gsl_stats_variance (vals, 1, index);

            printf ("The sample mean is %g\n", mean);
            printf ("The estimated variance is %g\n", variance);

            free (vals);
            index = 0;
        }
        printf ("bias hidden layer (%i):\n", l);
        vals = malloc (sizeof (double) * pMlpVA[mlpfd]->mlpIv.h[l]);
        nZeroWeights = 0;
        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
        {
            vals[index] = pMlpVA[mlpfd]->theta[l][i];
            index++;
            if(pMlpVA[mlpfd]->theta[l][i] >= - ZERO_PRECISION && pMlpVA[mlpfd]->theta[l][i] <= ZERO_PRECISION)
                nZeroWeights++;                
        }
        mean = gsl_stats_mean (vals, 1, index);
        variance = gsl_stats_variance (vals, 1, index);

        printf ("The sample mean is %g\n", mean);
        printf ("The estimated variance is %g\n", variance);

        free (vals);
        index = 0;
    }

    printf ("weights output layer:\n");
    vals = malloc (sizeof (double) * pMlpVA[mlpfd]->mlpIv.n * pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]);
    nZeroWeights = 0;
    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
        {
            vals[index] = pMlpVA[mlpfd]->w2[i][j];
            index++;
            if(pMlpVA[mlpfd]->w2[i][j] >= - ZERO_PRECISION && pMlpVA[mlpfd]->w2[i][j] <= ZERO_PRECISION)
                nZeroWeights++;                
        }
    }
    mean = gsl_stats_mean (vals, 1, index);
    variance = gsl_stats_variance (vals, 1, index);

    printf ("The sample mean is %g\n", mean);
    printf ("The estimated variance is %g\n", variance);

    free (vals);
    index = 0;

    printf ("bias output layer:\n");
    vals = malloc (sizeof (double) * pMlpVA[mlpfd]->mlpIv.n);
    nZeroWeights = 0;
    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        vals[index] = pMlpVA[mlpfd]->theta2[j];
        index++;
        if(pMlpVA[mlpfd]->theta2[j] >= - ZERO_PRECISION && pMlpVA[mlpfd]->theta2[j] <= ZERO_PRECISION)
            nZeroWeights++;
    }
    mean = gsl_stats_mean (vals, 1, index);
    variance = gsl_stats_variance (vals, 1, index);

    printf ("The sample mean is %g\n", mean);
    printf ("The estimated variance is %g\n", variance);

    free (vals);
    index = 0;

    return 0;
}


int outputWeights (
    int mlpfd)
{
    uint16_t k, i, i2, j, l;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        printf ("weights hidden layer (%i)\n", l);
        if (l == 0)
        {
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                {
                    printf ("%lf ", pMlpVA[mlpfd]->w[l][k][i]);
                }
                printf ("\n");
            }
        }
        else
        {
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                {
                    printf ("%lf ", pMlpVA[mlpfd]->w[l][i2][i]);
                }
                printf ("\n");
            }
        }
        printf ("bias hidden layer\n");
        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
        {
            printf ("%lf ", pMlpVA[mlpfd]->theta[l][i]);
        }
        printf ("\n");
    }

    printf ("weights output layer\n");
    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
        {
            printf ("%lf ", pMlpVA[mlpfd]->w2[i][j]);
        }
        printf ("\n");
    }

    printf ("bias output layer\n");
    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        printf ("%lf ", pMlpVA[mlpfd]->theta2[j]);
    }
    printf ("\n");

    return 0;
}


int setMlpParam (
    int mlpfd,
    struct mlp_param *pMlpP)
{
    if (pMlpVA[mlpfd] == NULL)
        return -1;

    if (pMlpP == NULL)
        return -2;

    pMlpVA[mlpfd]->mlpP.maxIterations = pMlpP->maxIterations;
    pMlpVA[mlpfd]->mlpP.eta1 = pMlpP->eta1;
    pMlpVA[mlpfd]->mlpP.eta2 = pMlpP->eta2;
    pMlpVA[mlpfd]->mlpP.etaNormalize = pMlpP->etaNormalize;
    pMlpVA[mlpfd]->mlpP.epsilon = pMlpP->epsilon;
    pMlpVA[mlpfd]->mlpP.beta = pMlpP->beta;
    pMlpVA[mlpfd]->mlpP.eta_pos = pMlpP->eta_pos;
    pMlpVA[mlpfd]->mlpP.eta_neg = pMlpP->eta_neg;
    pMlpVA[mlpfd]->mlpP.eta_start = pMlpP->eta_start;
    pMlpVA[mlpfd]->mlpP.eta_max = pMlpP->eta_max;
    pMlpVA[mlpfd]->mlpP.eta_min = pMlpP->eta_min;
    pMlpVA[mlpfd]->mlpP.alpha = pMlpP->alpha;
    pMlpVA[mlpfd]->mlpP.beta_max = pMlpP->beta_max;
    pMlpVA[mlpfd]->mlpP.trainingMode = pMlpP->trainingMode;
    pMlpVA[mlpfd]->mlpP.verboseOutput = pMlpP->verboseOutput;

    return 0;
}


int getMlpParam (
    int mlpfd,
    struct mlp_param *pMlpP)
{
    if (pMlpVA[mlpfd] == NULL)
        return -1;

    if (pMlpP == NULL)
        return -2;

    pMlpP->maxIterations = pMlpVA[mlpfd]->mlpP.maxIterations;
    pMlpP->eta1 = pMlpVA[mlpfd]->mlpP.eta1;
    pMlpP->eta2 = pMlpVA[mlpfd]->mlpP.eta2;
    pMlpP->etaNormalize = pMlpVA[mlpfd]->mlpP.etaNormalize;
    pMlpP->epsilon = pMlpVA[mlpfd]->mlpP.epsilon;
    pMlpP->beta = pMlpVA[mlpfd]->mlpP.beta;
    pMlpP->eta_pos = pMlpVA[mlpfd]->mlpP.eta_pos;
    pMlpP->eta_neg = pMlpVA[mlpfd]->mlpP.eta_neg;
    pMlpP->eta_start = pMlpVA[mlpfd]->mlpP.eta_start;
    pMlpP->eta_max = pMlpVA[mlpfd]->mlpP.eta_max;
    pMlpP->eta_min = pMlpVA[mlpfd]->mlpP.eta_min;
    pMlpP->alpha = pMlpVA[mlpfd]->mlpP.alpha;
    pMlpP->beta_max = pMlpVA[mlpfd]->mlpP.beta_max;
    pMlpP->trainingMode = pMlpVA[mlpfd]->mlpP.trainingMode;
    pMlpP->verboseOutput = pMlpVA[mlpfd]->mlpP.verboseOutput;

    return 0;
}


int getMlpInitValues (
    int mlpfd,
    struct mlp_init_values *pMlpInitVals)
{
    uint16_t l;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    if (pMlpInitVals == NULL)
        return -2;

    pMlpInitVals->m = pMlpVA[mlpfd]->mlpIv.m;
    pMlpInitVals->nrHiddenLayers = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers;
    for (l = 0; l < pMlpInitVals->nrHiddenLayers; l++)
        pMlpInitVals->h[l] = pMlpVA[mlpfd]->mlpIv.h[l];
    pMlpInitVals->n = pMlpVA[mlpfd]->mlpIv.n;
    pMlpInitVals->transFktTypeHidden = pMlpVA[mlpfd]->mlpIv.transFktTypeHidden;
    pMlpInitVals->transFktTypeOutput = pMlpVA[mlpfd]->mlpIv.transFktTypeOutput;
    pMlpInitVals->hasThresholdOutput = pMlpVA[mlpfd]->mlpIv.hasThresholdOutput;

    return 0;
}


int clearGradientTrace (
    int mlpfd)
{
    uint16_t k, i, i2, j;
    int32_t l;                     /*wegen for Schleife => Fehlerrückvermittlung... */

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
            {                   /*m+1 */
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    int maxJ;
                    if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1)
                        maxJ = pMlpVA[mlpfd]->mlpIv.n;
                    else
                        maxJ = pMlpVA[mlpfd]->mlpIv.h[l + 1];
                    
                    for (j = 0; j < maxJ; j++)
                        pMlpVA[mlpfd]->gradientTrace[l][i][j][k] = 0;
                }
            }
        }
        else
        {
            for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
            {                   /*h+1 */
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    int maxJ;
                    if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1)
                        maxJ = pMlpVA[mlpfd]->mlpIv.n;
                    else
                        maxJ = pMlpVA[mlpfd]->mlpIv.h[l + 1];
                    
                    for (j = 0; j < maxJ; j++)                    
                        pMlpVA[mlpfd]->gradientTrace[l][i][j][i2] = 0;
                }
            }
        }
    }

    for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
    {                           /*h+1 */
        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
            pMlpVA[mlpfd]->gradientTrace2[i][j] = 0;
    }
    
    return 0;    
}

int mlpSave (
    int mlpfd,
    const char *filename)
{
    FILE *fp;
    uint16_t i, i2, j, k, l;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    fp = fopen (filename, "w");
    if (fp == NULL)
        return -2;

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.m, sizeof (pMlpVA[mlpfd]->mlpIv.m), 1, fp) != 1)
    {
        fclose (fp);
        return -15;
    }

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.n, sizeof (pMlpVA[mlpfd]->mlpIv.n), 1, fp) != 1)
    {
        fclose (fp);
        return -17;
    }

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.nrHiddenLayers, sizeof (pMlpVA[mlpfd]->mlpIv.nrHiddenLayers), 1, fp) != 1)
    {
        fclose (fp);
        return -17;
    }

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (fwrite (&pMlpVA[mlpfd]->mlpIv.h[l], sizeof (pMlpVA[mlpfd]->mlpIv.h[l]), 1, fp) != 1)
        {
            fclose (fp);
            return -16;
        }
    }

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, sizeof (pMlpVA[mlpfd]->mlpIv.transFktTypeHidden), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, sizeof (pMlpVA[mlpfd]->mlpIv.transFktTypeOutput), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    if (fwrite (&pMlpVA[mlpfd]->a, sizeof (pMlpVA[mlpfd]->a), 1, fp) != 1)
    {
        fclose (fp);
        return -6;
    }

    if (fwrite (&pMlpVA[mlpfd]->mlpIv.hasThresholdOutput, sizeof (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    /* TODO: Remove me (kept for compatibility reasons) */

    int junksize = sizeof (pMlpVA[mlpfd]->mlpP.maxIterations) + 
                sizeof (pMlpVA[mlpfd]->mlpP.eta1) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta2) +
                sizeof (pMlpVA[mlpfd]->mlpP.epsilon) + 
                sizeof (pMlpVA[mlpfd]->mlpP.beta) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_pos) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_neg) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_start) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_max) +
                sizeof (pMlpVA[mlpfd]->mlpP.alpha) +
                sizeof (pMlpVA[mlpfd]->mlpP.trainingMode);
                                
    uint8_t junk[junksize];
    
    memset(junk, 0, sizeof(uint8_t) * junksize);
    
    if (fwrite (&junk, junksize, 1, fp) != 1)
    {
        fclose (fp);
        return -18;
    }

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if (fwrite (&pMlpVA[mlpfd]->w[l][k][i], sizeof (pMlpVA[mlpfd]->w[l][k][i]), 1, fp) != 1)
                    {
                        fclose (fp);
                        return -19;
                    }
                    
                    if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
                    {
                        if (fwrite (&pMlpVA[mlpfd]->w_prime[l][k][i], sizeof (pMlpVA[mlpfd]->w_prime[l][k][i]), 1, fp) != 1)
                        {
                            fclose (fp);
                            return -19;
                        }                        
                    }
                }
            }
        }
        else
        {
            for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if (fwrite (&pMlpVA[mlpfd]->w[l][i2][i], sizeof (pMlpVA[mlpfd]->w[l][i2][i]), 1, fp) != 1)
                    {
                        fclose (fp);
                        return -19;
                    }
                    
                    if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
                    {
                        if (fwrite (&pMlpVA[mlpfd]->w_prime[l][i2][i], sizeof (pMlpVA[mlpfd]->w_prime[l][i2][i]), 1, fp) != 1)
                        {
                            fclose (fp);
                            return -19;
                        }                        
                    }                    
                }
            }
        }

        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
        {
            if (fwrite (&pMlpVA[mlpfd]->theta[l][i], sizeof (pMlpVA[mlpfd]->theta[l][i]), 1, fp) != 1)
            {
                fclose (fp);
                return -20;
            }
            
            if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
            {
                if (fwrite (&pMlpVA[mlpfd]->theta_prime[l][i], sizeof (pMlpVA[mlpfd]->theta_prime[l][i]), 1, fp) != 1)
                {
                    fclose (fp);
                    return -20;
                }                
            }            
        }
    }

    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
    {
        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
        {
            if (fwrite (&pMlpVA[mlpfd]->w2[i][j], sizeof (pMlpVA[mlpfd]->w2[i][j]), 1, fp) != 1)
            {
                fclose (fp);
                return -21;
            }
            
            if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
            {
                if (fwrite (&pMlpVA[mlpfd]->w2_prime[i][j], sizeof (pMlpVA[mlpfd]->w2_prime[i][j]), 1, fp) != 1)
                {
                    fclose (fp);
                    return -21;
                }
            }
        }
    }

    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        if (fwrite (&pMlpVA[mlpfd]->theta2[j], sizeof (pMlpVA[mlpfd]->theta2[j]), 1, fp) != 1)
        {
            fclose (fp);
            return -22;
        }
        
        if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
        {
            if (fwrite (&pMlpVA[mlpfd]->theta2_prime[j], sizeof (pMlpVA[mlpfd]->theta2_prime[j]), 1, fp) != 1)
            {
                fclose (fp);
                return -22;
            }
        }
    }
    
    fclose (fp);
    
    return 0;
}

int mlpRestore (
    struct mlp_param *pMlpParam,
    const char *filename,
    uint32_t micro_max,
    unsigned int seed)
{
    FILE *fp;
    uint16_t i, i2, j, k, l;
    struct mlp_init_values mlpInitVals;
    double a;
    int mlpfd;

    fp = fopen (filename, "r");
    if (fp == NULL)
        return -2;

    if (fread (&mlpInitVals.m, sizeof (mlpInitVals.m), 1, fp) != 1)
    {
        fclose (fp);
        return -15;
    }

    if (fread (&mlpInitVals.n, sizeof (mlpInitVals.n), 1, fp) != 1)
    {
        fclose (fp);
        return -17;
    }

    if (fread (&mlpInitVals.nrHiddenLayers, sizeof (mlpInitVals.nrHiddenLayers), 1, fp) != 1)
    {
        fclose (fp);
        return -18;
    }

    for (l = 0; l < mlpInitVals.nrHiddenLayers; l++)
    {
        if (fread (&mlpInitVals.h[l], sizeof (mlpInitVals.h[l]), 1, fp) != 1)
        {
            fclose (fp);
            return -16;
        }
    }

    if (fread (&mlpInitVals.transFktTypeHidden, sizeof (mlpInitVals.transFktTypeHidden), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    if (fread (&mlpInitVals.transFktTypeOutput, sizeof (mlpInitVals.transFktTypeOutput), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    if (fread (&a, sizeof (a), 1, fp) != 1)
    {
        fclose (fp);
        return -6;
    }

    if (fread (&mlpInitVals.hasThresholdOutput, sizeof (mlpInitVals.hasThresholdOutput), 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    /* TODO: Remove the following line in case the output layer should have a configurable
     * threshold. Currently this is done to be able to use the old mlp saves (compatibility reasons) */
    
    mlpInitVals.hasThresholdOutput = 0;
    
    /* TODO: Remove me (compatibility reasons) */

    int junksize = sizeof (pMlpVA[mlpfd]->mlpP.maxIterations) + 
                sizeof (pMlpVA[mlpfd]->mlpP.eta1) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta2) +
                sizeof (pMlpVA[mlpfd]->mlpP.epsilon) + 
                sizeof (pMlpVA[mlpfd]->mlpP.beta) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_pos) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_neg) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_start) +
                sizeof (pMlpVA[mlpfd]->mlpP.eta_max) +
                sizeof (pMlpVA[mlpfd]->mlpP.alpha) +
                sizeof (pMlpVA[mlpfd]->mlpP.trainingMode);
                                
    uint8_t junk[junksize];
    
    if (fread (&junk, junksize, 1, fp) != 1)
    {
        fclose (fp);
        return -7;
    }

    mlpfd = initializeMlpNet (&mlpInitVals, pMlpParam, a, 0, 0, 0, micro_max, seed);
    if (mlpfd < 0)
        return -1;

    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
    {
        if (l == 0)
        {
            for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if (fread (&pMlpVA[mlpfd]->w[l][k][i], sizeof (pMlpVA[mlpfd]->w[l][k][i]), 1, fp) != 1)
                    {
                        fclose (fp);
                        return -20;
                    }
                    
                    if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
                    {
                        if (fread (&pMlpVA[mlpfd]->w_prime[l][k][i], sizeof (pMlpVA[mlpfd]->w_prime[l][k][i]), 1, fp) != 1)
                        {
                            fclose (fp);
                            return -20;
                        }
                    }
                }
            }
        }
        else
        {
            for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if (fread (&pMlpVA[mlpfd]->w[l][i2][i], sizeof (pMlpVA[mlpfd]->w[l][i2][i]), 1, fp) != 1)
                    {
                        fclose (fp);
                        return -21;
                    }
                    
                    if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
                    {
                        if (fread (&pMlpVA[mlpfd]->w_prime[l][i2][i], sizeof (pMlpVA[mlpfd]->w_prime[l][i2][i]), 1, fp) != 1)
                        {
                            fclose (fp);
                            return -21;
                        }
                    }
                }
            }
        }
        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
        {
            if (fread (&pMlpVA[mlpfd]->theta[l][i], sizeof (pMlpVA[mlpfd]->theta[l][i]), 1, fp) != 1)
            {
                fclose (fp);
                return -22;
            }
            
            if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
            {
                if (fread (&pMlpVA[mlpfd]->theta_prime[l][i], sizeof (pMlpVA[mlpfd]->theta_prime[l][i]), 1, fp) != 1)
                {
                    fclose (fp);
                    return -22;
                }
            }
        }
    }

    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
    {
        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
        {
            if (fread (&pMlpVA[mlpfd]->w2[i][j], sizeof (pMlpVA[mlpfd]->w2[i][j]), 1, fp) != 1)
            {
                fclose (fp);
                return -23;
            }
            
            if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
            {
                if (fread (&pMlpVA[mlpfd]->w2_prime[i][j], sizeof (pMlpVA[mlpfd]->w2_prime[i][j]), 1, fp) != 1)
                {
                    fclose (fp);
                    return -23;
                }
            }
        }
    }

    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
    {
        if (fread (&pMlpVA[mlpfd]->theta2[j], sizeof (pMlpVA[mlpfd]->theta2[j]), 1, fp) != 1)
        {
            fclose (fp);
            return -24;
        }
        
        if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
        {
            if (fread (&pMlpVA[mlpfd]->theta2_prime[j], sizeof (pMlpVA[mlpfd]->theta2_prime[j]), 1, fp) != 1)
            {
                fclose (fp);
                return -24;
            }
        }
    }

    fclose (fp);

    return mlpfd;
}


int mlpOutput (
    int mlpfd,
    unsigned long micro_max,
    uint32_t offset,
    struct train_data *pData)
{
    unsigned long micro;
    uint16_t k, i, i2, j, l;
    double fkt;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    if ((pData->x == NULL) || (pData->y == NULL))
        return -2;
    
    for (micro = 0; micro < micro_max; micro++)
    {
        for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
        {
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                pMlpVA[mlpfd]->u[l][0][i] = 0;
                
                if (l == 0)
                {
                    for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                        pMlpVA[mlpfd]->u[l][0][i] += pData->x[micro + offset][k] * pMlpVA[mlpfd]->w[l][k][i];
                }
                else
                {
                    for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                        pMlpVA[mlpfd]->u[l][0][i] += pMlpVA[mlpfd]->y[l - 1][0][i2] * pMlpVA[mlpfd]->w[l][i2][i];                        
                }
                
                pMlpVA[mlpfd]->u[l][0][i] -= pMlpVA[mlpfd]->theta[l][i];

                TRANSFKT (fkt, pMlpVA[mlpfd]->u[l][0][i], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);
                pMlpVA[mlpfd]->y[l][0][i] = fkt;
            }
        }

        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
        {
            pMlpVA[mlpfd]->u2[j] = 0;
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
                pMlpVA[mlpfd]->u2[j] += pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][0][i] * pMlpVA[mlpfd]->w2[i][j];                

            /* don't subtract the threshold of the neuron if the neuron has a linear transfer function */
            if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                pMlpVA[mlpfd]->u2[j] -= pMlpVA[mlpfd]->theta2[j];

            TRANSFKT (fkt, pMlpVA[mlpfd]->u2[j], pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, pMlpVA[mlpfd]->mlpP.beta);
            pData->y[micro + offset][j] = fkt;
        }
    }
    
    return 0;
}


double mlp (
    int mlpfd,
    uint32_t micro_max,
    struct train_data *pData,
    // Attention: Following arguments are only used in combination with Reinforcement Learning.
    // Currently, this is for the learning modes: RG_MODE, TDC_MODE
    double alpha,
    double alpha2,
    double gamma,
    double lambda,
    bool updateFirstLayer,
    bool updateSecondLayer,
    double *sampleAlphaDiscount,
    double gradientPrimeFactor,
    double normFactor,
    bool checkGradients,
    bool tdcOwnSummedGradient,
    bool replacingTraces)
{
    double E = 0, Emin = 999999999, Elast, Emax = 0;
    double fkt;
    uint32_t k, i, i2, j;
    int32_t l;                     /*wegen for Schleife => Fehlerrückvermittlung... */
    uint32_t micro;
    uint32_t muster;
    double S, deltaW, dS, beta;
    uint32_t numIterations = 0;
    int end = 0;
    int ouch = 0;
    double deltaWeight;

    if (pMlpVA[mlpfd] == NULL)
        return -1;

    if (micro_max > pMlpVA[mlpfd]->micro_max)
        return -2;

    if ((pData->x == NULL) || (pData->y == NULL))
        return -3;

    if (!pMlpVA[mlpfd]->mlpP.maxIterations)
        end = 1;
        
    if(pMlpVA[mlpfd]->mlpP.trainingMode == RG_MODE ||
       pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
    {
        // Must be a nonlinear transfer function in hidden layer
        if(pMlpVA[mlpfd]->mlpIv.transFktTypeHidden > 2)
            return -7;
        
        // Must be a single output neuron
        if(pMlpVA[mlpfd]->mlpIv.n != 1)
            return -8;
    }

    // Additional restrictions with TDC
    if(pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
    {
        // Must be one hidden layer
        if(pMlpVA[mlpfd]->mlpIv.nrHiddenLayers != 1)
            return -4;
        
        // Must be a single sample
        if(micro_max != 1)
            return -5;        
    }
    
    while (!end)
    {
        Elast = E;
        E = 0;
        Emax = 0;
        double Evals[micro_max];
        long z, maxz;

        if((pMlpVA[mlpfd]->mlpP.trainingMode == RG_MODE) ||
           (pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE))
            maxz = 2;
        else
            maxz = 1;
    
        for(z = maxz - 1; z >=0; z--)
        {
            for (micro = 0; micro < micro_max; micro++)
            {
                if(z && (!pData->hasXPrime[micro]))
                    continue;
                    
                /*Schritt 2: Berechnung der Netzausgabe y2 (Vorwärtsphase) */
                for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
                {
                    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    {
                        pMlpVA[mlpfd]->u[l][micro][i] = 0;
                        if(l == 0)
                        {
                            for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                            {
                                if(!z)
                                    pMlpVA[mlpfd]->u[l][micro][i] += pData->x[micro][k] * pMlpVA[mlpfd]->w[l][k][i];
                                else
                                    pMlpVA[mlpfd]->u[l][micro][i] += pData->x_prime[micro][k] * pMlpVA[mlpfd]->w[l][k][i];
                            }
                        }
                        else
                        {
                            for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                            {
                                if(!z)
                                    pMlpVA[mlpfd]->u[l][micro][i] += pMlpVA[mlpfd]->y[l - 1][micro][i2] * pMlpVA[mlpfd]->w[l][i2][i];
                                else
                                    pMlpVA[mlpfd]->u[l][micro][i] += pMlpVA[mlpfd]->y_tsignal[l - 1][micro][i2] * pMlpVA[mlpfd]->w[l][i2][i];
                            }                                
                        }
                        
                        pMlpVA[mlpfd]->u[l][micro][i] -= pMlpVA[mlpfd]->theta[l][i];
                        
                        TRANSFKT (fkt, pMlpVA[mlpfd]->u[l][micro][i], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);
                        if(!z)
                            pMlpVA[mlpfd]->y[l][micro][i] = fkt;
                        else
                            pMlpVA[mlpfd]->y_tsignal[l][micro][i] = fkt;             
                    } /* for i */
                } /* for l */

                /* from output layer to last hidden layer */

                double E_single_sample = 0;
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    pMlpVA[mlpfd]->u2[j] = 0;
                    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1]; i++)
                    {
                        if(!z)
                            pMlpVA[mlpfd]->u2[j] += pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][micro][i] * pMlpVA[mlpfd]->w2[i][j];
                        else
                            pMlpVA[mlpfd]->u2[j] += pMlpVA[mlpfd]->y_tsignal[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][micro][i] * pMlpVA[mlpfd]->w2[i][j];
                    }

                    /* don't subtract the threshold of the neuron if the neuron has a linear transfer function */
                    if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        pMlpVA[mlpfd]->u2[j] -= pMlpVA[mlpfd]->theta2[j];
                    
                    TRANSFKT (fkt, pMlpVA[mlpfd]->u2[j], pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, pMlpVA[mlpfd]->mlpP.beta);

                    if ((pMlpVA[mlpfd]->mlpP.trainingMode == RG_MODE || pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE) && z)
                    {
                        pMlpVA[mlpfd]->y2_prime[micro][j] = fkt;

                        /*Schritt 3: Bestimmung des Fehlers am Netzausgang */
                        TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y2_prime[micro][j], pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, pMlpVA[mlpfd]->mlpP.beta);
                    }
                    else
                    {
                        pMlpVA[mlpfd]->y2[micro][j] = fkt;
                        
                        /*Schritt 3: Bestimmung des Fehlers am Netzausgang */
                        TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y2[micro][j], pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, pMlpVA[mlpfd]->mlpP.beta);
                    }
                    
                    /* gradient vector for bias output layer
                    * iff multiplied by pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][micro][i] then it is
                    * the gradient vector for the weights output layer */
                    if(!z)
                    {
                         double fkt2;
                        TRANSFKT_DERIVATIVE_2 (fkt2, pMlpVA[mlpfd]->y2[micro][j], pMlpVA[mlpfd]->mlpIv.transFktTypeOutput, pMlpVA[mlpfd]->mlpP.beta);

                        pMlpVA[mlpfd]->grad2vec2[micro][j] = fkt2;
                        pMlpVA[mlpfd]->gradvec2[micro][j] = fkt;
                        pMlpVA[mlpfd]->delta2[micro][j] = (pData->y[micro][j] - pMlpVA[mlpfd]->y2[micro][j]) * fkt;

                        double diff = 0;
                        
                        if((pMlpVA[mlpfd]->mlpP.trainingMode != RG_MODE && 
                            pMlpVA[mlpfd]->mlpP.trainingMode != TDC_MODE))
                            diff = (pData->y[micro][j] - pMlpVA[mlpfd]->y2[micro][j]);
                        else
                        {
                            if(!pData->hasXPrime[micro])
                                diff = (pData->reward[micro] - pMlpVA[mlpfd]->y2[micro][j]);
                            else
                                diff = (pData->reward[micro] + gamma * normFactor * pMlpVA[mlpfd]->y2_prime[micro][j] + pData->y2[micro][j] - pMlpVA[mlpfd]->y2[micro][j]);                        
                        }
                        
                        E_single_sample += pow (diff, 2.0);
                        if(E_single_sample > Emax)
                            Emax = E_single_sample;
                        Evals[micro] = E_single_sample;
                    }
                    else
                        pMlpVA[mlpfd]->gradvec2_tsignal[micro][j] = fkt;
                } /* for j */
                
                if(!z)
                    E += E_single_sample;

                /*Schritt 4: Fehlerrückvermittlung (Rückwärtsphase) */

                /* from output layer to last hidden layer l */
                l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if(!z)
                    {
                        pMlpVA[mlpfd]->delta[l][micro][i] = 0;
                        TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y[l][micro][i], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);
                    }
                    else
                    {
                        TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y_tsignal[l][micro][i], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);                    
                    }
                    
                    double fkt2;
                    TRANSFKT_DERIVATIVE_2 (fkt2, pMlpVA[mlpfd]->y[l][micro][i], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);
                    
                    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                    {
                        if(!z)
                        {
                            pMlpVA[mlpfd]->delta[l][micro][i] += fkt * pMlpVA[mlpfd]->w2[i][j] * pMlpVA[mlpfd]->delta2[micro][j];
                            pMlpVA[mlpfd]->gradvec[l][micro][i][j] = fkt * pMlpVA[mlpfd]->w2[i][j] * pMlpVA[mlpfd]->gradvec2[micro][j];
                            
                            // TODO: Verify the following line
                            pMlpVA[mlpfd]->grad2vec[l][micro][i][j] = pMlpVA[mlpfd]->w2[i][j] * (fkt2 * pMlpVA[mlpfd]->gradvec2[micro][j] + 2.0 * pMlpVA[mlpfd]->grad2vec2[micro][j] * pow(fkt, 2.0) * pMlpVA[mlpfd]->w2[i][j]);
                        }
                        else
                            pMlpVA[mlpfd]->gradvec_tsignal[l][micro][i][j] = fkt * pMlpVA[mlpfd]->w2[i][j] * pMlpVA[mlpfd]->gradvec2_tsignal[micro][j];
                    }
                } /* for i */

                /* from hidden layer l to hidden layer l-1 */
                for (l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 2; l >= 0; l--)
                {
                    for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l]; i2++)
                    {
                        if(!z)
                        {
                            pMlpVA[mlpfd]->delta[l][micro][i2] = 0;
                            TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y[l][micro][i2], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);
                        }
                        else
                        {
                            TRANSFKT_DERIVATIVE (fkt, pMlpVA[mlpfd]->y_tsignal[l][micro][i2], pMlpVA[mlpfd]->mlpIv.transFktTypeHidden, pMlpVA[mlpfd]->mlpP.beta);                        
                        }

                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l + 1]; i++)
                        {
                            int maxJ;
                            if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 2)
                            {
                                // Naechste Schicht ist Ausgabeschicht
                                maxJ = pMlpVA[mlpfd]->mlpIv.n;
                            }
                            else
                            {
                                // Naechste Schicht ist naechste Zwischenschicht
                                maxJ = pMlpVA[mlpfd]->mlpIv.h[l + 2];
                            }
                            
                            if(!z)
                            {
                                pMlpVA[mlpfd]->delta[l][micro][i2] += fkt * pMlpVA[mlpfd]->w[l + 1][i2][i] * pMlpVA[mlpfd]->delta[l + 1][micro][i];
                                
                                pMlpVA[mlpfd]->gradvec[l][micro][i2][i] = 0;                                
                                for (j = 0; j < maxJ; j++)
                                    pMlpVA[mlpfd]->gradvec[l][micro][i2][i] += fkt * pMlpVA[mlpfd]->w[l + 1][i2][i] * pMlpVA[mlpfd]->gradvec[l + 1][micro][i][j];
                            }
                            else
                            {
                                pMlpVA[mlpfd]->gradvec_tsignal[l][micro][i2][i] = 0;                                
                                for (j = 0; j < maxJ; j++)
                                    pMlpVA[mlpfd]->gradvec_tsignal[l][micro][i2][i] += fkt * pMlpVA[mlpfd]->w[l + 1][i2][i] * pMlpVA[mlpfd]->gradvec_tsignal[l + 1][micro][i][j];
                            }
                        }
                    }
                } /* for l */
            
                if(checkGradients)
                {
                    if(pMlpVA[mlpfd]->mlpP.trainingMode != ONLINE_MODE)
                    {
                        printf("### checkGradients: Must be run in online mode, check trainingMode in configuration file\n");
                        return -4;
                    }
                    
                    /* Compares the gradients (gradvec and gradvec2) with the delta values (delta and delta2)
                     * This is only for debugging / verifying the internal works of this function.
                     * Normally checkGradients shall be false */
                    
                    for (l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l >= 0; l--)
                    {
                        if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1)
                        {
                            // Output layer
                            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                            {
                                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                                {
                                    double val = pMlpVA[mlpfd]->gradvec2[micro][j] * (pData->y[micro][j] - pMlpVA[mlpfd]->y2[micro][j]);
                                    
                                    if((float) val != (float) pMlpVA[mlpfd]->delta2[micro][j])
                                    {
                                        printf("### checkGradients: output layer, (%lf) != (%lf)\n", val, pMlpVA[mlpfd]->delta2[micro][j]);
                                        return -4;
                                    }
                                }
                            }
                            
                            // Last hidden layer
                            
                            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                            {
                                double sum = 0;
                            
                                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                                    sum += pMlpVA[mlpfd]->gradvec[l][micro][i][j] * (pData->y[micro][j] - pMlpVA[mlpfd]->y2[micro][j]);
                            
                                if((float) sum != (float)pMlpVA[mlpfd]->delta[l][micro][i])
                                {
                                    printf("### checkGradients: hidden layer (%i), i (%i), (%lf) != (%lf)\n", l, i, sum, pMlpVA[mlpfd]->delta[l][micro][i]);
                                    return -4;
                                }
                            }
                        }
                        else
                        {
                            // Hidden layer before last hidden layer

                            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                            {
                                double sum = 0;
                            
                                for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l + 1]; i2++)
                                {
                                    for(j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                                        sum += pMlpVA[mlpfd]->gradvec[l][micro][i][i2] * (pData->y[micro][j] - pMlpVA[mlpfd]->y2[micro][j]);
                                }
                            
                                if((float) sum != (float)pMlpVA[mlpfd]->delta[l][micro][i])
                                {
                                    printf("### checkGradients: hidden layer (%i), i (%i), (%lf) != (%lf)\n", l, i, sum, pMlpVA[mlpfd]->delta[l][micro][i]);
                                    return -4;
                                }
                            }
                        }
                    }
                }
                    
                /*Schritt 5: Lernen */
                if (pMlpVA[mlpfd]->mlpP.trainingMode == ONLINE_MODE)
                {
                    /*Error Backpropagation: Online Mode */

                    /*Adaptiere Gewichte nach Praesentation eines einzelnen Musters micro */
                    /*Adaptiere Schwellwerte nach Praesentation eines einzelnen Musters micro */

                    /*Adaptiere Gewichte der Ausgangsschicht */
                    l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
                    for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    {
                        for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                        {
                            if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                            {
                                deltaWeight = pMlpVA[mlpfd]->w2[i][j] + pMlpVA[mlpfd]->mlpP.eta2 * pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][micro][i] * pMlpVA[mlpfd]->delta2[micro][j];
                                if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]))
                                    deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]) * sign (deltaWeight);
                                pMlpVA[mlpfd]->w2[i][j] = deltaWeight;
                            }
                            else
                            {

                                /* neurons with linear activation function don't have a threshold */
                                if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                                {
                                    /*Adaptiere Schwellwerte der Ausgangsschicht */
                                    deltaWeight = pMlpVA[mlpfd]->theta2[j] - (pMlpVA[mlpfd]->mlpP.eta2 * pMlpVA[mlpfd]->delta2[micro][j]);
                                    if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]))
                                        deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]) * sign (deltaWeight);
                                    pMlpVA[mlpfd]->theta2[j] = deltaWeight;
                                }
                            }
                        }
                    }

                    /*Adaptiere Gewichte der Zwischenschichten */
                    for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
                    {
                        if (l == 0)
                        {
                            for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
                            {
                                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                                {
                                    if (k < pMlpVA[mlpfd]->mlpIv.m)
                                    {
                                        deltaWeight = pMlpVA[mlpfd]->w[l][k][i] + pMlpVA[mlpfd]->mlpP.eta1 * pData->x[micro][k] * pMlpVA[mlpfd]->delta[l][micro][i];
                                        if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.m))
                                            deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.m) * sign (deltaWeight);
                                        pMlpVA[mlpfd]->w[l][k][i] = deltaWeight;
                                    }
                                    else
                                    {
                                        /*Adaptiere Schwellwerte der Zwischenschichten */
                                        deltaWeight = pMlpVA[mlpfd]->theta[l][i] - (pMlpVA[mlpfd]->mlpP.eta1 * pMlpVA[mlpfd]->delta[l][micro][i]);
                                        if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.m))
                                            deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.m) * sign (deltaWeight);
                                        pMlpVA[mlpfd]->theta[l][i] = deltaWeight;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                            {
                                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                                {
                                    if (i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                                    {
                                        deltaWeight = pMlpVA[mlpfd]->w[l][i2][i] + pMlpVA[mlpfd]->mlpP.eta1 * pMlpVA[mlpfd]->y[l - 1][micro][i2] * pMlpVA[mlpfd]->delta[l][micro][i];
                                        if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]))
                                            deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]) * sign (deltaWeight);
                                        pMlpVA[mlpfd]->w[l][i2][i] = deltaWeight;
                                    }
                                    else
                                    {
                                        /*Adaptiere Schwellwerte der Zwischenschichten */
                                        deltaWeight = pMlpVA[mlpfd]->theta[l][i] - (pMlpVA[mlpfd]->mlpP.eta1 * pMlpVA[mlpfd]->delta[l][micro][i]);
                                        if (fabs (deltaWeight) > (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]))
                                            deltaWeight = (MAX_WEIGHT_ONLINE_MODE / pMlpVA[mlpfd]->mlpIv.h[l]) * sign (deltaWeight);
                                        pMlpVA[mlpfd]->theta[l][i] = deltaWeight;
                                    }
                                }
                            }
                        }
                    }
                } /* ONLINE_MODE */
            } /* for micro */ /*Schritt 6: Ende Epoche, naechstes Muster, falls noch weitere vorhanden */
        } /* for z */ 

        if (pMlpVA[mlpfd]->mlpP.trainingMode == BATCH_MODE)
        {
            /*Error Backpropagation: Batch Mode */

            /*Adaptiere Gewichte nach Praesentation aller Muster micro */

            /*Adaptiere Gewichte der Ausgangsschicht */
            /*Adaptiere Schwellwerte nach Praesentation aller Muster micro */

            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
            
            double eta = pMlpVA[mlpfd]->mlpP.eta2;
            if(pMlpVA[mlpfd]->mlpP.etaNormalize)
                eta /= (double) micro_max;
            
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                    {
                        CALC_GRADIENT_WEIGHTS_OUTPUT (S);
                        deltaW = -eta * S;                                                        
                        pMlpVA[mlpfd]->w2[i][j] = pMlpVA[mlpfd]->w2[i][j] + deltaW;
                    }
                    else
                    {
                        /* neurons with linear activation function don't have a threshold */
                        if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        {
                            /*Adaptiere Schwellwerte der Ausgangsschicht */
                            CALC_GRADIENT_BIAS_OUTPUT (S);
                            deltaW = -eta * S;                                                        
                            pMlpVA[mlpfd]->theta2[j] = pMlpVA[mlpfd]->theta2[j] + deltaW;
                        }
                    }
                }
            }

            eta = pMlpVA[mlpfd]->mlpP.eta1;
            if(pMlpVA[mlpfd]->mlpP.etaNormalize)
                eta /= (double) micro_max;

            /*Adaptiere Gewichte der Zwischenschichten */
            for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
            {
                if (l == 0)
                {
                    for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (k < pMlpVA[mlpfd]->mlpIv.m)
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER (S);
                                deltaW = -eta * S;
                                pMlpVA[mlpfd]->w[l][k][i] = pMlpVA[mlpfd]->w[l][k][i] + deltaW;
                            }
                            else
                            {
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                deltaW = -eta * S;
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + deltaW;
                            }
                        }
                    }
                }
                else
                {
                    for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN (S);
                                deltaW = -eta * S;                                
                                pMlpVA[mlpfd]->w[l][i2][i] = pMlpVA[mlpfd]->w[l][i2][i] + deltaW;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                deltaW = -eta * S;                                
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + deltaW;
                            }
                        }
                    }
                }
            }
        } /* BATCH_MODE */            
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == RG_MODE)
        {
            double gradient_prime, gradient;

            if(updateSecondLayer)
            {
                l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
                for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                    {
                        /* output neurons with linear activation function don't have a threshold */
                        if (i < pMlpVA[mlpfd]->mlpIv.h[l] || pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        {
                            S = 0;                        
                            for (muster = 0; muster < micro_max; muster++)
                            {
                                double diff;
                                if(!pData->hasXPrime[muster])
                                {
                                    diff = (pData->reward[muster] - pMlpVA[mlpfd]->y2[muster][j]);
                                    gradient_prime = 0;
                                }
                                else
                                {
                                    diff = (pData->reward[muster] + gamma * normFactor * pMlpVA[mlpfd]->y2_prime[muster][j] + pData->y2[muster][j] - pMlpVA[mlpfd]->y2[muster][j]);

                                    double tmp = -1.0;
                                    if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                                        tmp = pMlpVA[mlpfd]->y_tsignal[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][muster][i];
                                                                        
                                    gradient_prime = pMlpVA[mlpfd]->gradvec2_tsignal[muster][j] * tmp;
                                }
                                
                                if(sampleAlphaDiscount != NULL)
                                    diff *= sampleAlphaDiscount[muster];
                                
                                double tmp = -1.0;
                                if(i < pMlpVA[mlpfd]->mlpIv.h[l])
                                    tmp = pMlpVA[mlpfd]->y[pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1][muster][i];

                                gradient = pMlpVA[mlpfd]->gradvec2[muster][j] * tmp;

                                if(replacingTraces)
                                {
                                    // Replacing traces
                                    if(gradient == 0)
                                        pMlpVA[mlpfd]->gradientTrace2[i][j] = gamma * lambda * pMlpVA[mlpfd]->gradientTrace2[i][j];
                                    else
                                        pMlpVA[mlpfd]->gradientTrace2[i][j] = gradient;                                    
                                }
                                else
                                {
                                    // Accumulating traces
                                    pMlpVA[mlpfd]->gradientTrace2[i][j] = gamma * lambda * pMlpVA[mlpfd]->gradientTrace2[i][j] + gradient;
                                }

                                S += (pMlpVA[mlpfd]->gradientTrace2[i][j] - gradientPrimeFactor * gamma * gradient_prime) * diff;

                                /* Clear gradient traces upon observing a terminal state */
                                if(!pData->hasXPrime[muster])
                                    pMlpVA[mlpfd]->gradientTrace2[i][j] = 0;
                            } /* for muster */
                            
                            deltaW = alpha * S;
                            
                            if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                                pMlpVA[mlpfd]->w2[i][j] += deltaW;
                            else
                                pMlpVA[mlpfd]->theta2[j] += deltaW;
                        }
                    }
                }
            }

            if(updateFirstLayer)
            {
                /*Adaptiere Gewichte der Zwischenschichten */
                for (l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1; l >= 0; l--)
                {
                    int maxK;
                    if(l == 0)
                        maxK = pMlpVA[mlpfd]->mlpIv.m;
                    else
                        maxK = pMlpVA[mlpfd]->mlpIv.h[l - 1];
                    
                    for (k = 0; k <= maxK; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            S = 0;
                            for (muster = 0; muster < micro_max; muster++)
                            {
                                int maxJ;
                                if(l == pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1)
                                    maxJ = pMlpVA[mlpfd]->mlpIv.n;
                                else
                                    maxJ = pMlpVA[mlpfd]->mlpIv.h[l + 1];
                                
                                for (j = 0; j < maxJ; j++)
                                {
                                    double diff;
                                    if(!pData->hasXPrime[muster])
                                    {
                                        diff = (pData->reward[muster] - pMlpVA[mlpfd]->y2[muster][0]);
                                        gradient_prime = 0;
                                    }
                                    else
                                    {
                                        diff = (pData->reward[muster] + gamma * normFactor * pMlpVA[mlpfd]->y2_prime[muster][0] + pData->y2[muster][0] - pMlpVA[mlpfd]->y2[muster][0]);

                                        double tmp = -1.0;
                                        if(l == 0)
                                        {
                                            if(k < pMlpVA[mlpfd]->mlpIv.m)
                                                tmp = pData->x_prime[muster][k];
                                        }
                                        else
                                        {
                                            if (k < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                                                tmp = pMlpVA[mlpfd]->y_tsignal[l - 1][muster][k];                                                
                                        }
                                        
                                        gradient_prime = pMlpVA[mlpfd]->gradvec_tsignal[l][muster][i][j] * tmp;
                                    }

                                    if(sampleAlphaDiscount != NULL)
                                        diff *= sampleAlphaDiscount[muster];

                                    double tmp = -1.0;
                                    if(l == 0)
                                    {
                                        if(k < pMlpVA[mlpfd]->mlpIv.m)
                                            tmp = pData->x[muster][k];
                                    }
                                    else
                                    {
                                        if (k < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                                            tmp = pMlpVA[mlpfd]->y[l - 1][muster][k];
                                    }
                                    
                                    gradient = pMlpVA[mlpfd]->gradvec[l][muster][i][j] * tmp;
                                    
                                    if(replacingTraces)
                                    {
                                        // Replacing traces
                                        if(gradient == 0)
                                            pMlpVA[mlpfd]->gradientTrace[l][i][j][k] = gamma * lambda * pMlpVA[mlpfd]->gradientTrace[l][i][j][k];
                                        else
                                            pMlpVA[mlpfd]->gradientTrace[l][i][j][k] = gradient;                                        
                                    }
                                    else
                                    {
                                        // Accumulating traces
                                        pMlpVA[mlpfd]->gradientTrace[l][i][j][k] = gamma * lambda * pMlpVA[mlpfd]->gradientTrace[l][i][j][k] + gradient;
                                    }
                                                                        
                                    S += (pMlpVA[mlpfd]->gradientTrace[l][i][j][k] - gradientPrimeFactor * gamma * gradient_prime) * diff;
                                    
                                    /* Clear gradient traces upon observing a terminal state */
                                    if(!pData->hasXPrime[muster])
                                        pMlpVA[mlpfd]->gradientTrace[l][i][j][k] = 0;
                                } /* for j */
                            } /* for muster */
                            
                            deltaW = alpha2 * S;
                            
                            if((l == 0 && k < pMlpVA[mlpfd]->mlpIv.m) || (l > 0 && k < pMlpVA[mlpfd]->mlpIv.h[l - 1]))
                                pMlpVA[mlpfd]->w[l][k][i] += deltaW;
                            else
                                pMlpVA[mlpfd]->theta[l][i] += deltaW;
                        } /* for i */
                    } /* for k */
                } /* for l */
            } /* updateFirstLayer */
        } /* RG_MODE */
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == TDC_MODE)
        {
            muster = 0; /* We assume that there is only one state transition (sample) */
            
            // Calculate \theta^T w

            double summed_W_Gradient_outputLayer = 0;
            double summed_W_Gradient_outputLayerBias = 0;
            double summed_W_Gradient_hiddenLayer = 0;
            double summed_W_Gradient_hiddenLayerBias = 0;
            if(tdcOwnSummedGradient)
            {
                l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    summed_W_Gradient_outputLayer += pMlpVA[mlpfd]->gradvec2[muster][0] * pMlpVA[mlpfd]->y[l][muster][i] * pMlpVA[mlpfd]->w2_prime[i][0];

                
                if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                    summed_W_Gradient_outputLayerBias += pMlpVA[mlpfd]->gradvec2[muster][0] * -1.0 * pMlpVA[mlpfd]->theta2_prime[0];

                
                l = 0;
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                        summed_W_Gradient_hiddenLayer += pMlpVA[mlpfd]->gradvec[l][muster][i][0] * pData->x[muster][k] * pMlpVA[mlpfd]->w_prime[l][k][i];
                    
                    summed_W_Gradient_hiddenLayerBias += pMlpVA[mlpfd]->gradvec[l][muster][i][0] * -1.0 * pMlpVA[mlpfd]->theta_prime[l][i];
                }                
            }
            else
            {
                double summed_W_Gradient = 0;
                
                l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    summed_W_Gradient += pMlpVA[mlpfd]->gradvec2[muster][0] * pMlpVA[mlpfd]->y[l][muster][i] * pMlpVA[mlpfd]->w2_prime[i][0];
                
                if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                    summed_W_Gradient += pMlpVA[mlpfd]->gradvec2[muster][0] * -1.0 * pMlpVA[mlpfd]->theta2_prime[0];

                l = 0;
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                        summed_W_Gradient += pMlpVA[mlpfd]->gradvec[l][muster][i][0] * pData->x[muster][k] * pMlpVA[mlpfd]->w_prime[l][k][i];
                    
                    summed_W_Gradient += pMlpVA[mlpfd]->gradvec[l][muster][i][0] * -1.0 * pMlpVA[mlpfd]->theta_prime[l][i];
                }
                
                summed_W_Gradient_outputLayer = summed_W_Gradient_outputLayerBias = summed_W_Gradient_hiddenLayer = summed_W_Gradient_hiddenLayerBias = summed_W_Gradient;
            }

            double diff;
            if(!pData->hasXPrime[muster])
                diff = (pData->reward[muster] - pMlpVA[mlpfd]->y2[muster][0]);
            else
                diff = (pData->reward[muster] + gamma * normFactor * pMlpVA[mlpfd]->y2_prime[muster][0] + pData->y2[muster][0] - pMlpVA[mlpfd]->y2[muster][0]);
            
            // Adapt the first weights theta

            // Output layer
            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;            
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                {
                    double gradient = pMlpVA[mlpfd]->gradvec2[muster][0] * pMlpVA[mlpfd]->y[l][muster][i];                    
                    double gradient2 = pMlpVA[mlpfd]->grad2vec2[muster][0] * pow(pMlpVA[mlpfd]->y[l][muster][i], 2.0);
                    double h = (diff - summed_W_Gradient_outputLayer) * gradient2 * pMlpVA[mlpfd]->w2_prime[i][0];

                    double gradient_prime = 0;
                    if(pData->hasXPrime[muster])
                        gradient_prime = pMlpVA[mlpfd]->gradvec2_tsignal[muster][0] * pMlpVA[mlpfd]->y_tsignal[l][muster][i];                        
                    
                    S = diff * gradient - gamma * gradient_prime * summed_W_Gradient_outputLayer - h;

                    deltaW = alpha * S;
                    pMlpVA[mlpfd]->w2[i][0] += deltaW;
                }
                else if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                {
                    /*Adaptiere Schwellwerte der Ausgangsschicht */
                    double gradient = pMlpVA[mlpfd]->gradvec2[muster][0] * -1.0;
                    double gradient2 = pMlpVA[mlpfd]->grad2vec2[muster][0];
                    double h = (diff - summed_W_Gradient_outputLayerBias) * gradient2 * pMlpVA[mlpfd]->theta2_prime[0];
                                            
                    double gradient_prime = 0;
                    if(pData->hasXPrime[muster])
                        gradient_prime = pMlpVA[mlpfd]->gradvec2_tsignal[muster][0] * -1.0;
                    
                    S = diff * gradient - gamma * gradient_prime * summed_W_Gradient_outputLayerBias - h;

                    deltaW = alpha * S;
                    pMlpVA[mlpfd]->theta2[0] += deltaW;
                }
            }

            // Hidden layer
            l = 0;
            for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    if (k < pMlpVA[mlpfd]->mlpIv.m)
                    {
                        double gradient = pMlpVA[mlpfd]->gradvec[l][muster][i][0] * pData->x[muster][k];                        
                        double gradient2 = pMlpVA[mlpfd]->grad2vec[l][muster][i][0] * pow(pData->x[muster][k], 2.0);
                        double h = (diff - summed_W_Gradient_hiddenLayer) * gradient2 * pMlpVA[mlpfd]->w_prime[l][k][i];
                        
                        double gradient_prime = 0;
                        if(pData->hasXPrime[muster])
                            gradient_prime = pMlpVA[mlpfd]->gradvec_tsignal[l][muster][i][0] * pData->x_prime[muster][k];
                        
                        S = diff * gradient - gamma * gradient_prime * summed_W_Gradient_hiddenLayer - h;

                        deltaW = alpha * S;
                        pMlpVA[mlpfd]->w[l][k][i] += deltaW;
                    }
                    else
                    {
                        /*Adaptiere Schwellwerte der Zwischenschichten */
                        double gradient = pMlpVA[mlpfd]->gradvec[l][muster][i][0] * -1.0;                        
                        double gradient2 = pMlpVA[mlpfd]->grad2vec[l][muster][i][0];
                        double h = (diff - summed_W_Gradient_hiddenLayerBias) * gradient2 * pMlpVA[mlpfd]->theta_prime[l][i];

                        double gradient_prime = 0;
                        if(pData->hasXPrime[muster])
                            gradient_prime = pMlpVA[mlpfd]->gradvec_tsignal[l][muster][i][0] * -1.0;

                        S = diff * gradient - gamma * gradient_prime * summed_W_Gradient_hiddenLayerBias - h;

                        deltaW = alpha * S;
                        pMlpVA[mlpfd]->theta[l][i] += deltaW;
                    }
                } /* for i */
            } /* for k */

            // Update the second weights

            // Output layer
            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;            
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                double gradient = pMlpVA[mlpfd]->gradvec2[muster][0];
                if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                {
                    gradient *= pMlpVA[mlpfd]->y[l][muster][i];
                    pMlpVA[mlpfd]->w2_prime[i][0] += alpha2 * (diff - summed_W_Gradient_outputLayer) * gradient;                    
                }
                else if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                {
                    gradient *= -1.0;                                            
                    pMlpVA[mlpfd]->theta2_prime[0] += alpha2 * (diff - summed_W_Gradient_outputLayerBias) * gradient;
                }
            }
            
            // Hidden layer
            l = 0;
            for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
            {
                for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                {
                    double gradient = pMlpVA[mlpfd]->gradvec[l][muster][i][0];
                    if (k < pMlpVA[mlpfd]->mlpIv.m)
                    {
                        gradient *= pData->x[muster][k];                        
                        pMlpVA[mlpfd]->w_prime[l][k][i] += alpha2 * (diff - summed_W_Gradient_hiddenLayer) * gradient;
                    }
                    else
                    {
                        gradient *= -1.0;                        
                        pMlpVA[mlpfd]->theta_prime[l][i] += alpha2 * (diff - summed_W_Gradient_hiddenLayerBias) * gradient;
                    }
                }
            }
        } /* TDC_MODE */
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == MOMENTUMTERM_MODE)
        {
            /*Error Backpropagation mit Momentumterm: Batch Mode */

            /*Adaptiere Gewichte nach Praesentation aller Muster micro */
            /*Adaptiere Schwellwerte nach Praesentation aller Muster micro */

            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                    {
                        CALC_GRADIENT_WEIGHTS_OUTPUT (S);
                        deltaW = -pMlpVA[mlpfd]->mlpP.eta2 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->w2[i][j] += deltaW;
                        pMlpVA[mlpfd]->delta_w2[i][j] = deltaW;
                    }
                    else
                    {
                        /* neurons with linear activation function don't have a threshold */
                        if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        {
                            /*Adaptiere Schwellwerte der Ausgangsschicht */
                            CALC_GRADIENT_BIAS_OUTPUT (S);
                            deltaW = -pMlpVA[mlpfd]->mlpP.eta2 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w2[i][j];
                            pMlpVA[mlpfd]->theta2[j] += deltaW;
                            pMlpVA[mlpfd]->delta_w2[i][j] = deltaW;
                        }
                    }
                }
            }

            /*Adaptiere Gewichte der Zwischenschichten */
            for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
            {
                if (l == 0)
                {
                    for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (k < pMlpVA[mlpfd]->mlpIv.m)
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER (S);
                                deltaW = -pMlpVA[mlpfd]->mlpP.eta1 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                pMlpVA[mlpfd]->w[l][k][i] += deltaW;
                                pMlpVA[mlpfd]->delta_w1[l][k][i] = deltaW;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                deltaW = -pMlpVA[mlpfd]->mlpP.eta1 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                pMlpVA[mlpfd]->theta[l][i] += deltaW;
                                pMlpVA[mlpfd]->delta_w1[l][k][i] = deltaW;
                            }
                        }
                    }
                }
                else
                {
                    for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN (S);
                                deltaW = -pMlpVA[mlpfd]->mlpP.eta1 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                pMlpVA[mlpfd]->w[l][i2][i] += deltaW;
                                pMlpVA[mlpfd]->delta_w1[l][i2][i] = deltaW;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                deltaW = -pMlpVA[mlpfd]->mlpP.eta1 * S + pMlpVA[mlpfd]->mlpP.alpha * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                pMlpVA[mlpfd]->theta[l][i] += deltaW;
                                pMlpVA[mlpfd]->delta_w1[l][i2][i] = deltaW;
                            }
                        }
                    }
                }
            }
        }
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == QUICKPROP_MODE)
        {
            /*Quickpropagation Mode */

            /*Adaptiere Gewichte nach Praesentation aller Muster micro */
            /*Adaptiere Schwellwerte nach Praesentation aller Muster micro */

            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                    {
                        CALC_GRADIENT_WEIGHTS_OUTPUT (S);
                        if ((pMlpVA[mlpfd]->delta_w2[i][j] == 0) || (S > pMlpVA[mlpfd]->desc_w2[i][j]))
                        {
                            /*Standard Error Backpropagation Lernregel */
                            deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta2 * S;
                        }
                        else
                        {
                            /*                            printf("Yes, Quickprop like...\n"); */
                            if (S > pMlpVA[mlpfd]->desc_w2[i][j])
                            {
                                printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w2[i][j]);
                                return -2;
                            }
                            beta = S / (pMlpVA[mlpfd]->desc_w2[i][j] - S);
                            if (fabs (beta * pMlpVA[mlpfd]->delta_w2[i][j]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w2[i][j]))
                                deltaW = beta * pMlpVA[mlpfd]->delta_w2[i][j];
                            else
                            {
                                if (beta < 0)
                                    deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w2[i][j];
                                else
                                    deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w2[i][j];
                            }
                        }
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                        pMlpVA[mlpfd]->delta_w2[i][j] = deltaW;
                        pMlpVA[mlpfd]->w2[i][j] = pMlpVA[mlpfd]->w2[i][j] + deltaW;
                    }
                    else
                    {
                        /* neurons with linear activation function don't have a threshold */
                        if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        {

                            /*Adaptiere Schwellwerte der Ausgangsschicht */
                            CALC_GRADIENT_BIAS_OUTPUT (S);
                            if ((pMlpVA[mlpfd]->delta_w2[i][j] == 0) || (S > pMlpVA[mlpfd]->desc_w2[i][j]))
                            {
                                /*Standard Error Backpropagation Lernregel */
                                deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta2 * S;
                            }
                            else
                            {
                                if (S > pMlpVA[mlpfd]->desc_w2[i][j])
                                {
                                    printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w2[i][j]);
                                    return -2;
                                }
                                beta = S / (pMlpVA[mlpfd]->desc_w2[i][j] - S);
                                if (fabs (beta * pMlpVA[mlpfd]->delta_w2[i][j]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w2[i][j]))
                                    deltaW = beta * pMlpVA[mlpfd]->delta_w2[i][j];
                                else
                                {
                                    if (beta < 0)
                                        deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w2[i][j];
                                    else
                                        deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w2[i][j];
                                }
                            }
                            pMlpVA[mlpfd]->desc_w2[i][j] = S;
                            pMlpVA[mlpfd]->delta_w2[i][j] = deltaW;
                            pMlpVA[mlpfd]->theta2[j] = pMlpVA[mlpfd]->theta2[j] + deltaW;
                        }
                    }
                }
            }

            /*Adaptiere Gewichte der Zwischenschichten */
            for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
            {
                if (l == 0)
                {
                    for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (k < pMlpVA[mlpfd]->mlpIv.m)
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER (S);
                                if ((pMlpVA[mlpfd]->delta_w1[l][k][i] == 0) || (S > pMlpVA[mlpfd]->desc_w[l][k][i]))
                                {
                                    /*Standard Error Backpropagation Lernregel */
                                    deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta1 * S;
                                }
                                else
                                {
                                    if (S > pMlpVA[mlpfd]->desc_w[l][k][i])
                                    {
                                        printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w[l][k][i]);
                                        return -2;
                                    }
                                    beta = S / (pMlpVA[mlpfd]->desc_w[l][k][i] - S);
                                    if (fabs (beta * pMlpVA[mlpfd]->delta_w1[l][k][i]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w1[l][k][i]))
                                        deltaW = beta * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                    else
                                    {
                                        if (beta < 0)
                                            deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                        else
                                            deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                    }
                                }
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                                pMlpVA[mlpfd]->delta_w1[l][k][i] = deltaW;
                                pMlpVA[mlpfd]->w[l][k][i] = pMlpVA[mlpfd]->w[l][k][i] + deltaW;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                if ((pMlpVA[mlpfd]->delta_w1[l][k][i] == 0) || (S > pMlpVA[mlpfd]->desc_w[l][k][i]))
                                {
                                    /*Standard Error Backpropagation Lernregel */
                                    deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta1 * S;
                                }
                                else
                                {
                                    if (S > pMlpVA[mlpfd]->desc_w[l][k][i])
                                    {
                                        printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w[l][k][i]);
                                        return -2;
                                    }
                                    beta = S / (pMlpVA[mlpfd]->desc_w[l][k][i] - S);
                                    if (fabs (beta * pMlpVA[mlpfd]->delta_w1[l][k][i]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w1[l][k][i]))
                                        deltaW = beta * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                    else
                                    {
                                        if (beta < 0)
                                            deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                        else
                                            deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][k][i];
                                    }
                                }
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                                pMlpVA[mlpfd]->delta_w1[l][k][i] = deltaW;
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + deltaW;
                            }
                        }
                    }
                }
                else
                {
                    for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN (S);
                                if ((pMlpVA[mlpfd]->delta_w1[l][i2][i] == 0) || (S > pMlpVA[mlpfd]->desc_w[l][i2][i]))
                                {
                                    /*Standard Error Backpropagation Lernregel */
                                    deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta1 * S;
                                }
                                else
                                {
                                    if (S > pMlpVA[mlpfd]->desc_w[l][i2][i])
                                    {
                                        printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                                        return -2;
                                    }
                                    beta = S / (pMlpVA[mlpfd]->desc_w[l][i2][i] - S);
                                    if (fabs (beta * pMlpVA[mlpfd]->delta_w1[l][i2][i]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w1[l][i2][i]))
                                        deltaW = beta * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                    else
                                    {
                                        if (beta < 0)
                                            deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                        else
                                            deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                    }
                                }
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                                pMlpVA[mlpfd]->delta_w1[l][i2][i] = deltaW;
                                pMlpVA[mlpfd]->w[l][i2][i] = pMlpVA[mlpfd]->w[l][i2][i] + deltaW;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                if ((pMlpVA[mlpfd]->delta_w1[l][i2][i] == 0) || (S > pMlpVA[mlpfd]->desc_w[l][i2][i]))
                                {
                                    /*Standard Error Backpropagation Lernregel */
                                    deltaW = -1 * pMlpVA[mlpfd]->mlpP.eta1 * S;
                                }
                                else
                                {
                                    if (S > pMlpVA[mlpfd]->desc_w[l][i2][i])
                                    {
                                        printf ("### Quickprop: S(t) = %f, S(t-1) = %f\n", S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                                        return -2;
                                    }
                                    beta = S / (pMlpVA[mlpfd]->desc_w[l][i2][i] - S);
                                    if (fabs (beta * pMlpVA[mlpfd]->delta_w1[l][i2][i]) <= pMlpVA[mlpfd]->mlpP.beta_max * fabs (pMlpVA[mlpfd]->delta_w1[l][i2][i]))
                                        deltaW = beta * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                    else
                                    {
                                        if (beta < 0)
                                            deltaW = -pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                        else
                                            deltaW = pMlpVA[mlpfd]->mlpP.beta_max * pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                    }
                                }
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                                pMlpVA[mlpfd]->delta_w1[l][i2][i] = deltaW;
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + deltaW;
                            }
                        }
                    }
                }
            }
        } /* MOMENTUMTERM_MODE */
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == RPROPM_MODE)
        {
            /*Resilient Propagation Minus (RPROP-) Mode */

            /*Adaptiere Gewichte nach Praesentation aller Muster micro */
            /*Adaptiere Schwellwerte nach Praesentation aller Muster micro */

            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
            for (i = 0; i <= pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    if (i < pMlpVA[mlpfd]->mlpIv.h[l])
                    {
                        CALC_GRADIENT_WEIGHTS_OUTPUT (S);
                        
                        /*Berechnung von delta ij */
                        CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w2[i][j]);
                        if (dS > 0)
                        {
                            /*S(t-1)*S(t)>0 */
                            CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                        }
                        else if (dS < 0)
                        {
                            /*S(t-1)*S(t)<0 */
                            CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                        }
                        else
                        {
                            /*S(t-1)*S(t)=0 */
                        }

                        CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->w2[i][j], pMlpVA[mlpfd]->eta_w2[i][j]);
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                    }
                    else
                    {
                        /* neurons with linear activation function don't have a threshold */
                        if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
                        {
                            /*Adaptiere Schwellwerte der Ausgangsschicht */
                            CALC_GRADIENT_BIAS_OUTPUT (S)

                            /*Berechnung von delta ij */
                            CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w2[i][j]);
                            if (dS > 0)
                            {
                                /*S(t-1)*S(t)>0 */
                                CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                            }
                            else if (dS < 0)
                            {
                                /*S(t-1)*S(t)<0 */
                                CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                /*                        S = 0; */
                            }
                            else
                            {
                                /*S(t-1)*S(t)=0 */
                            }
                            CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->theta2[j], pMlpVA[mlpfd]->eta_w2[i][j]);
                            pMlpVA[mlpfd]->desc_w2[i][j] = S;
                        }
                    }
                }
            }

            /*Adaptiere Gewichte der Zwischenschichten */
            for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
            {
                if (l == 0)
                {
                    for (k = 0; k <= pMlpVA[mlpfd]->mlpIv.m; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (k < pMlpVA[mlpfd]->mlpIv.m)
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER (S);
                                /*Berechnung von delta ij */
                                CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][k][i]);
                                if (dS > 0)
                                {
                                    /*S(t-1)*S(t)>0 */
                                    CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                }
                                else if (dS < 0)
                                {
                                    /*S(t-1)*S(t)<0 */
                                    CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                    /*                                    S = 0; */
                                }
                                else
                                {
                                    /*S(t-1)*S(t)=0 */
                                }
                                CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->w[l][k][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                /*Berechnung von delta ij */
                                CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][k][i]);
                                if (dS > 0)
                                {
                                    /*S(t-1)*S(t)>0 */
                                    CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                }
                                else if (dS < 0)
                                {
                                    /*S(t-1)*S(t)<0 */
                                    CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                    /*                                S = 0; */
                                }
                                else
                                {
                                    /*S(t-1)*S(t)=0 */
                                }
                                CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->theta[l][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                            }
                        }
                    }
                }
                else
                {
                    for (i2 = 0; i2 <= pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            if (i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1])
                            {
                                CALC_GRADIENT_WEIGHTS_HIDDEN (S);
                                /*Berechnung von delta ij */
                                CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                                if (dS > 0)
                                {
                                    /*S(t-1)*S(t)>0 */
                                    CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                }
                                else if (dS < 0)
                                {
                                    /*S(t-1)*S(t)<0 */
                                    CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                    /*                                    S = 0; */
                                }
                                else
                                {
                                    /*S(t-1)*S(t)=0 */
                                }
                                CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->w[l][i2][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                            }
                            else
                            {
                                /*Adaptiere Schwellwerte der Zwischenschichten */
                                CALC_GRADIENT_BIAS_HIDDEN (S);
                                /*Berechnung von delta ij */
                                CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                                if (dS > 0)
                                {
                                    /*S(t-1)*S(t)>0 */
                                    CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                }
                                else if (dS < 0)
                                {
                                    /*S(t-1)*S(t)<0 */
                                    CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                    /*                                S = 0; */
                                }
                                else
                                {
                                    /*S(t-1)*S(t)=0 */
                                }
                                CALC_RESILIENT_UPDATE_WEIGHT (S, pMlpVA[mlpfd]->theta[l][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                            }
                        }
                    }
                }
            }
        } /* RPROPM_MODE */
        else if (pMlpVA[mlpfd]->mlpP.trainingMode == RPROPP_MODE)
        {
            /*Resilient Propagation Plus (RPROP+) Mode */

            /*Adaptiere Gewichte nach Praesentation aller Muster micro */

            l = pMlpVA[mlpfd]->mlpIv.nrHiddenLayers - 1;
            /*Adaptiere Gewichte der Ausgangsschicht */
            for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
            {
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    CALC_GRADIENT_WEIGHTS_OUTPUT (S);
                    /*Berechnung von delta ij */
                    CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w2[i][j]);
                    if (dS > 0)
                    {
                        /*S(t-1)*S(t)>0 */
                        CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                        CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w2[i][j], pMlpVA[mlpfd]->eta_w2[i][j]);
                        pMlpVA[mlpfd]->w2[i][j] = pMlpVA[mlpfd]->w2[i][j] + pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                    }
                    else if (dS < 0)
                    {
                        /*S(t-1)*S(t)<0 */
                        CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                        if (E > Elast)
                            pMlpVA[mlpfd]->w2[i][j] = pMlpVA[mlpfd]->w2[i][j] - pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = 0;
                    }
                    else
                    {
                        /*S(t-1)*S(t)=0 */
                        CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w2[i][j], pMlpVA[mlpfd]->eta_w2[i][j]);
                        pMlpVA[mlpfd]->w2[i][j] = pMlpVA[mlpfd]->w2[i][j] + pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                    }
                }
            }

            /*Adaptiere Schwellwerte nach Praesentation aller Muster micro */

            /* neurons with linear activation function don't have a threshold */
            if (pMlpVA[mlpfd]->mlpIv.hasThresholdOutput)
            {

                /*Adaptiere Schwellwerte der Ausgangsschicht */
                for (j = 0; j < pMlpVA[mlpfd]->mlpIv.n; j++)
                {
                    CALC_GRADIENT_BIAS_OUTPUT (S);
                    i = pMlpVA[mlpfd]->mlpIv.h[l];
                    /*Berechnung von delta ij */
                    CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w2[i][j]);
                    if (dS > 0)
                    {
                        /*S(t-1)*S(t)>0 */
                        CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                        CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w2[i][j], pMlpVA[mlpfd]->eta_w2[i][j]);
                        pMlpVA[mlpfd]->theta2[j] = pMlpVA[mlpfd]->theta2[j] + pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                    }
                    else if (dS < 0)
                    {
                        /*S(t-1)*S(t)<0 */
                        CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w2[i][j], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                        if (E > Elast)
                            pMlpVA[mlpfd]->theta2[j] = pMlpVA[mlpfd]->theta2[j] - pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = 0;
                    }
                    else
                    {
                        /*S(t-1)*S(t)=0 */
                        CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w2[i][j], pMlpVA[mlpfd]->eta_w2[i][j]);
                        pMlpVA[mlpfd]->theta2[j] = pMlpVA[mlpfd]->theta2[j] + pMlpVA[mlpfd]->delta_w2[i][j];
                        pMlpVA[mlpfd]->desc_w2[i][j] = S;
                    }
                }
            }

            /*Adaptiere Gewichte der Zwischenschichten */
            for (l = 0; l < pMlpVA[mlpfd]->mlpIv.nrHiddenLayers; l++)
            {
                if (l == 0)
                {
                    for (k = 0; k < pMlpVA[mlpfd]->mlpIv.m; k++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            CALC_GRADIENT_WEIGHTS_HIDDEN_FIRSTLAYER (S);
                            /*Berechnung von delta ij */
                            CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][k][i]);
                            if (dS > 0)
                            {
                                /*S(t-1)*S(t)>0 */
                                CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][k][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                                pMlpVA[mlpfd]->w[l][k][i] = pMlpVA[mlpfd]->w[l][k][i] + pMlpVA[mlpfd]->delta_w1[l][k][i];
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                            }
                            else if (dS < 0)
                            {
                                /*S(t-1)*S(t)<0 */
                                CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                if (E > Elast)
                                    pMlpVA[mlpfd]->w[l][k][i] = pMlpVA[mlpfd]->w[l][k][i] - pMlpVA[mlpfd]->delta_w1[l][k][i];
                                pMlpVA[mlpfd]->desc_w[l][k][i] = 0;
                            }
                            else
                            {
                                /*S(t-1)*S(t)=0 */
                                CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][k][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                                pMlpVA[mlpfd]->w[l][k][i] = pMlpVA[mlpfd]->w[l][k][i] + pMlpVA[mlpfd]->delta_w1[l][k][i];
                                pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                            }
                        }
                    }
                    /*Adaptiere Schwellwerte der Zwischenschichten */
                    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    {
                        CALC_GRADIENT_BIAS_HIDDEN (S);
                        k = pMlpVA[mlpfd]->mlpIv.m;
                        /*Berechnung von delta ij */
                        CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][k][i])
                        if (dS > 0)
                        {
                            /*S(t-1)*S(t)>0 */
                            CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                            CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][k][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                            pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + pMlpVA[mlpfd]->delta_w1[l][k][i];
                            pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                        }
                        else if (dS < 0)
                        {
                            /*S(t-1)*S(t)<0 */
                            CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][k][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                            if (E > Elast)
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] - pMlpVA[mlpfd]->delta_w1[l][k][i];
                            pMlpVA[mlpfd]->desc_w[l][k][i] = 0;
                        }
                        else
                        {
                            /*S(t-1)*S(t)=0 */
                            CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][k][i], pMlpVA[mlpfd]->eta_w[l][k][i]);
                            pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + pMlpVA[mlpfd]->delta_w1[l][k][i];
                            pMlpVA[mlpfd]->desc_w[l][k][i] = S;
                        }
                    }
                }
                else
                {
                    for (i2 = 0; i2 < pMlpVA[mlpfd]->mlpIv.h[l - 1]; i2++)
                    {
                        for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                        {
                            CALC_GRADIENT_WEIGHTS_HIDDEN (S);
                            /*Berechnung von delta ij */
                            CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                            if (dS > 0)
                            {
                                /*S(t-1)*S(t)>0 */
                                CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                                CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][i2][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                                pMlpVA[mlpfd]->w[l][i2][i] = pMlpVA[mlpfd]->w[l][i2][i] + pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                            }
                            else if (dS < 0)
                            {
                                /*S(t-1)*S(t)<0 */
                                CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                                if (E > Elast)
                                    pMlpVA[mlpfd]->w[l][i2][i] = pMlpVA[mlpfd]->w[l][i2][i] - pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = 0;
                            }
                            else
                            {
                                /*S(t-1)*S(t)=0 */
                                CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][i2][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                                pMlpVA[mlpfd]->w[l][i2][i] = pMlpVA[mlpfd]->w[l][i2][i] + pMlpVA[mlpfd]->delta_w1[l][i2][i];
                                pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                            }
                        }
                    }

                    /*Adaptiere Schwellwerte der Zwischenschichten */
                    for (i = 0; i < pMlpVA[mlpfd]->mlpIv.h[l]; i++)
                    {
                        CALC_GRADIENT_BIAS_HIDDEN (S);
                        i2 = pMlpVA[mlpfd]->mlpIv.h[l - 1];
                        /*Berechnung von delta ij */
                        CALC_RESILIENT_DS (dS, S, pMlpVA[mlpfd]->desc_w[l][i2][i]);
                        if (dS > 0)
                        {
                            /*S(t-1)*S(t)>0 */
                            CALC_RESILIENT_DELTA_INCREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_pos, pMlpVA[mlpfd]->mlpP.eta_max);
                            CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][i2][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                            pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + pMlpVA[mlpfd]->delta_w1[l][i2][i];
                            pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                        }
                        else if (dS < 0)
                        {
                            /*S(t-1)*S(t)<0 */
                            CALC_RESILIENT_DELTA_DECREASE (pMlpVA[mlpfd]->eta_w[l][i2][i], pMlpVA[mlpfd]->mlpP.eta_neg, pMlpVA[mlpfd]->mlpP.eta_min);
                            if (E > Elast)
                                pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] - pMlpVA[mlpfd]->delta_w1[l][i2][i];
                            pMlpVA[mlpfd]->desc_w[l][i2][i] = 0;
                        }
                        else
                        {
                            /*S(t-1)*S(t)=0 */
                            CALC_RESILIENT_DELTA_WEIGHT (S, pMlpVA[mlpfd]->delta_w1[l][i2][i], pMlpVA[mlpfd]->eta_w[l][i2][i]);
                            pMlpVA[mlpfd]->theta[l][i] = pMlpVA[mlpfd]->theta[l][i] + pMlpVA[mlpfd]->delta_w1[l][i2][i];
                            pMlpVA[mlpfd]->desc_w[l][i2][i] = S;
                        }
                    }
                }
            }
        } /* RPROPP_MODE */

        /*Schritt 7: Ende Training */
        numIterations++;
        if (E < Emin)
            Emin = E;
        if ((E <= pMlpVA[mlpfd]->mlpP.epsilon) || (numIterations >= pMlpVA[mlpfd]->mlpP.maxIterations))
            end = 1;

        if(pMlpVA[mlpfd]->mlpP.trainingMode != RG_MODE && 
           pMlpVA[mlpfd]->mlpP.trainingMode != TDC_MODE)
        {
            if(E <= Elast)
                ouch = 0;
            else if(E > Elast)
            {
                ouch++;
            
                if (ouch > 10)
                {
                    ouch = 0;
                    printf ("### Algorithm does not converge (E > Elast), lower the learning rate or stop earlier\n");
                }
            }
        }
        
        if (pMlpVA[mlpfd]->mlpP.verboseOutput)
        {
            double mean = gsl_stats_mean (Evals, 1, micro_max);
            double largest = gsl_stats_max (Evals, 1, micro_max);
            gsl_sort (Evals, 1, micro_max);
            double median = gsl_stats_median_from_sorted_data (Evals, 1, micro_max);
            double upperq = gsl_stats_quantile_from_sorted_data (Evals, 1, micro_max, 0.75);

            printf ("number of iterations: %i, E: %lf. Max: %lf, Mean: %lf, Median: %lf, uQuart: %lf\n", numIterations, E, largest, mean, median, upperq);
        }
        
        if (pMlpVA[mlpfd]->mlpP.verboseOutput >= 2)
            outputWeights (mlpfd);
        
    } /* while */
    
    if (pMlpVA[mlpfd]->mlpP.verboseOutput)
        outputWeights (mlpfd);

    return E;
}
