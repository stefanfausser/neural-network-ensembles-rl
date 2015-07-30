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
 * \file commonNeuralLib.c
 * \brief Training data allocation library
 *
 * \author Stefan Fausser
 * 
 * Modification history:
 * 
 * 2010-07-01, S. Fausser - written
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "commonNeuralLib.h"
#include "matrixLib.h"

int freeTrainData (
    struct train_data *pData)
{
    unsigned long micro;

    if (pData == NULL)
        return -2;

    for (micro = 0; micro < pData->samples; micro++)
    {
        free (pData->x[micro]);
        if(pData->x_prime != NULL)
            free( pData->x_prime[micro]);
        free (pData->y[micro]);
        free (pData->y2[micro]);
    }
    free (pData->x);
    if(pData->x_prime != NULL)
        free (pData->x_prime);
    free (pData->y);
    free (pData->y2);
    
    free(pData->gamma);
    free(pData->reward);
    free(pData->delta);
    free(pData->hasXPrime);
    
    return 0;
}

int allocateTrainData (
    struct train_data *pData,
    unsigned long micro_max,
    unsigned short m,
    unsigned short n,
    bool allocatePrime)
{
    int ret;

    if (pData == NULL)
        return -2;

    ret = allocateMatrix2 (&pData->x, micro_max, m);
    if (ret)
        return -3;

    if(allocatePrime)
    {
        ret = allocateMatrix2 (&pData->x_prime, micro_max, m);
        if (ret)
            return -3;
    }
    else
        pData->x_prime = NULL;
    
    ret = allocateMatrix2 (&pData->y, micro_max, n);
    if (ret)
        return -4;

    ret = allocateMatrix2 (&pData->y2, micro_max, n);
    if (ret)
        return -5;
    
    pData->reward = malloc(sizeof(double) * micro_max);
    if(pData->reward == NULL)
        return -6;
    
    pData->gamma = malloc(sizeof(double) * micro_max);
    if(pData->gamma == NULL)
        return -7;

    pData->delta = malloc(sizeof(double) * micro_max);
    if(pData->delta == NULL)
        return -8;

    pData->hasXPrime = malloc(sizeof(bool) * micro_max);
    if(pData->hasXPrime == NULL)
        return -9;

    pData->samples = micro_max;
    pData->xSize = m;
    pData->ySize = n;

    return 0;
}

int getAndAllocateTrainDataFile (
    char *filename,
    struct train_data *pData,
    unsigned short *m,
    unsigned short *n,
    unsigned long *micro_max)
{
    FILE *fp = fopen (filename, "r");
    if (fp == NULL)
    {
        printf ("### getAndAllocateTrainDataFile: unable to open file (%s)\n", filename);
        return -1;
    }
    
    unsigned long param[3];
    int i;
    
    for(i = 0; i < 3; i++)
    {
        char buffer[128];
        
        if(fgets(buffer, 128, fp) == NULL)
        {
            printf("### getAndAllocateTrainDataFile: Unable to read 'param' (%i)\n", i);
            return -2;    
        }
        
        int ret = sscanf(buffer, "%lu", &param[i]);
        if(ret != 1)
        {
            printf("### getAndAllocateTrainDataFile: Unable to parse 'param' (%i)\n", i);
            return -3;
        }        
    }

    *m = param[0];
    *n = param[1];
    *micro_max = param[2];

    int ret = allocateTrainData(pData, *micro_max, *m, *n, false);
    if(ret)
    {
        printf("### getAndAllocateTrainDataFile: Unable to allocate data for (%lu) samples\n", param[2]);
        return -4;
    }    
    
    unsigned long micro;
    
    for(micro = 0; micro < *micro_max; micro++)
    {
        char buffer[1024];
        
        if(fgets(buffer, 1024, fp) == NULL)
        {
            printf("### getAndAllocateTrainDataFile: Unable to read sample (%lu)\n", micro);
            return -2;
        }
        
        char *pBuffer = buffer;
        
        unsigned long k;

        for(k = 0; k < *m; k++)
            pData->x[micro][k] = strtod(pBuffer, &pBuffer);

        unsigned long j;

        for(j = 0; j < *n; j++)
            pData->y[micro][j] = strtod(pBuffer, &pBuffer);        
    }
    
    fclose (fp);

    return 0;
}

int getTrainDataLen (
    struct train_data *pData,
    unsigned long *len)
{
    if (pData == NULL || len == NULL)
        return -2;

    *len = pData->samples;

    return 0;
}

void outputTrainDataStdout(
    struct train_data *pData)
{
    printf("Outputting training data:\n");
    
    printf("m (%lu), n (%lu), micro_max (%lu)\n", pData->xSize, pData->ySize, pData->samples);

    unsigned short k, j;
    unsigned long micro;
    
    for(micro = 0; micro < pData->samples; micro++)
    {
        printf("Sample (%10lu): ", micro);

        for(k = 0; k < pData->xSize; k++)
            printf("%lf ", pData->x[micro][k]);

        for(j = 0; j < pData->ySize; j++)
            printf("%lf ", pData->y[micro][j]);
        
        printf("\n");
    }
}
