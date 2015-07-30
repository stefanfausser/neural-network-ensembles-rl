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
 * \file matrixLib.c
 * \brief Matrix allocation library
 *
 * \author Stefan Faußer
 * 
 * Modification history:
 * 
 * 2010-07-01, S. Fausser - written
 */

#include <stdlib.h>
#include <string.h>

int allocateVector (
    double **x,
    unsigned long length)
{
    double *xinternal;
    xinternal = (double *) malloc (sizeof (double) * length);
    if (xinternal == NULL)
        return -1;

    *x = xinternal;
    return 0;
}

int freeVector (
    double *x)
{
    free (x);
    return 0;
}

int allocateMatrix2 (
    double ***x,
    unsigned long rows,
    unsigned long columns)
{
    double **xinternal;
    unsigned long i;

    xinternal = (double **) malloc (sizeof (double *) * rows);
    if (xinternal == NULL)
        return -1;
    for (i = 0; i < rows; i++)
    {
        xinternal[i] = (double *) malloc (sizeof (double) * columns);
        if (xinternal[i] == NULL)
            return -2;
    }

    *x = xinternal;
    return 0;
}

int freeMatrix2 (
    double **x,
    unsigned long rows)
{
    unsigned long i;

    for (i = 0; i < rows; i++)
    {
        free (x[i]);
    }
    free (x);

    return 0;
}

int allocateArray3 (
    double ****x,
    unsigned long firstDim,
    unsigned long secondDim,
    unsigned long thirdDim)
{
    double ***xinternal;
    unsigned long i, j;

    xinternal = (double ***) malloc (sizeof (double **) * firstDim);
    if (xinternal == NULL)
        return -1;
    for (i = 0; i < firstDim; i++)
    {
        xinternal[i] = (double **) malloc (sizeof (double *) * secondDim);
        if (xinternal[i] == NULL)
            return -1;
        for (j = 0; j < secondDim; j++)
        {
            xinternal[i][j] = (double *) malloc (sizeof (double) * thirdDim);
            if (xinternal[i][j] == NULL)
                return -1;
        }
    }

    *x = xinternal;
    return 0;
}

int freeArray3 (
    double ***x,
    unsigned long firstDim,
    unsigned long secondDim)
{
    unsigned long i, j;

    for (i = 0; i < firstDim; i++)
    {
        for (j = 0; j < secondDim; j++)
        {
            free (x[i][j]);
        }
        free (x[i]);
    }
    free (x);

    return 0;
}

double sign (
    double val)
{
    if (val > 0)
        return +1.0;
    else if (val < 0)
        return -1.0;
    else
        return 0;
}

int setMatrixValue (
    double **x,
    unsigned long rows,
    unsigned long columns,
    double value)
{
    unsigned long i;

    for (i = 0; i < rows; i++)
    {
        memset (x[i], value, sizeof (double) * columns);
    }

    return 0;
}

int copyMatrix (
    double **dst,
    double **src,
    unsigned long rows,
    unsigned long columns)
{
    unsigned long i;

    for (i = 0; i < rows; i++)
    {
        memcpy (dst[i], src[i], sizeof (double) * columns);
    }

    return 0;
}
