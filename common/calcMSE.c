#include <stdio.h>
#include <string.h>
#include <math.h>

void calcMSE(double *err, int *M, int *N, double *weights, double *mse)
{
    *mse = 0;

    double sumWeights = 0;
    
    unsigned m;
    for(m = 0; m < *M; m++)
        sumWeights += weights[m];

    for(m = 0; m < *M; m++)
    {
        double summedErr = 0;

        unsigned i;
        for(i = 0; i < *N; i++)
            summedErr += pow(err[m + *M * i], 2.0);
        
        *mse = *mse + 1.0 / sumWeights * 1.0 / (double) *N * weights[m] * summedErr;
    }
}
