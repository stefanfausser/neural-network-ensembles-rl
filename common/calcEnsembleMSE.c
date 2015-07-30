#include <stdio.h>
#include <string.h>
#include <math.h>

void calcEnsembleMSE(double *err, int *M, int *N, double *weights, double *mse)
{
    *mse = 0;

    double sumWeights = 0;
    
    unsigned m;
    for(m = 0; m < *M; m++)
        sumWeights += weights[m];
    
    unsigned i;
    for(i = 0; i < *N; i++)
    {
        double summedErr = 0;

        for(m = 0; m < *M; m++)
            summedErr += weights[m] * err[m + *M * i];

        *mse = *mse + 1.0 / (double) *N * pow((1.0 / sumWeights * summedErr), 2.0);
    }
}
