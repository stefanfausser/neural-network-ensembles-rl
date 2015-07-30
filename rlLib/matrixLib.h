int allocateVector (
    double **x,
    unsigned long length);
    
int freeVector (
    double *x);
    
int allocateMatrix2 (
    double ***x,
    unsigned long rows,
    unsigned long columns);
    
int freeMatrix2 (
    double **x,
    unsigned long rows);
    
int allocateArray3 (
    double ****x,
    unsigned long firstDim,
    unsigned long secondDim,
    unsigned long thirdDim);
    
int freeArray3 (
    double ***x,
    unsigned long firstDim,
    unsigned long secondDim);
    
double sign (
    double val);
    
int setMatrixValue (
    double **x,
    unsigned long rows,
    unsigned long columns,
    double value);
    
int copyMatrix (
    double **dst,
    double **src,
    unsigned long rows,
    unsigned long columns);
