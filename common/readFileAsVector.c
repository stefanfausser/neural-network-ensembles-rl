#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void readFileAsVector(char **path, int *maxVals, double *vals, int *readVals)
{
    FILE *fp = fopen(*path, "r");
    
    if(fp == NULL)
    {
//        printf("### getFileLines: File (%s) not found\n", *path);
        return;
    }

    char buf[256];
    
    *readVals = 0;
    
    while (fgets (buf, sizeof(buf), fp))
    {
        vals[*readVals] = atof(buf);
    
        *readVals = *readVals + 1;
        
        if(*readVals >= *maxVals)
        {
            fclose(fp);
            
            return;
        }
    }
    
    fclose(fp);
}
