#include <stdio.h>
#include <string.h>

void getFileLines(char **path, int *nLines)
{
    FILE *fp = fopen(*path, "r");
    
    if(fp == NULL)
    {
//        printf("### getFileLines: File (%s) not found\n", *path);
        return;
    }
    
    *nLines = 0;
    
    char buf[256];
    
    while (fgets (buf, sizeof(buf), fp))
    {
        *nLines = *nLines + 1;
    }
    
    fclose(fp);
}
