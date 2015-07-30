// The MIT License (MIT)
// 
// Copyright (c) 2012 - 2015 Stefan Faußer
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
 * \file sz-tetris.c
 * \brief SZ-Tetris environment
 *
 * \author Stefan Fausser
 * 
 * Modification history:
 * 
 * 2012-04-01, S.Fausser - written
 */

/* Tested functions:
 * tetrisGetAllPos / tetrisGetLowestPos
 * tetrisApplyPos
 * tetrisUnapplyPos
 * tetrisCheckFullLines
 * tetrisEraseLines
 * tetrisEraseBoard
 */ 

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>             /*malloc */
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#include "tdlLib.h"
#include "matrixLib.h"

#define MAX_NUM_ACTIONS         17
#define ALPHA_DISCOUNT          1.0
#define MAX_HOLES               150
// Attention: MAX_PIECE_SEQUENCE must be a multiple of 8
#define MAX_PIECE_SEQUENCE      1000000

//#define BENCHMARK_DIFFERENCES
#undef BENCHMARK_DIFFERENCES

#ifndef INFINITY
#   define INFINITY 999999
#endif

#define BOARD_WIDTH     10
#define BOARD_HEIGHT    20

#define BOARD_SQUARES   (BOARD_WIDTH * BOARD_HEIGHT)

#define VERIFY_APPLY_POS

int allocateMatrixType2 (
    void ***x,
    int size,
    unsigned long rows,
    unsigned long columns)
{
    void **xinternal;
    unsigned long i;

    xinternal = (void **) malloc (sizeof (void *) * rows);
    if (xinternal == NULL)
        return -1;
    for (i = 0; i < rows; i++)
    {
        xinternal[i] = (void *) malloc (size * columns);
        if (xinternal[i] == NULL)
            return -2;
    }

    *x = xinternal;
    return 0;
}

int freeMatrixType2 (
    void **x,
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

typedef struct
{
    uint8_t board[BOARD_HEIGHT][BOARD_WIDTH];
    uint8_t maxHeight;
    // TODO: Remove maxHeightLast
    uint8_t maxHeightLast;
} tTetris;

typedef struct
{
    int x;
    int y;
    bool isHorizontal;
} tPosition;

typedef struct
{
    tTetris tetris;
    int type;
} __attribute__((packed)) tStateInternal;

typedef struct
{
    uint8_t bitmap[3][3];
    unsigned height;
    unsigned width;
} tPiece;

typedef struct
{
    tPiece pieceHorizontal;
    tPiece pieceVertical;
} tPieceSet;

typedef struct
{
    int nStateRepoMax;    
    int nStateRepo;    
    tStateInternal *pStateInternalRepo;    
} tRepo;

static const uint8_t zPieceHorizontal[3][3]       = {{1,1,0}, {0,1,1}, {0,0,0}};
static const uint8_t zPieceVertical[3][3]         = {{0,1,0}, {1,1,0}, {1,0,0}};
static const uint8_t sPieceHorizontal[3][3]       = {{0,1,1}, {1,1,0}, {0,0,0}};
static const uint8_t sPieceVertical[3][3]         = {{1,0,0}, {1,1,0}, {0,1,0}};

const double biFeatures[][22] = { 
    {-3.9449, -9.3878, -3.2907, -9.4902, -15.1198, -2.5428, -7.2624, -3.6481, -24.8931, 8.2739, -23.8090, 3.9224, -21.8455, 2.5704, -25.9445, 5.4411, -10.6643, 1.5848, -27.1720, 3.6956, -68.3404, -1.1340},
    {11.0572, -31.3095, -1.94, -11.3902, -7.4452, -11.4852, -9.1258, 0.4228, -30.0521, 11.1677, -8.4923, 5.9814, -14.7293, 5.8015, -45.5280, 6.1028, -19.9065, 5.1730, -5.3666, 12.5566, -82.6131, -16.3142},
};

typedef enum
{
    eInputCoding_raw,
    eInputCoding_feature1,
    eInputCoding_bertsekasIoffe,
    eInputCoding_raw_top4,
    eInputCoding_raw_top6,
    eInputCoding_raw_top8,
    eInputCoding_raw_top10,
    eInputCoding_raw_top14,
    eInputCoding_bertsekasIoffe_normalized,
    eInputCoding_bertsekasIoffe_discretized1,
    eInputCoding_bertsekasIoffe_discretized1_heights,
    eInputCoding_bertsekasIoffe_discretized2,
    eInputCoding_bertsekasIoffe_discretized2_heights,
    eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles_reducedHeightDifferences,
    eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles,
    eInputCoding_bertsekasIoffe_discretized_onlyHeightDifferences,
    eInputCoding_bertsekasIoffe_discretized_onlyNumberOfHoles,
} eInputCoding;

const eInputCoding lInputCoding = eInputCoding_bertsekasIoffe_discretized1;
//const eInputCoding lInputCoding = eInputCoding_bertsekasIoffe_discretized2_heights;

typedef enum
{
    eReward_holes = 0,
    eReward_holes_logistic = 1,
    eReward_maxHeight = 2,
    eReward_bertsekasIoffe = 3,
} eReward;

eReward lRewardFunc = eReward_holes_logistic;

typedef struct
{
    int accessed;
    int date;
    tTetris tetris;
    int type;
    tPosition posTo;
} tLookupEntry;

static uint8_t **lPieceSequence = NULL;
static int lPieceSequenceLen = 0;
static int lMlpInputNeurons = 0;
static tPieceSet lPieceSet[2];
static bool lVerbose = false;
static bool lIncreasingEpsilon = false;

static tLookupEntry *pLookupEntries = NULL;
static unsigned long lNumberListEntries = 0;
static unsigned long lNumberListEntriesMax = 0;
static unsigned long lLookupDate = 0;

int tetrisApplyPos(
    tTetris *pTetris,
    int type,
    tPosition pos);

int tetrisUnapplyPos(
    tTetris *pTetris,
    int type,
    tPosition pos);

int tetrisEraseLines(
    tTetris *pTetris);

bool tetrisCheckFullLines(
    tTetris *pTetris);

int tetrisHeightsGet(
    tTetris *pTetris,
    int *pHeights,
    int *pHeightsDiff);

int tetrisCountHoles(
    tTetris *pTetris,
    int nLinesFromTop);

int tetrisCountHolesColWise(
    tTetris *pTetris);

int tetrisRewardGet(
    tTetris *pTetris,
    tConfigParam *pParam,
    double *pReward);

void outputBoard(
    tTetris *pTetris,
    int lineStart,
    int lineEnd)
{
    int row, col;

    if(lineStart > BOARD_HEIGHT - 1 || lineStart < 0)
        lineStart = BOARD_HEIGHT - 1;

    if(lineEnd < 0 || lineEnd > BOARD_HEIGHT - 1)
        lineEnd = 0;

    printf("Output board with maximum height (%i), last height (%i), from (%i), to (%i)\n", pTetris->maxHeight, pTetris->maxHeightLast, lineStart, lineEnd);

    int heights[BOARD_WIDTH];
    int heightsDiff[BOARD_WIDTH - 1];

    tetrisHeightsGet(pTetris, heights, heightsDiff);
    int x;
    printf("Heights: ");
    for(x = 0; x < BOARD_WIDTH; x++)
        printf("%i ", heights[x]);
    printf("\n");
    printf("Heights diffs: ");
    for(x = 0; x < (BOARD_WIDTH - 1); x++)
        printf("%i ", heightsDiff[x]);
    printf("\n");

    int nHoles = tetrisCountHoles(pTetris, BOARD_HEIGHT);
    printf("nHoles row-wise (%i)\n", nHoles);
    nHoles = tetrisCountHolesColWise(pTetris);
    printf("nHoles col-wise (%i)\n", nHoles);

    for(row = lineStart; row >= lineEnd; row--)
    {
        printf("|");

        for(col = 0; col < BOARD_WIDTH; col++)
        {
            if(pTetris->board[row][col])
                printf("#");
            else
                printf(" ");
        }

        printf("|\n");
    }
    printf("\n");
}

void copyBoard(
    tTetris *pTetrisTo,
    tTetris *pTetrisFrom)
{
    int y;

    for(y = 0; y < BOARD_HEIGHT; y++)
        memcpy(&pTetrisTo->board[y][0], &pTetrisFrom->board[y][0], BOARD_WIDTH);

    pTetrisTo->maxHeight = pTetrisFrom->maxHeight;
    pTetrisTo->maxHeightLast = pTetrisFrom->maxHeightLast;
}

int lookup_init(void)
{
    unsigned long size = lNumberListEntriesMax * sizeof(tLookupEntry);
    
    printf("lookup_init: Allocate (%li) elements, each with (%lf) MBytes, summed (%lf) MBytes\n", lNumberListEntriesMax, (double) sizeof(tLookupEntry) / 1024 / 1024, (double) size / 1024 / 1024);
    
    pLookupEntries = malloc(size);
    
    if(pLookupEntries == NULL)
    {
        printf("### malloc pLookupEntries failed\n");
        return -1;
    }
    
    int i;
    for(i = 0; i < lNumberListEntriesMax; i++)
    {
        pLookupEntries[i].accessed = 0;
        pLookupEntries[i].date = 0;
    }
    
    return 0;
}

void lookup_free(void)
{
    printf("lookup_free: Have (%li) entries:\n", lNumberListEntries);
    int i;
    for(i = 0; i < lNumberListEntries; i++)
    {
        printf("%i ", pLookupEntries[i].accessed);
    }
    printf("\n");
    
    if(pLookupEntries != NULL)
        free(pLookupEntries);    
}

int store(
    tTetris *pTetris,
    int type,
    tPosition posTo)
{
    if(lNumberListEntries < lNumberListEntriesMax)
    {
        memcpy(&pLookupEntries[lNumberListEntries].tetris, pTetris, sizeof(tTetris));
        pLookupEntries[lNumberListEntries].accessed = 0;
        pLookupEntries[lNumberListEntries].date = lLookupDate++;
        pLookupEntries[lNumberListEntries].type = type;
        pLookupEntries[lNumberListEntries].posTo = posTo;
        
        lNumberListEntries++;
        
        return 1;
    }
    else
    {
        // Overwrite one of the elements that have not been accessed
        int i, bestI = 0;
        int minAccessed = pLookupEntries[0].accessed;
        int minDate = pLookupEntries[0].date;
        for(i = 1; i < lNumberListEntries; i++)
        {
            if(pLookupEntries[i].accessed < minAccessed)
            {
                bestI = i;
                minAccessed = pLookupEntries[i].accessed;
                minDate = pLookupEntries[i].date;
            }
            else if(pLookupEntries[i].accessed == minAccessed &&
                    pLookupEntries[i].date < minDate)
            {
                bestI = i;
                minDate = pLookupEntries[i].date;                
            }
        }
        
        memcpy(&pLookupEntries[bestI].tetris, pTetris, sizeof(tTetris));
        pLookupEntries[bestI].accessed = 0;
        pLookupEntries[bestI].date = lLookupDate++;
        pLookupEntries[bestI].type = type;
        pLookupEntries[bestI].posTo = posTo;
        
        return 1;
    }

    return 0;
}

int lookup(
    tTetris *pTetris,
    int type,
    tPosition *pPosTo)
{
    int i;
    for(i = 0; i < lNumberListEntries; i++)
    {
        if(pLookupEntries[i].type == type)
        {
            if(!memcmp(&pLookupEntries[i].tetris, pTetris, sizeof(*pTetris)))
            {
                /* found */

                *pPosTo = pLookupEntries[i].posTo;

                pLookupEntries[i].accessed++;

                return 1;
            }
        }
    }
    
    return 0;
}

int tetrisHeightsGet(
    tTetris *pTetris,
    int *pHeights,
    int *pHeightsDiff)
{
    int x;
    for(x = 0; x < BOARD_WIDTH; x++)
    {
        int y = -1;
        for(y = pTetris->maxHeight - 1; y >= 0; y--)
        {
            if(pTetris->board[y][x])
                break;
        }

        pHeights[x] = y + 1;

        if(x > 0)
            pHeightsDiff[x - 1] = abs(pHeights[x] - pHeights[x - 1]);
    }

    return 0;
}

int tetrisCountHoles(
    tTetris *pTetris,
    int nLinesFromTop)
{
    int nHoles = 0;
    int y;
    int ymin = pTetris->maxHeight - nLinesFromTop;
    if(ymin < 0)
        ymin = 0;
    for(y = pTetris->maxHeight - 1; y >= ymin; y--)
    {
        int x;
        for(x = 0; x < BOARD_WIDTH; x++)
            nHoles += (pTetris->board[y][x] == 0 ? 1 : 0);
    }

    return nHoles;
}

int tetrisCountHolesColWise(
    tTetris *pTetris)
{
    int nHoles = 0;

    int x;
    for(x = 0; x < BOARD_WIDTH; x++)
    {
        int y = -1;
        for(y = pTetris->maxHeight - 1; y >= 0; y--)
        {
            if(pTetris->board[y][x])
                break;
        }

        for(; y >= 0; y--)
            nHoles += (pTetris->board[y][x] == 0 ? 1 : 0);
    }

    return nHoles;
}

int tetrisCountHolesInCol(
    tTetris *pTetris,
    int x)
{
    int nHoles = 0;

    int y = -1;
    for(y = pTetris->maxHeight - 1; y >= 0; y--)
    {
        if(pTetris->board[y][x])
            break;
    }

    for(; y >= 0; y--)
        nHoles += (pTetris->board[y][x] == 0 ? 1 : 0);

    return nHoles;
}

int tetrisInputVCodingGet(
    tTetris *pTetris,
    double *x)
{
    if(lInputCoding == eInputCoding_raw)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
        {
            int row = i / BOARD_WIDTH, col = i % BOARD_WIDTH;

            x[i] = (double) pTetris->board[row][col];
        }
    }
    else if(lInputCoding == eInputCoding_raw_top4 ||
            lInputCoding == eInputCoding_raw_top6 ||
            lInputCoding == eInputCoding_raw_top8 ||
            lInputCoding == eInputCoding_raw_top10 ||
            lInputCoding == eInputCoding_raw_top14)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
        {
            int row = i / BOARD_WIDTH, col = i % BOARD_WIDTH;

            if(row >= pTetris->maxHeight)
                x[i] = 0;
            else
            {
                /* X lowest lines */
//                x[i] = (double) pTetris->board[row][col];
                /* X highest lines */
                x[i] = (double) pTetris->board[pTetris->maxHeight - 1 - row][col];
            }
        }
    }
    else if(lInputCoding == eInputCoding_feature1)
    {
/*        x[0] = (double) tetrisCountHoles(pTetris, BOARD_HEIGHT);*/
        x[0] = (double) tetrisCountHolesColWise(pTetris);
        x[1] = BOARD_HEIGHT - pTetris->maxHeight;
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe)
    {
        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        int i;
        for(i = 0; i < BOARD_WIDTH; i++)
            x[offset + i] = heights[i];

        offset += BOARD_WIDTH;

        for(i = 0; i < (BOARD_WIDTH - 1); i++)
            x[offset + i] = heightsDiff[i];

        offset += (BOARD_WIDTH - 1);

        x[offset] = (BOARD_HEIGHT - pTetris->maxHeight);

        offset += 1;

        x[offset] = tetrisCountHolesColWise(pTetris);
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_normalized)
    {
        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        int i;
        for(i = 0; i < BOARD_WIDTH; i++)
            x[offset + i] = heights[i] / (double) BOARD_HEIGHT;

        offset += BOARD_WIDTH;

        for(i = 0; i < (BOARD_WIDTH - 1); i++)
            x[offset + i] = heightsDiff[i] / (double) BOARD_HEIGHT;

        offset += (BOARD_WIDTH - 1);

        x[offset] = (pTetris->maxHeight) / (double) BOARD_HEIGHT;

        offset += 1;

        // Expected are maximum holes = 1/2 of the board */
        x[offset] = tetrisCountHolesColWise(pTetris) / (double)(BOARD_HEIGHT * BOARD_WIDTH) / 2.0;
//        x[offset] = tetrisCountHolesColWise(pTetris) / (double)(BOARD_HEIGHT * BOARD_WIDTH) / 3.0;
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized1)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board height differences (seem to be more effective features than board heights)
        for(i = 0; i < (BOARD_WIDTH - 1); i++)
        {
            offset = i * BOARD_HEIGHT + heightsDiff[i];
            x[offset] = 1;
        }

        offset = (BOARD_WIDTH - 1) * BOARD_HEIGHT;

        // Expected are maximum holes = 3/4 of the board */
        int nHoles = tetrisCountHolesColWise(pTetris);
        if(nHoles >= MAX_HOLES)
        {
            printf("tetrisInputVCodingGet: nHoles overrun (%i)\n", nHoles);
            nHoles = MAX_HOLES - 1;
        }

        offset += nHoles;

        x[offset] = 1;
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized1_heights)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board heights
        for(i = 0; i < BOARD_WIDTH; i++)
        {
            offset = i * BOARD_HEIGHT + heights[i];
            x[offset] = 1;
        }

        offset = BOARD_WIDTH * BOARD_HEIGHT;

        // Expected are maximum holes = 3/4 of the board */
        int nHoles = tetrisCountHolesColWise(pTetris);
        if(nHoles >= MAX_HOLES)
        {
            printf("tetrisInputVCodingGet: nHoles overrun (%i)\n", nHoles);
            nHoles = MAX_HOLES - 1;
        }

        offset += nHoles;

        x[offset] = 1;
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized2)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board height differences (seem to be more effective features than board heights)
        for(i = 0; i < (BOARD_WIDTH - 1); i++)
        {
            offset = i * BOARD_HEIGHT + heightsDiff[i];
            x[offset] = 1;
        }

        int oldoffset = (BOARD_WIDTH - 1) * BOARD_HEIGHT;

        for(i = 0; i < BOARD_WIDTH; i++)
        {            
            int nHoles = tetrisCountHolesInCol(pTetris, i);
            
            offset = oldoffset + i * BOARD_HEIGHT + nHoles;
            
            x[offset] = 1;
        }
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized2_heights)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board heights
        for(i = 0; i < BOARD_WIDTH; i++)
        {
            offset = i * BOARD_HEIGHT + heights[i];
            x[offset] = 1;
        }

        int oldoffset = BOARD_WIDTH * BOARD_HEIGHT;

        for(i = 0; i < BOARD_WIDTH; i++)
        {            
            int nHoles = tetrisCountHolesInCol(pTetris, i);
            
            offset = oldoffset + i * BOARD_HEIGHT + nHoles;
            
            x[offset] = 1;
        }
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_onlyHeightDifferences)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board height differences (seem to be more effective features than board heights)
        for(i = 0; i < (BOARD_WIDTH - 1); i++)
        {
            int hDiff = heightsDiff[i];

            offset = i * BOARD_HEIGHT + hDiff;

            x[offset] = 1;            
        }        
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_onlyNumberOfHoles)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int offset = 0;

        for(i = 0; i < BOARD_WIDTH; i++)
        {            
            int nHoles = tetrisCountHolesInCol(pTetris, i);
            
            offset = i * BOARD_HEIGHT + nHoles;
            
            x[offset] = 1;
        }
        
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles_reducedHeightDifferences)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board height differences (seem to be more effective features than board heights)
        for(i = 0; i < (BOARD_WIDTH - 1); i++)
        {
            int hDiff = heightsDiff[i];

            // Restrict it to maximum 9
            if(hDiff >= 9)
                hDiff = 9;
            
            offset = i * 10 + hDiff;

            x[offset] = 1;            
        }
        
        int oldoffset = (BOARD_WIDTH - 1) * 10;

        for(i = 0; i < BOARD_WIDTH; i++)
        {            
            int nHoles = tetrisCountHolesInCol(pTetris, i);
            if(nHoles >= 9)
                nHoles = 9;
            
            offset = oldoffset + i * 10 + nHoles;
            
            x[offset] = 1;
        }
    }
    else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles)
    {
        int i;
        for(i = 0; i < lMlpInputNeurons; i++)
            x[i] = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        int offset = 0;

        // Board height differences (seem to be more effective features than board heights)
        for(i = 0; i < (BOARD_WIDTH - 1); i++)
        {
            int hDiff = heightsDiff[i];

            offset = i * BOARD_HEIGHT + hDiff;

            x[offset] = 1;            
        }
        
        int oldoffset = (BOARD_WIDTH - 1) * BOARD_HEIGHT;

        for(i = 0; i < BOARD_WIDTH; i++)
        {            
            int nHoles = tetrisCountHolesInCol(pTetris, i);
            if(nHoles >= 9)
                nHoles = 9;
            
            offset = oldoffset + i * 10 + nHoles;
            
            x[offset] = 1;
        }        
    }
    else
        return -1;

    return 0;
}

int tetrisRewardGet(
    tTetris *pTetris,
    tConfigParam *pParam,
    double *pReward)
{
    static double minScore = INFINITY, maxScore = -INFINITY;

    if(lRewardFunc == eReward_holes)
    {
//        int nHoles = tetrisCountHoles(pTetris, BOARD_HEIGHT);
        int nHoles = tetrisCountHolesColWise(pTetris);

        *pReward = pParam->reward_draw * (1.0 - nHoles / 150.0);

//        printf("nHoles (%i), reward (%lf)\n", nHoles, *pReward);

        return 0;
    }
    else if(lRewardFunc == eReward_holes_logistic)
    {
//        int nHoles = tetrisCountHoles(pTetris, BOARD_HEIGHT);
        int nHoles = tetrisCountHolesColWise(pTetris);

        *pReward = pParam->reward_draw * (1.0 / exp(1.0 / 33.0 * nHoles));

//        printf("nHoles (%i), reward (%lf)\n", nHoles, *pReward);
        
        return 0;
    }
    else if(lRewardFunc == eReward_bertsekasIoffe)
    {
        const int weightNum = 0;

        int heights[BOARD_WIDTH];
        int heightsDiff[BOARD_WIDTH - 1];

        tetrisHeightsGet(pTetris, heights, heightsDiff);

        double score = 0;
        int offset = 0;

        int x;
        for(x = 0; x < BOARD_WIDTH; x++)
            score += ((double) heights[x] * (double) biFeatures[weightNum][offset + x]);

        offset += BOARD_WIDTH;

        for(x = 0; x < (BOARD_WIDTH - 1); x++)
            score += ((double) heightsDiff[x] * (double) biFeatures[weightNum][offset + x]);

        offset += (BOARD_WIDTH - 1);

        score += ((double) (BOARD_HEIGHT - pTetris->maxHeight) * (double) biFeatures[weightNum][offset]);

        offset += 1;

        score += ((double) tetrisCountHolesColWise(pTetris) * (double) biFeatures[weightNum][offset]);

        // Scale it to [0,1]
        score = (score + 10876.0) / 10900;

        *pReward = score;

        if(score > maxScore)
            maxScore = score;
        if(score < minScore)
            minScore = score;

//        printf("minScore (%lf), maxScore (%lf)\n", minScore, maxScore);

        return 0;
    }
    else if(lRewardFunc == eReward_maxHeight)
    {
        *pReward = pParam->reward_draw / (pTetris->maxHeight + 1);

        return 0;
    }
    
    return 0;
}

int tetrisGetLowestPos(
    tTetris *pTetris,
    tPiece piece,
    unsigned x,
    bool *pIsValidPos,
    tPosition *pPos)
{
    pPos->x = -1;
    pPos->y = -1;
    
    // Is x outside the range?
    if(x + piece.width > BOARD_WIDTH)
    {
        *pIsValidPos = false;
        return 0;
    }

    *pIsValidPos = false;

    // Start at the top, try to get the piece far down
    int y;
    for(y = BOARD_HEIGHT - 1; y >= 0; y--)
    {
        // Check if the piece fits (all non-empty fields in the piece are only in the empty fields on the board)

        int x2,y2;
        int ymax = y + piece.height;
        if(ymax > BOARD_HEIGHT)
            ymax = BOARD_HEIGHT;
        for(y2 = y; y2 < ymax; y2++)
        {
            for(x2 = x; x2 < x + piece.width; x2++)
            {
                if(pTetris->board[y2][x2] && piece.bitmap[y2 - y][x2 - x])
                {
                    // As soon as the piece hits another piece then leave the function returning the last known-good position
                    return 0;
                }
            }
        }

        // Is the piece fitting on the board?
        if(y <= BOARD_HEIGHT - piece.height)
        {
            pPos->x = x;
            pPos->y = y;

            *pIsValidPos = true;
        }
    }

    return 0;
}

int tetrisGetAllPos(
    tTetris *pTetris,
    int type,
    tPosition *pPos,
    int *pNPos,
    int nPosMax)
{
    *pNPos = 0;

    unsigned x;
    bool isValidPos;

    if(nPosMax <= 0)
    {
        printf("### tetrisGetAllPos: Invalid nPosMax (%i)\n", nPosMax);
        return -1;
    }

    // The S-Z pieces have a minimum width of 2
    int i;
    for(i = 0; i < 2; i++)
    {
        tPiece piece;
        if(!i)
            piece = lPieceSet[type].pieceVertical;
        else
            piece = lPieceSet[type].pieceHorizontal;

        for(x = 0; x < BOARD_WIDTH - 1; x++)
        {
            tPosition pos;

            int ret = tetrisGetLowestPos(pTetris, piece, x, &isValidPos, &pos);
            if(ret)
            {
                printf("### tetrisGetAllPos: tetrisGetLowestPos failed (%i)\n", ret);
                return -2;
            }

            if(isValidPos)
            {
                if(!i)
                    pos.isHorizontal = false;
                else
                    pos.isHorizontal = true;
                pPos[*pNPos] = pos;
                *pNPos = (*pNPos) + 1;
                if(*pNPos >= nPosMax)
                    return 0;
            }
        }
    }

    return 0;
}

int tetrisApplyPos(
    tTetris *pTetris,
    int type,
    tPosition pos)
{
    tPiece piece;
    if(pos.isHorizontal)
        piece = lPieceSet[type].pieceHorizontal;
    else
        piece = lPieceSet[type].pieceVertical;

    int x2,y2;
    int ymax = pos.y + piece.height;
    for(y2 = pos.y; y2 < ymax; y2++)
    {
        for(x2 = pos.x; x2 < pos.x + piece.width; x2++)
        {
            if(piece.bitmap[y2 - pos.y][x2 - pos.x])
            {
#ifdef VERIFY_APPLY_POS
                if(pTetris->board[y2][x2] != 0)
                {
                    printf("### tetrisApplyPos: Fatal error, x (%i) and y (%i) position on board is not empty\n", x2, y2);
                    return -1;
                }
#endif
                pTetris->board[y2][x2] = piece.bitmap[y2 - pos.y][x2 - pos.x];
            }
        }
    }

    pTetris->maxHeightLast = pTetris->maxHeight;
    if(ymax > pTetris->maxHeight)
        pTetris->maxHeight = ymax;

    return 0;
}

int tetrisUnapplyPos(
    tTetris *pTetris,
    int type,
    tPosition pos)
{
    tPiece piece;
    if(pos.isHorizontal)
        piece = lPieceSet[type].pieceHorizontal;
    else
        piece = lPieceSet[type].pieceVertical;

    int x2,y2;
    for(y2 = pos.y; y2 < pos.y + piece.height; y2++)
    {
        for(x2 = pos.x; x2 < pos.x + piece.width; x2++)
        {
            if(piece.bitmap[y2 - pos.y][x2 - pos.x])
                pTetris->board[y2][x2] = 0;
        }
    }

    // Attention: This simple reversal of the maximum height only works if unapply is directly called after an apply call
    pTetris->maxHeight = pTetris->maxHeightLast;

    return 0;
}

bool tetrisCheckFullLines(
    tTetris *pTetris)
{
    int y;

    // Go from top to bottom
    for(y = BOARD_HEIGHT - 1; y >= 0; y--)
    {
        bool isFull = true;

        int x;
        for(x = 0; x < BOARD_WIDTH; x++)
        {
            if(!pTetris->board[y][x])
            {
                isFull = false;
                break;
            }
        }

        if(isFull)
            return true;            
    }
    
    return false;
}

int tetrisEraseLines(
    tTetris *pTetris)
{
    int y;

    int nErasedLines = 0;

    // Go from top to bottom
    for(y = BOARD_HEIGHT - 1; y >= 0; y--)
    {
        bool isFull = true;

        int x;
        for(x = 0; x < BOARD_WIDTH; x++)
        {
            if(!pTetris->board[y][x])
            {
                isFull = false;
                break;
            }
        }

        if(isFull)
        {
            // Erase the line, i.e. all lines above are falling down
            int y2;

            for(y2 = y; y2 < BOARD_HEIGHT; y2++)
            {
                if(y2 < BOARD_HEIGHT - 1)
                    memcpy(&pTetris->board[y2][0], &pTetris->board[y2 + 1][0], BOARD_WIDTH);
                else
                {
                    int x;
                    for(x = 0; x < BOARD_WIDTH; x++)
                        pTetris->board[y2][x] = 0;
                }
            }

            nErasedLines++;
        }
    }

    pTetris->maxHeight -= nErasedLines;

    return nErasedLines;
}

int tetrisEraseBoard(
    tTetris *pTetris)
{
    // Initialize board
    int row, col;
    for(row = BOARD_HEIGHT; row >= 0; row--)
    {
        for(col = 0; col < BOARD_WIDTH; col++)
        {
            pTetris->board[row][col] = 0;
        }
    }

    pTetris->maxHeight = 0;
    pTetris->maxHeightLast = 0;

    return 0;
}

int tetrisLearn(
    tTetris *pTetris,
    tConfigParam *pParam,
    int agentNo,
    bool isTerminalState,
    bool isExploited,
    double reward,
    int type,
    tPosition posTo,
    double VsRealTo,
    double VsTo2,
    tPosition lastPos,
    bool learnExploredActions)
{
    if(lVerbose)
        printf("tetrisLearn: Called, agentNo (%i)\n", agentNo);
    // isTerminalState describes if posFrom is a terminal state independent of actionFrom

    // TODO: Fix this function for batch learning combined with ensemble learning.
    // In this condition M - 1 agents must wait for the last agent to end his
    // episode before learning, if updateWeightsImmediate == 0
        
    int z, startZ = 0, endZ = 0;
    
    if(pParam->updateWeightsImmediate)
    {
        startZ = agentNo;
        endZ = agentNo + 1;
    }
    else if(agentNo == pParam->agents - 1)
    {
        startZ = 0;
        endZ = pParam->agents;
    }
    else
    {
        // Do not update the weights if agentNo < pParam->agents - 1,
        // instead wait for the last agent in the sequence
        
        startZ = 0;
        endZ = 0;            
    }
    
    if(isTerminalState)
    {
        // Always learn from a terminal state, no matter if this state has been explored or exploited
        // Terminal transition

        // Note: posTo is undefined

        if(lVerbose)
            printf("tetrisLearn: Terminal state, reward (%lf)\n", reward);
        
        tState stateFrom;

        int ret = tdlAddState(agentNo, &stateFrom, NULL, NULL, reward, ALPHA_DISCOUNT);
        if(ret)
        {
            printf("### tetrisLearn: tdlAddState returned error (%i)\n", ret);
            return -4;
        }

        tetrisInputVCodingGet(pTetris, stateFrom.s);

        if(!pParam->batchSize)
        {
            tdlAddStateDone(agentNo, false);
        }
        
        for(z = startZ; z < endZ; z++)
        {
            if(lVerbose)
                printf("tetrisLearn: loop, terminal state, z (%i)\n", z);

            if(!pParam->batchSize)
            {
                double mse = 0;
                int ret = tdlLearn(z, &mse);
                if(ret)
                {
                    printf("### tetrisLearn: tdlLearn returned error (%i).\n", ret);
                    return -5;
                }
//                printf("mse (%lf)\n",mse);
            }
            else
                tdlAddStateDone(z, true);
        }
    }
    else
    {
        if(isExploited || learnExploredActions)
        {
            tState stateFrom;
            tState stateTo;

            if(lVerbose)
                printf("tetrisLearn: Non-terminal state, reward (%lf)\n", reward);

            stateTo.Vs = VsRealTo;

            tState *pStateToFurtherAgents = NULL;
            tState stateToFurtherAgents;
            if(pParam->learnFromAverageStateValues && pParam->agents > 1)
            {
                pStateToFurtherAgents = &stateToFurtherAgents;
                stateToFurtherAgents.Vs = VsTo2;
    //            printf("tetrisLearn: VsTo2 (%lf)\n", VsTo2);
            }

            int ret = tdlAddState(agentNo, &stateFrom, &stateTo, pStateToFurtherAgents, reward, ALPHA_DISCOUNT);
            if(ret)
            {
                printf("### tetrisLearn: tdlAddState returned error (%i)\n", ret);
                return -3;
            }

            tetrisInputVCodingGet(pTetris, stateFrom.s);

            // Set piece in board (given x,y and rotation)
            tetrisApplyPos(pTetris, type, posTo);
            
            tTetris tetris;
            tTetris *pTetrisTmp = pTetris;

            bool fullLines = tetrisCheckFullLines(pTetris);
            
            if(fullLines)
            {
                copyBoard(&tetris, pTetris);

                // Erase the full lines
                int nErasedLines = tetrisEraseLines(&tetris);
                if(lVerbose)
                {
                    if(nErasedLines > 0)
                        printf("tetrisLearn: Cleared (%i) lines\n", nErasedLines);
                }            

                pTetrisTmp = &tetris;                
            }
                        
            // Get feature coding for successor state (posTmp)
            tetrisInputVCodingGet(pTetrisTmp, stateTo.s);

            // Remove piece from board (given x,y and rotation)
            tetrisUnapplyPos(pTetris, type, posTo);
            
            if(!pParam->batchSize)
                tdlAddStateDone(agentNo, false);
        }
        
        if(pParam->batchSize > 0)
        {
            for(z = startZ; z < endZ; z++)
            {
                if(lVerbose)
                    printf("tetrisLearn: loop, non-terminal state, z (%i)\n", z);
                tdlAddStateDone(z, true);
            }            
        }        
    }

    return 0;
}

int tetrisBestState(
    tTetris *pTetris,
    tConfigParam *pParam,
    int agentNo,
    bool forceExploitation,
    bool forceExploration,
    int type,
    unsigned long *pBestStatesLen,
    bool *pIsTerminalState,
    bool *pExploited,
    double *pReward,
    tPosition *pPosTo,
    double *pVsRealTo,
    double *pVsTo2,
    bool isBenchmark,
    unsigned long t2,
    unsigned long maxT,
    bool forceSingleDecisions,
    bool decideWithRewards)
{
    tPosition pos[MAX_NUM_ACTIONS];
    tState stateTo[MAX_NUM_ACTIONS];

    // reward for being in an empty space and not being in a barrier or being outside
    const double reward_none = 0;
    double rewardTmp = reward_none;
    
    // Verify if the agent has reached a terminal state

    tPosition posArr[MAX_NUM_ACTIONS];
    int nPos;

    // Check for completed lines
    bool fullLines = tetrisCheckFullLines(pTetris);
    if(fullLines)
    {
        // In this mode states with completed lines are no longer valid states
        printf("### tetrisBestState: Detected completes lines in a valid state but only board full is a terminal state?\n");
        return -2;
    }
    else
    {
        // Get all available positions

        tetrisGetAllPos(pTetris, type, posArr, &nPos, MAX_NUM_ACTIONS);

        if(lVerbose)
            printf("tetrisBestState: Got (%i) available positions for type (%i)\n", nPos, type);

        if(nPos <= 0)
        {
            // Terminal transition (board is full)
            *pIsTerminalState = true;
            
            // Overwrite the reward
            rewardTmp = 0;
            tetrisRewardGet(pTetris, pParam, &rewardTmp);
            if(lVerbose)
                printf("tetrisBestState: Board is full, override reward by reward function (%lf)\n", rewardTmp);
        }
        else
        {
            // Normal transition
            *pIsTerminalState = false;
        }
    }

    if(*pIsTerminalState)
    {
        // State values: There is no possible successor state in a terminal state

        *pBestStatesLen = 1;
        // *pPosTo is undefined in a terminal state
        pReward[0] = rewardTmp;
        *pExploited = true;

        return 0;
    }

    // Normal transition

    double rewardFrom = 0;
    tetrisRewardGet(pTetris, pParam, &rewardFrom);
    
    if(!forceExploitation && !forceExploration && pParam->tau <= 0)
    {
        double epsilon = pParam->epsilon;
        
        if(lIncreasingEpsilon && !isBenchmark)
        {
            epsilon += (1.0 - pParam->epsilon) * (double) t2 / (double) maxT;
            
            // Keep exploring actions/states with a small probability
            if(epsilon > 0.99)
                epsilon = 0.99;
        }
        
        double myrand = randValDouble (0, 1);
        if (myrand >= epsilon)
           forceExploration = true;
        else
           forceExploitation = true;
    }
    
    if(forceExploration && pVsTo2 == NULL)
    {
        *pBestStatesLen = nPos;
        *pExploited = false;
        unsigned long i;
        for(i = 0; i < *pBestStatesLen; i++)
        {
            pPosTo[i] = posArr[i];
            pReward[i] = rewardFrom; /* same rewards, independent on the chosen after-state s' => r(s) */
        }
        
        return 0;
    }
    
    if(forceExploitation && lNumberListEntriesMax && isBenchmark)
    {
        /* Lookup at a table and try to find the current state.
         * If found, then return the saved values:
         * pBestStatesLen, pReward, pPosTo, pVsTo2 */
        
        /* getting a value */
        
        pReward[0] = rewardFrom; /* same rewards, independent on the chosen after-state s' => r(s) */
        
        if(lookup(pTetris, type, &pPosTo[0]))
        {
            *pBestStatesLen = 1;
            *pExploited = true;
            
            return 0;
        }
    }

    int nStates = 0;

    void *pHandle = NULL;
    if(!decideWithRewards)
    {
        int ret = tdlGetStateValuesPrepare(true, &pHandle, stateTo, MAX_NUM_ACTIONS);
        if(ret)
        {
            printf("### tetrisBestState: tdlGetStateValuesPrepare failed (%i)\n", ret);
            return -1;
        }

        if(pHandle == NULL)
        {
            printf("### tetrisBestState: tdlGetStateValuesPrepare returned pHandle = NULL\n");
            return -1;
        }
    }
    
    double bestReward = -99999999;
    
    int a;
    
    // Iterate all positions
    for(a = 0; a < nPos; a++)
    {
        // Set piece in board (given x,y and rotation)
        tetrisApplyPos(pTetris, type, posArr[a]);
        
        tTetris tetris;
        tTetris *pTetrisTmp = pTetris;

        bool fullLines = tetrisCheckFullLines(pTetris);
        
        if(fullLines)
        {
            copyBoard(&tetris, pTetris);

            // Erase the full lines
            int nErasedLines = tetrisEraseLines(&tetris);
            if(lVerbose)
            {
                if(nErasedLines > 0)
                    printf("tetrisBestState: Cleared (%i) lines\n", nErasedLines);
            }            

            pTetrisTmp = &tetris;                
        }
        
        if(decideWithRewards)
        {
            double r = 0;
            tetrisRewardGet(pTetrisTmp, pParam, &r);

            if(r > bestReward)
            {
                bestReward = r;
                
                pPosTo[0] = posArr[a];
            }
        }
        else
        {
            // Get feature coding for successor state (posTmp)
            tetrisInputVCodingGet(pTetrisTmp, stateTo[nStates].s);
        }
        
        // Remove piece from board (given x,y and rotation)
        tetrisUnapplyPos(pTetris, type, posArr[a]);
        
        pos[nStates] = posArr[a];
        
        nStates++;        
    } /* for a */        
    
    if(decideWithRewards)
    {
        *pExploited = true;
        *pBestStatesLen = 1;
        pReward[0] = bestReward;
    }
    else
    {
        eEnsembleDecision ensembleDecision;

        /* Differences between eEnsembleDecision_single_agent_decision and eEnsembleDecision_no_ensemble:
        * eEnsembleDecision_single_agent_decision returns the correct values for pVsTo2,
        * and must retrieve the values from all agents.
        * eEnsembleDecision_no_ensemble does not and, therefore, needs only to retrieve the values
        * from the current agent */
        
        if(pParam->agents <= 1)
            ensembleDecision = eEnsembleDecision_no_ensemble;
        else
        {
            // pParam->agents > 1
            if(isBenchmark)
            {
                if(pParam->averageDecisionBenchmark)
                    ensembleDecision = eEnsembleDecision_average_state_values_decision;
                else if(pParam->votingDecisionBenchmark)
                    ensembleDecision = eEnsembleDecision_voting_decision;                
                else
                {
    //                ensembleDecision = eEnsembleDecision_single_agent_decision;
                    ensembleDecision = eEnsembleDecision_no_ensemble;
                }
            }
            else
            {
                if(forceSingleDecisions)
                {
                    if(pParam->learnFromAverageStateValues)
                        ensembleDecision = eEnsembleDecision_single_agent_decision;
                    else
                        ensembleDecision = eEnsembleDecision_no_ensemble;
                }
                else if(pParam->averageDecision)
                {
                    if(pParam->weightedDecisions)
                        ensembleDecision = eEnsembleDecision_average_state_values_decision_weighted;
                    else
                        ensembleDecision = eEnsembleDecision_average_state_values_decision;
                }
                else if(pParam->votingDecision)
                {
                    if(pParam->weightedDecisions)
                        ensembleDecision = eEnsembleDecision_voting_decision_weighted;
                    else
                        ensembleDecision = eEnsembleDecision_voting_decision;                
                }
                else
                {
                    if(pParam->learnFromAverageStateValues)
                        ensembleDecision = eEnsembleDecision_single_agent_decision;
                    else
                        ensembleDecision = eEnsembleDecision_no_ensemble;
                }
            }
        }

        eStateDecision stateDecision;
        if(forceExploitation)
        {
            // Forcing exploitation, i.e. be 100% greedy
            stateDecision = eStateDecision_exploitation;
        }
        else if(forceExploration)
        {
            // Forcing exploration, i.e. act 100% randomly
            stateDecision = eStateDecision_exploration;
        }
        else if(pParam->tau > 0)
        {
//            stateDecision = eStateDecision_softmax;
            stateDecision = eStateDecision_softmax_statistics;
        }
        else
        {
            printf("### tetrisBestState: Unknown state decision\n");
            tdlFreeStateValues(pHandle);
            return -4;
        }            

        double VsTo[MAX_NUM_ACTIONS];
        unsigned long bestStatesIndices[MAX_NUM_ACTIONS];
        *pBestStatesLen = 0;

        // Select the best Q(s,a) independent of the reward that will be received in the new state and the new state s'
        int ret = tdlGetBestStateEnsemble(pHandle, agentNo, ensembleDecision, stateTo, NULL, nStates, stateDecision, pParam->tau, 0, pExploited, bestStatesIndices, pBestStatesLen, VsTo, pVsTo2, pVsRealTo, NULL, NULL, false);
        if(ret)
        {
            printf("### tetrisBestState: tdlGetBestStateEnsemble failed (%i)\n", ret);

            tdlFreeStateValues(pHandle);

            return -2;
        }

        if(*pBestStatesLen > nStates)
        {
            printf("### tetrisBestState: *pBestStatesLen > nStates?\n");

            tdlFreeStateValues(pHandle);
            
            return -3;
        }

        unsigned long i;
        for(i = 0; i < *pBestStatesLen; i++)
        {
            int index = bestStatesIndices[i];
            pPosTo[i] = pos[index];
            pReward[i] = rewardFrom; /* same rewards, independent on the chosen after-state s' => r(s)*/                        
    //        printf("tetrisBestState: (%li) = (%i)\n", i, index);
        }
        
        if(forceExploitation && lNumberListEntriesMax && isBenchmark && *pBestStatesLen == 1)
        {
            /* Remove the oldest entry from the table of saved states
             * and add a new state with the values:
             * pBestStatesLen, pReward, pPosTo, pVsTo2 */
            
            store(pTetris, type, pPosTo[0]);
        }
            
        tdlFreeStateValues(pHandle);
    }
    
    return 0;
}

int addStateToRepo(
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,
    double probToAddState,
    int *pNStatesAdded,
    tTetris *pTetris,
    int type,
    double reward,
    bool isTerminalState)
{    
    if(pRepo != NULL)
    {
        if((pRepo->nStateRepoMax > 0) &&
            (pRepo->nStateRepo < pRepo->nStateRepoMax))
        {
            if(isTerminalState)
            {
                // In the current view, terminal states shall not be added
                if(lVerbose)
                    printf("addStateToRepo: Not adding terminal state\n");
                
                return 0;
            }
            
            // drawing random values using drand48 doesn't affect the sequence of random numbers
            // drawn by using the rand / random functions
            
            bool add = false;
            if(probToAddState == 1.0)
                add = true;
            else if(drand48() < probToAddState)
                add = true;
            
            if(add)
            {
                *pNStatesAdded = *pNStatesAdded + 1;                
                
                // Search, if the state is already present
                
                bool present = false;
                unsigned long i = 0;
                for(i = 0; i < pRepo->nStateRepo; i++)
                {
                    // A state consists of a tetris board and a piece type
                    
                    // Examine piece type
                    
                    if(pRepo->pStateInternalRepo[i].type != type)
                        continue;

                    // Examine Tetris Board

                    if(pRepo->pStateInternalRepo[i].tetris.maxHeight != pTetris->maxHeight)
                        continue;

                    // TODO: Remove the following two lines
                    if(pRepo->pStateInternalRepo[i].tetris.maxHeightLast != pTetris->maxHeightLast)
                        continue;
                    
                    if(memcmp(&pRepo->pStateInternalRepo[i].tetris.board, &pTetris->board, sizeof(pTetris->board)))
                        continue;
                    
                    present = true;
                    break; /* found state */                    
                }
                
                if(present)
                {
                    // Update p(s), leave function
                    
                    pStateProbs[i] ++;
                    
                    return 0;
                }
                
                pStateProbs[pRepo->nStateRepo] = 1;
                
                if(pNActions != NULL)
                {
                    if(isTerminalState)
                        pNActions[i] = 1;
                    else
                    {
                        // Get number of actions = all available positions

                        tPosition posArr[MAX_NUM_ACTIONS];
                        int nPos;
                        
                        tetrisGetAllPos(pTetris, type, posArr, &nPos, MAX_NUM_ACTIONS);

                        if(lVerbose)
                            printf("addStateToRepo: Got (%i) available positions for type (%i)\n", nPos, type);
                                                
                        pNActions[i] = nPos;
                    }
                }
                
                if(lVerbose)
                {
                    printf("Adding following tetris field to from-state:\n");
                    printf("type (%i), isTerminalState (%i),  reward (%lf)\n", type, isTerminalState, reward);
                    outputBoard(pTetris, BOARD_HEIGHT - 1, 0);
                }
                
                // ... and add the state to the repository

                pRepo->pStateInternalRepo[pRepo->nStateRepo].tetris = *pTetris;

                // Select piece randomly (stochastic SZ-Tetris)
                pRepo->pStateInternalRepo[pRepo->nStateRepo].type = type;
                
                pRepo->nStateRepo++;
            }
        }
        return 0;
    }
    
    return -1;
}

int performBenchmark(
    tConfigParam *pParam,
    int agentNo,
    int testruns,
    double *pScore,
    double *pScoreMin,
    double *pScoreMax,
    double *pReward,
    double *pEpisodeLen,
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,
    double probToAddState,
    bool decideWithRewards,
    bool deterministicEnvironment,
    int nStatesPerAgentMax)
{
    tTetris tetris;

    *pScore = 0;
    *pReward = 0;
    *pEpisodeLen = 0;

    unsigned long nEpisodes = 0;

    int nStatesAdded = 0;    
    
    *pScoreMin = INFINITY;
    *pScoreMax = -INFINITY;
    
    int t;
    
    for(t = 0; t < testruns; t++)
    {
        // Erase board
        tetrisEraseBoard(&tetris);

        int t2, t3;
        int type = 0;
        bool terminalStateReached = false;
        double score = 0;

        for(t2 = 0, t3 = 0; !terminalStateReached; t2++, t3++)
        {
            if(lPieceSequenceLen > 0)
            {
                // Select piece from database (were initialized randomly, stochastic SZ-Tetris)

                if(t3 > lPieceSequenceLen)
                {
                    printf("### performBenchmark: t3 (%i) exceeded (%i)\n", t3, lPieceSequenceLen);
                    return -1;
                }
                
                type = (lPieceSequence[t][t3 / 8] >> (t3 % 8)) & 0x01;
                
                if(type != 0 && type != 1)
                {
                    printf("### performBenchmark: wrong value in piece sequence\n");
                    return -1;
                }
            }
            else if(deterministicEnvironment)
            {
                // Piece alternates (deterministic SZ-Tetris)

                type = (type + 1) % 2;                    
            }
            else
            {
                // Select piece randomly (stochastic SZ-Tetris)
                
                type = randValLong(0, 1);
            }

            bool isTerminalState = false;
            bool exploited = false;
            double rewardArr[MAX_NUM_ACTIONS];
            tPosition posArr[MAX_NUM_ACTIONS];
            double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
            unsigned long bestStatesIndicesLen = 0;

            int ret = tetrisBestState(&tetris, pParam, agentNo, true, false, type, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, posArr, VsRealToArr, VsTo2Arr, true, 0, 0, 0, decideWithRewards);

            if(ret)
            {
                printf("### performBenchmark: tetrisBestState failed (%i)\n", ret);
                return -1;
            }

            if(!bestStatesIndicesLen)
            {
                printf("### bestStatesIndicesLen is Zero\n");
                break;
            }

            // Add all states, including the empty board (t2 == 0)
            
            addStateToRepo(pRepo, pStateProbs, pNActions, probToAddState, &nStatesAdded, &tetris, type, rewardArr[0], isTerminalState);
            
            if(pRepo != NULL && nStatesPerAgentMax && nStatesAdded >= nStatesPerAgentMax)
            {
                // We have collected enough states, leave the benchmark immediately (invalid scores)
                
                printf("Agent (%i) has collected (%i) states with run (%i), aborting benchmark\n", agentNo, nStatesAdded, t);
                
                *pScore = 0;
                *pScoreMin = INFINITY;
                *pScoreMax = -INFINITY;
                *pReward = 0;
                
                return 0;
            }
            
            tPosition pos;                
            pos.x = -1;
            pos.y = -1;
            pos.isHorizontal = 0;                
            
            *pReward = *pReward + pow(pParam->gamma, (double) t3) * rewardArr[0];
            
            if(isTerminalState)
            {
                nEpisodes++;

                *pEpisodeLen = *pEpisodeLen + t3;

                // Give up if board is full, continue otherwise
                if(lVerbose)
                {
                    printf("performBenchmark: Board full at t2 = %i\n", t2);
                    outputBoard(&tetris, BOARD_HEIGHT - 1, 0);
                }
                
                terminalStateReached = true;
                                        
                break; /* t2 */
            }
            else
            {
                // Randomly select one of the best actions
                unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                pos = posArr[myrand];

                ret = tetrisApplyPos(&tetris, type, pos);
                if(ret)
                {
                    printf("### tetrisApplyPos failed (%i)\n", ret);

                    printf("Tried to apply x (%i), y (%i), horizontal (%i)\n", pos.x, pos.y, pos.isHorizontal);
                    
                    tPosition posTmp[20];
                    int nPos = 0;
                    
                    tetrisGetAllPos(&tetris, type, posTmp, &nPos, 20);

                    printf("Have (%i) available positions:\n", nPos);
                    int i;
                    for(i = 0; i < nPos; i++)
                        printf("x (%i), y (%i), horizontal (%i)\n", posTmp[i].x, posTmp[i].y, posTmp[i].isHorizontal);
                    
                    return -3;
                }
                
                // Erase the full lines
                int nErasedLines = tetrisEraseLines(&tetris);

                score += nErasedLines;

                if(lVerbose)
                {
                    printf("performBenchmark: (best) actions (%li), piece (%i), pos.x (%i), pos.y (%i), isHorizontal (%i)\n", bestStatesIndicesLen, type, posArr[myrand].x, posArr[myrand].y, posArr[myrand].isHorizontal);
                    outputBoard(&tetris, BOARD_HEIGHT - 1, 0);
                }
            }            
        } /* t2 */
        
        *pScore = *pScore + score;
        if(score < *pScoreMin)
            *pScoreMin = score;
        if(score > *pScoreMax)
            *pScoreMax = score;            
    } /* for t */        
    
    *pScore = *pScore / (double) testruns;
    *pReward = *pReward / (double) testruns;
    *pEpisodeLen = *pEpisodeLen / (double) testruns;

    return 0;
}

static int calculateConsistencies(
    tConfigParam *pParam,
    int agentNo,
    char *repoErrorsFile,
    char *repoActionNoFile,
    tStateInternal *pState,
    int nStates,                         
    bool forceExploitation,
    int bestActions,
    bool allActions,
    bool singleActions,
    bool ensembleActions,    
    int retries,
    bool verbose
    )
{
    unsigned i;
    double *yPredicted = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);
    double *yReal = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);
    int *actionNumber = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);

    if(ensembleActions)
    {
        printf("### calculateConsistencies: ensembleActions are not supported\n");
        free(actionNumber);
        free(yPredicted);
        free(yReal);
        return -1;
    }
    
    int nStatesReal = 0;
    
    tState state;
    void *pHandle;
    if(tdlGetStateValuesPrepare(false, &pHandle, &state, 1))
    {
        printf("### calculateConsistencies: tdlGetStateValuesPrepare function returned error.\n");
        free(actionNumber);
        free(yPredicted);
        free(yReal);
        return -4;
    }
                
    if(pHandle == NULL)
    {
        printf("### calculateConsistencies: tdlGetStateValuesPrepare function returned NULL handle\n");
        free(actionNumber);
        free(yPredicted);
        free(yReal);
        return -4;                
    }

    for(i = 0; i < nStates; i++)
    {
        tTetris tetris = pState[i].tetris;
        int type = pState[i].type;

        if(verbose)
        {
            printf("Getting state (%i) from repo\n", i);
            printf("Outputting state:\n");
            outputBoard(&tetris, BOARD_HEIGHT - 1, 0);
            printf("Type: (%i)\n", type);
        }
        
        tPosition posArrReal[MAX_NUM_ACTIONS];
        int nPosReal;
        double V[MAX_NUM_ACTIONS];
        
        /* Get all actions */
        tetrisGetAllPos(&tetris, type, posArrReal, &nPosReal, MAX_NUM_ACTIONS);

        if(verbose)
            printf("calculateConsistencies: Got (%i) available positions for type (%i)\n", nPosReal, type);
        
        int nPos = nPosReal;
        tPosition posArr2[nPos];

        int j;
        for(j = 0; j < nPos; j++)
            posArr2[j] = posArrReal[j];
        
//        memcpy(posArr2, posArrReal, sizeof(tPosition) * nPos);
        
        if(bestActions > 0 || allActions)
        {
            // Iterate all or the best X actions
                        
            if(verbose)
                printf("calculateConsistencies: Iterate the best (%i) actions\n", bestActions);
             
            if(bestActions > 0)
            {
                int a;
                for(a = 0; a < nPos; a++)
                {
                    // Set piece in board (given x,y and rotation)
                    int ret = tetrisApplyPos(&tetris, type, posArr2[a]);
                    if(ret)
                    {
                        printf("### tetrisApplyPos failed (%i)\n", ret);
                        free(actionNumber);
                        free(yPredicted);
                        free(yReal);
                        return -5;
                    }
                    
                    tTetris tetris2;
                    tTetris *pTetrisTmp = &tetris;

                    bool fullLines = tetrisCheckFullLines(&tetris);
                    
                    if(fullLines)
                    {
                        copyBoard(&tetris2, &tetris);

                        // Erase the full lines
                        int nErasedLines = tetrisEraseLines(&tetris2);
                        if(lVerbose)
                        {
                            if(nErasedLines > 0)
                                printf("tetrisLearn: Cleared (%i) lines\n", nErasedLines);
                        }            

                        pTetrisTmp = &tetris2;                
                    }
                                        
                    // Get feature coding for the state
                    tetrisInputVCodingGet(pTetrisTmp, state.s);
                    
                    ret = tdlGetStateValues(pHandle, agentNo, &state, 1);
                    if(ret)
                    {
                        printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                        free(actionNumber);
                        free(yPredicted);
                        free(yReal);
                        return -4;
                    }
                    
                    V[a] = state.Vs;                
                    
                    // Remove piece from board (given x,y and rotation)
                    tetrisUnapplyPos(&tetris, type, posArr2[a]);
                } /* for nPos */

                int indices[nPos];
                int rank[nPos];

                bubbleRank(V, nPos, true, indices, rank);
                
                if(bestActions < nPos)
                    nPos = bestActions;
                
                tPosition posArrTmp[nPos];
                double Vtmp[nPos];
                                                
                int j;
                for(j = 0; j < nPos; j++)
                {
                    // indices[0] has the index of the value with the highest value,
                    // indices[nPos] has the index of the value with the lowest value
                    
                    posArrTmp[j] = posArr2[indices[j]];
                    Vtmp[j] = V[indices[j]];
                }
                
                for(j = 0; j < nPos; j++)
                {
                    posArr2[j] = posArrTmp[j];
                    V[j] = Vtmp[j];
                }
            }
        }
        else
            nPos = 1;
        
        typedef struct
        {
            tPosition pos;
            int type;
            double value;
        } tValues;
        
        int a;
        for(a = 0; a < nPos; a++)
        {
            double vsFrom;
            double vsTo = 0;
            double reward = 0;
            
            tValues valRepo[2 * MAX_NUM_ACTIONS];
            int nValsRepo = 0;            
            
            bool isTerminalState = false;
            bool exploited = false;
            double rewardArr[MAX_NUM_ACTIONS];
            tPosition posArr[MAX_NUM_ACTIONS];
            double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
            unsigned long bestStatesIndicesLen = 0;
            
            if(a > 0)
            {
                // Revert changes from last time
                tetris = pState[i].tetris;                
            }
            
            tPosition pos;
            
            if(bestActions > 0 || allActions)
            {
                // Iterate all (valid) actions
                pos = posArr2[a];
            }
            else
            {
                // Get the best action a ( = pos) in the current state s (force exploitation / greedy action selection)

                int ret = tetrisBestState(&tetris, pParam, agentNo, true, false, type, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, posArr, VsRealToArr, VsTo2Arr, true, 0, 0, 0, false);
                
                if(ret)
                {
                    printf("### calculateConsistencies: tetrisBestState failed (%i)\n", ret);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    tdlFreeStateValues(pHandle);            
                    return -2;
                }
                
                if(!bestStatesIndicesLen)
                {
                    printf("### calculateConsistencies: bestStatesIndicesLen is Zero\n");
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    tdlFreeStateValues(pHandle);            
                    return -3;
                }

                if(isTerminalState)
                {
                    printf("### calculateConsistencies: got terminal state, must not happen\n");
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    tdlFreeStateValues(pHandle);            
                    return -4;
                }
                
                // Randomly select one of the best actions
                unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                pos = posArr[myrand];
            }
                
            if(verbose)
                printf("Non-terminal state s: Best position (%i / %i)\n", pos.x, pos.y);
            
            int ret = tetrisApplyPos(&tetris, type, pos);
            if(ret)
            {
                printf("### tetrisApplyPos failed (%i)\n", ret);
                free(actionNumber);
                free(yPredicted);
                free(yReal);
                return -5;
            }
            
            // Erase the full lines                
            tetrisEraseLines(&tetris);
            
            // Get the reward
            
            tetrisRewardGet(&tetris, pParam, &reward);            
            
            if(bestActions > 0)
            {
                vsFrom = V[a];
            }
            else
            {
                // Get the (average) state-value of the from-State

                tetrisInputVCodingGet(&tetris, state.s);

                ret = tdlGetStateValues(pHandle, agentNo, &state, 1);
                if(ret)
                {
                    printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -4;
                }
                
                vsFrom = state.Vs;
            }
            
            tTetris tetrisBak = tetris;
            
            tPosition bestStateType0;
            tPosition bestStateType1;            
            
            bool bestStateType0Known = false;
            bool bestStateType1Known = false;
            
            int r;
            for(r = 0; r < retries; r++)
            {
                if(r > 0)
                {
                    // Revert changes from last time
                    tetris = tetrisBak;
                }
                
                // Get the successor state s' due to action a in state s

                int type2 = randValLong(0,1);
                
                if(verbose)
                    printf("Simulated action. New piece: (%i)\n", type2);
                
                bool selectActionRandomly = false;
                bool exploit = true;
                if(!forceExploitation)
                {
                    if(pParam->tau <= 0)
                    {
                        double myrand = randValDouble (0, 1);
                        if (myrand >= pParam->epsilon)
                        {
                            /* Randomly select one of the actions */
                            selectActionRandomly = true;
                            
                            exploit = false;
                            
                            if(verbose)
                                printf("calculateConsistencies: Epsilon-greedy, select action randomly\n");
                        }
                    }
                    else
                        exploit = false;
                }
                                
                tPosition pos2;
                
                if(selectActionRandomly)
                {
                    tPosition posArr3[MAX_NUM_ACTIONS];
                    int nPos3;
                    
                    tetrisGetAllPos(&tetris, type2, posArr3, &nPos3, MAX_NUM_ACTIONS);

                    if(verbose)
                        printf("calculateConsistencies: 2 Iterate all actions, got (%i) available positions for type (%i)\n", nPos3, type2);
                    
                    if(!nPos3)
                    {
                        // Terminal state observed
                        
                        if(verbose)
                            printf("Terminal state #2\n");
                        
                        break; // for r
                    }
                    
                    unsigned long myrand = randValLong(0, nPos3 - 1);
                    
                    pos2 = posArr3[myrand];                    
                }
                else if(type2 == 0 && bestStateType0Known && exploit)
                    pos2 = bestStateType0;
                else if(type2 == 1 && bestStateType1Known && exploit)
                    pos2 = bestStateType1;
                else
                {
                    // Get the best action a in the current state s

                    exploited = false;
                    isTerminalState = false;
                    bestStatesIndicesLen = 0;

                    int ret = tetrisBestState(&tetris, pParam, agentNo, exploit, false, type2, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, posArr, VsRealToArr, VsTo2Arr, true, 0, 0, 0, false);

                    if(ret)
                    {
                        printf("### calculateConsistencies: tetrisBestState failed (%i)\n", ret);
                        free(actionNumber);
                        free(yPredicted);
                        free(yReal);
                        tdlFreeStateValues(pHandle);            
                        return -2;
                    }

                    if(!bestStatesIndicesLen)
                    {
                        printf("### calculateConsistencies: bestStatesIndicesLen is Zero\n");
                        free(actionNumber);
                        free(yPredicted);
                        free(yReal);
                        tdlFreeStateValues(pHandle);            
                        return -3;
                    }
                    
                    if(isTerminalState)
                    {
                        if(verbose)
                            printf("Terminal state\n");
                        
                        break; // for r
                    }
                    
                    // Randomly select one of the best actions
                    unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                    pos2 = posArr[myrand];
                    
                    if(verbose)
                        printf("Non-terminal state s: Best position2 (%i / %i)\n", pos2.x, pos2.y);
                    
                    if(type2 == 0)
                    {
                        bestStateType0 = pos2;
                        bestStateType0Known = true;
                    }
                    else
                    {
                        bestStateType1 = pos2;
                        bestStateType1Known = true;
                    }
                }
                
                ret = tetrisApplyPos(&tetris, type2, pos2);
                if(ret)
                {
                    printf("### tetrisApplyPos failed (%i)\n", ret);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -5;
                }
                
                // Erase the full lines                
                tetrisEraseLines(&tetris);
                
                // Get the (average) state-value of the to-State                    
                
                int k;
                bool found = false;
                for(k = 0; k < nValsRepo; k++)
                {
                    if(valRepo[k].type != type2)
                        continue;

                    if(memcmp(&valRepo[k].pos, &pos2, sizeof(pos2)))
                        continue;

                    found = true;
                    break;
                }
                    
                if(found)
                    state.Vs = valRepo[k].value;
                else
                {
                    tetrisInputVCodingGet(&tetris, state.s);
                                                    
                    int ret = tdlGetStateValues(pHandle, agentNo, &state, 1);
                    if(ret)
                    {
                        printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                        free(actionNumber);
                        free(yPredicted);
                        free(yReal);
                        tdlFreeStateValues(pHandle);            
                        return -4;
                    }
                    
                    valRepo[nValsRepo].value = state.Vs;
                    valRepo[nValsRepo].type = type2;
                    valRepo[nValsRepo].pos = pos2;
                    
                    nValsRepo++;
                }
                
                vsTo += (state.Vs / (double) retries);
            } /* for r */
        
            double V_s_successor = reward + pParam->gamma * vsTo;
            
            yPredicted[nStatesReal] = vsFrom;
            yReal[nStatesReal] = V_s_successor;
            
            if(allActions)
                actionNumber[nStatesReal] = a;
            else
            {
                int a2;
                int found = 0;
                for(a2 = 0; a2 < nPosReal; a2++)
                {
                    if(!memcmp(&posArrReal[a2], &pos, sizeof(tPosition)))
                    {
                        found = 1;
                        break;
                    }
                }
                
                if(!found)
                {
                    printf("### calculateConsistencies: Were not able to find action (%i)\n", a);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    tdlFreeStateValues(pHandle);            
                    return -4;
                }
                
                actionNumber[nStatesReal] = a2;                
            }
            
            if(verbose)            
                printf("calculateConsistencies: gamma (%lf), reward (%lf), V(s) = %lf, V(s') = %lf, delta(s') = %lf\n", pParam->gamma, reward, vsFrom, vsTo, V_s_successor);
            
            nStatesReal++;
        } /* for a */
    } /* for i */
    
    tdlFreeStateValues(pHandle);

    if(verbose)
        printf("calculateConsistencies: Have (%i) states\n", nStatesReal);
    
    // Write errors to file
    
    FILE *fp = fopen (repoErrorsFile, "w");
    if(fp == NULL)
    {
        printf("### fopen failed with file (%s)\n", repoErrorsFile);
        
        free(actionNumber);
        free(yPredicted);
        free(yReal);        
        return -6;            
    }
    
    for(i = 0; i < nStatesReal; i++)
    {
        char str[100];
        snprintf(str, 100, "%lf\n", yPredicted[i]);
        fwrite(str, sizeof(char), strlen(str), fp);
        
        snprintf(str, 100, "%lf\n", yReal[i]);
        fwrite(str, sizeof(char), strlen(str), fp);        
    }
    
    fclose(fp);

    free(yPredicted);
    free(yReal);        

    printf("Written (%i) errors to file (%s)\n", nStatesReal, repoErrorsFile);

    // Write action numbers to file
    
    fp = fopen (repoActionNoFile, "w");
    if(fp == NULL)
    {
        printf("### fopen failed with file (%s)\n", repoActionNoFile);
        
        free(actionNumber);
        return -6;            
    }
    
    for(i = 0; i < nStatesReal; i++)
    {
        char str[100];
        snprintf(str, 100, "%i\n", actionNumber[i]);
        fwrite(str, sizeof(char), strlen(str), fp);
    }
    
    fclose(fp);
    
    free(actionNumber);
    
    printf("Written (%i) action numbers to file (%s)\n", nStatesReal, repoActionNoFile);
    
    return 0;
}

int performBenchmarks(
    tConfigParam *pParam,
    int retries,
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,
    double probToAddState,
    bool decideWithRewards,
    bool deterministicEnvironment,
    int nStatesPerAgentMax
)
{
    int z2, endZ3;

    if(pParam->averageDecisionBenchmark || pParam->votingDecisionBenchmark)
        endZ3 = 1;
    else
        endZ3 = pParam->agents;
    
    double averageScore = 0;
    double averageTotalReward = 0;
    double averageEpisodeLen = 0;
    double minScore = INFINITY;
    double maxScore = -INFINITY;
#ifdef BENCHMARK_DIFFERENCES
    double averageScore2 = 0;
    double averageTotalReward2 = 0;
    double averageEpisodeLen2 = 0;
#endif /* BENCHMARK_DIFFERENCES */

    printf("Perform benchmark for (%i) agents\n", endZ3);
    
    // Iterate all agents
    for(z2 = 0; z2 < endZ3; z2++)
    {
        double score = 0;
        double scoreMinTmp = 0;
        double scoreMaxTmp = 0;
        double reward = 0;
        double episodeLen = 0;

        int ret = performBenchmark(pParam, z2, retries, &score, &scoreMinTmp, &scoreMaxTmp, &reward, &episodeLen, pRepo, pStateProbs, pNActions, probToAddState, decideWithRewards, deterministicEnvironment, nStatesPerAgentMax);
        if(ret)
        {
            printf("### performBenchmark returned error (%i)\n", ret);
            return -1;
        }

        averageScore += score;
        averageTotalReward += reward;
        averageEpisodeLen += episodeLen;

        if(scoreMinTmp < minScore)
            minScore = scoreMinTmp;
        if(scoreMaxTmp > maxScore)
            maxScore = scoreMaxTmp;
        
#ifdef BENCHMARK_DIFFERENCES
        // Repeat the benchmark to get the differences
        ret = performBenchmark(pParam, z2, retries, &score, &scoreMinTmp, &scoreMaxTmp, &reward, &episodeLen, NULL, NULL, NULL, NULL, 0, deterministicEnvironment, 0);
        if(ret)
        {
            printf("### performBenchmark returned error (%i)\n", ret);
            return -2;
        }

        averageScore2 += score;
        averageTotalReward2 += reward;
        averageEpisodeLen2 += episodeLen;
#endif /* BENCHMARK_DIFFERENCES */
    } /* for z2 */

    averageScore /= (double) endZ3;
    averageTotalReward /= (double) endZ3;
    averageEpisodeLen /= (double) endZ3;
#ifdef BENCHMARK_DIFFERENCES
    averageScore2 /= (double) endZ3;
    averageTotalReward2 /= (double) endZ3;
    averageEpisodeLen2 /= (double) endZ3;
#endif /* BENCHMARK_DIFFERENCES */

    printf ("average score = %lf, min score = %lf, max score = %lf, average total reward = %lf, average episode len = %lf\n", averageScore, minScore, maxScore, averageTotalReward, averageEpisodeLen);
#ifdef BENCHMARK_DIFFERENCES
    printf ("diff score = %lf, diff total reward = %lf, diff episode len = %lf\n", fabs(averageScore-averageScore2), fabs(averageTotalReward-averageTotalReward2), fabs(averageEpisodeLen - averageEpisodeLen2));
#endif /* BENCHMARK_DIFFERENCES */
    
    return 0;    
}

int main (
    int argc,
    char **argv)
{
    // Initialize piece set
    memcpy(lPieceSet[0].pieceHorizontal.bitmap, sPieceHorizontal, 3*3);
    lPieceSet[0].pieceHorizontal.height = 2;
    lPieceSet[0].pieceHorizontal.width = 3;

    memcpy(lPieceSet[0].pieceVertical.bitmap, sPieceVertical, 3*3);
    lPieceSet[0].pieceVertical.height = 3;
    lPieceSet[0].pieceVertical.width = 2;

    memcpy(lPieceSet[1].pieceHorizontal.bitmap, zPieceHorizontal, 3*3);
    lPieceSet[1].pieceHorizontal.height = 2;
    lPieceSet[1].pieceHorizontal.width = 3;

    memcpy(lPieceSet[1].pieceVertical.bitmap, zPieceVertical, 3*3);
    lPieceSet[1].pieceVertical.height = 3;
    lPieceSet[1].pieceVertical.width = 2;

    char tetrisConfFile[100] = "tetris.conf";

    bool verbose = false;
    int retries = 100;
    int iterationsPerAgent = 1;
    bool benchmark = false;
    char savfile[100];
    bool createInitWeights = false;
    unsigned int seed = 0;
    int nStatesRepo = 0;
    int nStatesRepoEvaluate = 0;
    tRepo *pRepo = NULL;
    unsigned long *pStateProbs = NULL;
    unsigned long *pNActions = NULL;
    char repoFile[100] = "tetrisStateRepo";
    char repoValuesFile[100] = "tetrisStateRepoValues";
    char repoErrorsFile[100] = "tetrisStateRepoErrors";
    char repoActionNoFile[100] = "tetrisStateRepoActionNumbers";
    char repoFileStateProbs[100] = "tetrisStateRepoProbs";
    char repoFileNumberActions[100] = "tetrisStateRepoNumberActions";    
    bool calcConsistencies = false;
    char pieceSequenceFile[100] = "tetrisPieceSeq";
    bool createPieceSequence = false;
    bool loadPieceSequence = false;
    double decisionWeight[MAX_MLPS];
    int nDecisionWeights = 0;
    double probToAddState = 0.5;
    int minFrequencyState = 1;
    int pieceSequenceLen = MAX_PIECE_SEQUENCE;
    bool learnExploredActions = false;
    bool decreasingDecisionWeight = false;
    unsigned long itOff = 0;
    unsigned long itMax = 0;
    double jointDecisionsEpsilon = 0;
    bool forceExploitationConsistencies = false;
    bool allActions = false;
    bool singleActions = false;
    bool ensembleActions = false;
    int bestActions = 0;
    bool decideWithRewards = false;
    int rewardFunc = lRewardFunc;
    unsigned long lookupTable = 0;
    bool deterministicEnvironment = false;
    int nStatesPerAgentMax = 0;
    
    int j;
    for(j = 0; j < MAX_MLPS; j++)
        decisionWeight[j] = 0;
    
    strcpy(savfile, "tetrisMlpSav");
    
    for (j = 0; j < argc; j++)
    {
        if (strcmp (argv[j], "--conf") == 0)
            sscanf (argv[j + 1], "%s", tetrisConfFile);

        if (strcmp (argv[j], "--verbose") == 0)
            verbose = true;

        if (strcmp (argv[j], "--retries") == 0)
            sscanf (argv[j + 1], "%i", &retries);

        if (strcmp (argv[j], "--iterationsPerAgent") == 0)
            sscanf (argv[j + 1], "%i", &iterationsPerAgent);

        if (strcmp (argv[j], "--benchmark") == 0)
            benchmark = true;
        
        if (strcmp (argv[j], "--savfile") == 0)
            sscanf (argv[j + 1], "%s", savfile);

        if (strcmp (argv[j], "--createInitWeights") == 0)
            createInitWeights = true;
        
        if (strcmp (argv[j], "--seed") == 0)
            sscanf (argv[j + 1], "%i", &seed);

        if (strcmp (argv[j], "--createStateRepo") == 0)
            sscanf (argv[j + 1], "%i", &nStatesRepo);
        
        if (strcmp (argv[j], "--evaluateStateRepo") == 0)
            sscanf (argv[j + 1], "%i", &nStatesRepoEvaluate);

        if (strcmp (argv[j], "--calcConsistencies") == 0)
            calcConsistencies = true;

        if (strcmp (argv[j], "--repoFile") == 0)
            sscanf (argv[j + 1], "%s", repoFile);
        
        if (strcmp (argv[j], "--repoFileStateProbs") == 0)
            sscanf (argv[j + 1], "%s", repoFileStateProbs);
        
        if (strcmp (argv[j], "--repoFileNumberActions") == 0)
            sscanf (argv[j + 1], "%s", repoFileNumberActions);

        if (strcmp (argv[j], "--repoValuesFile") == 0)
            sscanf (argv[j + 1], "%s", repoValuesFile);
        
        if (strcmp (argv[j], "--repoErrorsFile") == 0)
            sscanf (argv[j + 1], "%s", repoErrorsFile);
        
        if (strcmp (argv[j], "--repoActionNumbersFile") == 0)
            sscanf (argv[j + 1], "%s", repoActionNoFile);

        if (strcmp (argv[j], "--pieceSequenceFile") == 0)
            sscanf (argv[j + 1], "%s", pieceSequenceFile);
        
        if (strcmp (argv[j], "--createPieceSequence") == 0)
            createPieceSequence = true;
        
        if (strcmp (argv[j], "--loadPieceSequence") == 0)
            loadPieceSequence = true;
        
        if (strcmp (argv[j], "--decisionWeights") == 0)
        {
            sscanf (argv[j + 1], "%i", &nDecisionWeights);
            unsigned i;
            for(i = 1; i <= nDecisionWeights; i++)
            {
                sscanf (argv[j + i + 1], "%lf", &decisionWeight[i - 1]);
            }
        }
        
        if(strcmp (argv[j], "--probToAddState") == 0)
            sscanf (argv[j + 1], "%lf", &probToAddState);

        if(strcmp (argv[j], "--minFrequencyState") == 0)
            sscanf (argv[j + 1], "%i", &minFrequencyState);
        
        if(strcmp (argv[j], "--nStatesPerAgentMax") == 0)
            sscanf (argv[j + 1], "%i", &nStatesPerAgentMax);

        if(strcmp (argv[j], "--learnExploredActions") == 0)
            learnExploredActions = true;
        
        if(strcmp (argv[j], "--decreasingDecisionWeight") == 0)
            decreasingDecisionWeight = true;

        if(strcmp (argv[j], "--itOff") == 0)
            sscanf (argv[j + 1], "%li", &itOff);
        
        if(strcmp (argv[j], "--itMax") == 0)
            sscanf (argv[j + 1], "%li", &itMax);

        if(strcmp (argv[j], "--jointDecisionsEpsilon") == 0)
            sscanf (argv[j + 1], "%lf", &jointDecisionsEpsilon);
        
        if(strcmp (argv[j], "--forceExploitationConsistencies") == 0)
            forceExploitationConsistencies = true;
        
        if(strcmp (argv[j], "--allActions") == 0)
            allActions = true;

        if(strcmp (argv[j], "--singleActions") == 0)
            singleActions = true;
        
        if(strcmp (argv[j], "--ensembleActions") == 0)
            ensembleActions = true;
        
        if(strcmp (argv[j], "--bestActions") == 0)
            sscanf (argv[j + 1], "%i", &bestActions);
        
        if(strcmp (argv[j], "--decideWithRewards") == 0)
            decideWithRewards = true;
        
        if(strcmp (argv[j], "--rewardFunc") == 0)
            sscanf (argv[j + 1], "%i", &rewardFunc);
        
        if(strcmp (argv[j], "--lookupTable") == 0)
            sscanf (argv[j + 1], "%li", &lookupTable);
        
        if(strcmp (argv[j], "--deterministicEnvironment") == 0)
            deterministicEnvironment = true;
    }

    if(createPieceSequence)
    {
        printf("Creating piece sequence, writing to file (%s), and exiting\n", pieceSequenceFile);
        
        printf("Each sequence has a length of (%i) and we generate (%i) sequences\n", pieceSequenceLen, retries);
        
        int i;

        // Write piece sequence to file
        
        FILE *fp;

        fp = fopen (pieceSequenceFile, "w");
        if(fp == NULL)
        {
            printf("### fopen failed with file (%s)\n", pieceSequenceFile);
            
            return -6;            
        }

        allocateMatrixType2((void ***) &lPieceSequence, sizeof(double), retries, pieceSequenceLen / 8);
                
        int r;
        for(r = 0; r < retries; r++)
        {
            for(i = 0; i < pieceSequenceLen / 8; i++)
                lPieceSequence[0][i] = 0;
            
            for(i = 0; i < pieceSequenceLen; i++)
            {
                uint8_t piece = (uint8_t) randValLong(0,1);

                lPieceSequence[0][i / 8] |= (piece << (i % 8));
            }

            for(i = 0; i < pieceSequenceLen / 8; i++)
            {
                if(fwrite((void *) &lPieceSequence[0][i], sizeof(uint8_t), 1, fp) != 1)
                {
                    freeMatrixType2((void **) lPieceSequence, retries);
                    
                    fclose(fp);

                    printf("### fwrite failed (2) \n");
                    
                    return -7;
                }
            }
        }
        
        freeMatrixType2((void **) lPieceSequence, retries);

        fclose(fp);
        
        return 0;
    }
    
    if(loadPieceSequence)
    {
        printf("Loading piece sequence from file (%s)\n", pieceSequenceFile);
        
        allocateMatrixType2((void ***) &lPieceSequence, sizeof(double), retries, pieceSequenceLen / 8);
        
        int i, r;
        
        for(r = 0; r < retries; r++)
        {
            for(i = 0; i < pieceSequenceLen / 8; i++)
                lPieceSequence[r][i] = 0;
        }
                
        // Read piece sequence from file
        
        FILE *fp;

        fp = fopen (pieceSequenceFile, "r");
        if(fp == NULL)
        {
            printf("### fopen failed with file (%s)\n", pieceSequenceFile);
            
            return -6;            
        }

        for(r = 0; r < retries; r++)
        {
            for(i = 0; i < pieceSequenceLen / 8; i++)
            {
                if(fread((void *) &lPieceSequence[r][i], sizeof(uint8_t), 1, fp) != 1)
                {
                    freeMatrixType2((void **) lPieceSequence, retries);
                    fclose(fp);

                    printf("### fread failed (2) \n");
                    
                    return -7;
                }
            }
        }
        
        fclose(fp);

        lPieceSequenceLen = pieceSequenceLen;
        
        printf("Read (%i) piece sequences with length (%i)\n", retries, lPieceSequenceLen);
    }
    
    lRewardFunc = rewardFunc;    
    lVerbose = verbose;
    
    tConfigParam param;

    if(nStatesRepo > 0 && nStatesRepoEvaluate > 0)
    {
        printf("### either define --createStateRepo or --evaluateStateRepo but not both\n");
        return -6;
    }

    if(nStatesRepo > 0 && !benchmark)
    {
        printf("### createStateRepo given but without benchmark, start again with --createStateRepo N --benchmark\n");
        return -6;
    }
    
    if(nStatesRepoEvaluate > 0 && !benchmark)
    {
        printf("### evaluateStateRepo given but without benchmark, start again with --evaluateStateRepo N --benchmark\n");
        return -6;
    }

    if(decideWithRewards && !benchmark)
    {
        printf("### decideWithRewards given but without benchmark, start again with --decideWithRewards --benchmark\n");
        return -6;
    }

    if(lookupTable && !benchmark)
    {
        printf("### lookupTable given but without benchmark, start again with --lookupTable size --benchmark\n");
        return -6;
    }
    
    printf("Opening config file (%s)\n", tetrisConfFile);

    int ret = tdlParseConfig(&param, tetrisConfFile);
    if(ret)
    {
        printf("### Unable to open config file (%s)\n", tetrisConfFile);
        return -1;
    }

    if((nDecisionWeights != 0) && 
        (nDecisionWeights < param.agents))
    {
        printf("### Wrong number of decision weights. Have (%i) but should be minimum (%i)\n", nDecisionWeights, param.agents);
        return -4;        
    }

    if(nDecisionWeights > 0)
    {
        printf("Performing weighted decisions with the following weights:\n");
        unsigned i;
        for(i = 0; i < nDecisionWeights; i++)
            printf("%lf ", decisionWeight[i]);
        printf("\n");
    }
                
    printf("Learning state values, i.e. V(s)\n");
    
    if(learnExploredActions)
        printf("On-Policy Learning: Learn from exploited and explored actions\n");
    else
        printf("Off-Policy Learning: Only learn from exploited actions\n");

    printf("Reward func (%i)\n", lRewardFunc);
    
    if(jointDecisionsEpsilon > 0)
        printf("Only for committees with joint decisions during learning: Do single decisions with a probability of (%lf)\n", jointDecisionsEpsilon);
    
    double maxWeightCurrentAgent = param.weightCurrentAgent;
    
    if(decreasingDecisionWeight)
        printf("Decreasing the decision weight, starting from 1.0 down to (%lf), itOff (%li), itMax (%li)\n", maxWeightCurrentAgent, itOff, itMax);
    else
        printf("Not decreasing the decision weight\n");
    
    printf("Retries (%i)\n", retries);

    if(benchmark)
    {
        if(lookupTable)
        {
            printf("Use lookup table with (%li) elements for benchmark\n", lookupTable);
            lNumberListEntriesMax = lookupTable;
            ret = lookup_init();
            if(ret)
            {
                printf("### lookup_init() failed (%i)\n", ret);
                return -4;
            }
        }
        else
            printf("Not using lookup table for benchmark\n");
    }
    
    unsigned long micro_max_test = MAX_NUM_ACTIONS;
    unsigned long micro_max = 0;
    if(!decideWithRewards)
    {
        if(lInputCoding == eInputCoding_raw)
            lMlpInputNeurons = BOARD_HEIGHT * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_raw_top4)
            lMlpInputNeurons = 4 * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_raw_top6)
            lMlpInputNeurons = 6 * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_raw_top8)
            lMlpInputNeurons = 8 * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_raw_top10)
            lMlpInputNeurons = 10 * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_raw_top14)
            lMlpInputNeurons = 14 * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_feature1)
            lMlpInputNeurons = 2;
        else if(lInputCoding == eInputCoding_bertsekasIoffe ||
                lInputCoding == eInputCoding_bertsekasIoffe_normalized)
            lMlpInputNeurons = 21;
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized1)
            lMlpInputNeurons = BOARD_HEIGHT * (BOARD_WIDTH - 1) + MAX_HOLES;
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized1_heights)
            lMlpInputNeurons = BOARD_HEIGHT * BOARD_WIDTH + MAX_HOLES;
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized2)
            lMlpInputNeurons = BOARD_HEIGHT * (BOARD_WIDTH - 1) + BOARD_HEIGHT * (BOARD_WIDTH);        
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized2_heights)
            lMlpInputNeurons = 2 * BOARD_HEIGHT * BOARD_WIDTH;
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles_reducedHeightDifferences)
            lMlpInputNeurons = 10 * (BOARD_WIDTH - 1) + 10 * (BOARD_WIDTH);        
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_reducedNumberHoles)
            lMlpInputNeurons = BOARD_HEIGHT * (BOARD_WIDTH - 1) + 10 * (BOARD_WIDTH);        
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_onlyHeightDifferences)
            lMlpInputNeurons = BOARD_HEIGHT * (BOARD_WIDTH - 1);        
        else if(lInputCoding == eInputCoding_bertsekasIoffe_discretized_onlyNumberOfHoles)
            lMlpInputNeurons = BOARD_HEIGHT * BOARD_WIDTH;

        printf("Input neurons: (%i)\n", lMlpInputNeurons);
        printf("Iterations per agent (%i)\n", iterationsPerAgent);

        if(!param.batchSize)
        {
            micro_max = micro_max_test;
        }
        else
            micro_max = param.batchSize;

        if(nStatesRepoEvaluate > 0)
            micro_max_test = nStatesRepoEvaluate;
    }
    
    if(seed == 0)
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        seed = tv.tv_sec / 1000000UL + tv.tv_usec;
    }

    if(!decideWithRewards)
    {
        int i;
        char savfileToLoad[100];

        for(i = 0; i < param.agents; i++)
        {
            unsigned int seedAgent = seed + i;
            
            int ret;
            if(!param.stateValueFunctionApproximation)
            {
                printf("### Learning from state-value tables is not supported.\n");
                lookup_free();
                return -2;
            }
            else
            {
                int nAgents = 1;

                if(param.agents > 1 && param.learnFromAverageStateValues)
                    nAgents = param.agents;

                if(param.linearApproximation)
                {
                    ret = tdlInit(i, nAgents, param.agents, lMlpInputNeurons, micro_max, micro_max_test, NULL, NULL, seedAgent, eStateValueFuncApprox_linear, true, param.gradientPrimeFactor, param.a, param.trainingMode, 0, 0, param.weightedAverage, param.weightCurrentAgent, nDecisionWeights > 0, decisionWeight[i], param.decisionNoise, param.stateValueImprecision, param.tdcOwnSummedGradient, param.replacingTraces);
    /*                printf("### Linear function approximation is not supported.\n");
                    return -3;*/
                }
                else
                {
                    sprintf (savfileToLoad, "%s_%i", savfile, i);
            
                    printf("Load MLP from file (%s)\n", savfileToLoad);

                    ret = tdlInit(i, nAgents, param.agents, lMlpInputNeurons, micro_max, micro_max_test, param.conffile, savfileToLoad, seedAgent, eStateValueFuncApprox_nonlinear_mlp, true, param.gradientPrimeFactor, 0, 0, 0, 0, param.weightedAverage, param.weightCurrentAgent, nDecisionWeights > 0, decisionWeight[i], param.decisionNoise, param.stateValueImprecision, param.tdcOwnSummedGradient, param.replacingTraces);
                }
            }

            if(ret)
            {
                printf("### tdlInit returned error.\n");
                lookup_free();
                return -5;
            }
            else
                printf("tdlInit: Agent (%i) successfully initialized with seed (%u)\n", i, seedAgent);
        } /* for agents */
    }
    
    srandom (seed);

    if(!decideWithRewards)
    {
        int i;
        
        for(i = 0; i < param.agents; i++)
        {
            printf("Agent (%i), gamma (%lf)\n", i, param.gamma);
            tdlSetParam(i, param.normalizeLearningRate, param.alpha, param.beta, param.gamma, param.lambda);
        }
    
        if(createInitWeights)
        {
            printf("Creating initial weights and exiting\n");
            
            char filename[100];
            
            for(i = 0; i < param.agents; i++)
            {
                sprintf (filename, "%s_%i", savfile, i);

                printf("Save MLP to file (%s)\n", filename);

                tdlSaveNet (i, filename);
            }
            
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);
            
            lookup_free();
            return 0;
        }
        
        if(nStatesRepoEvaluate > 0)
        {
            // Get states from repo file
            
            tState *state = malloc(sizeof(tState) * nStatesRepoEvaluate);
            
            void *pHandle = NULL;
            
            int ret = tdlGetStateValuesPrepare(false, &pHandle, state, nStatesRepoEvaluate);
            if(ret)
            {
                printf("###tdlGetStateValuesPrepare returned error (%i)\n", ret);

                free(state);
                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                lookup_free();
                return -6;            
            }

            if(pHandle == NULL)
            {
                printf("### tdlGetStateValuesPrepare returned NULL handle\n");

                free(state);
                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                lookup_free();
                return -6;            
            }
            
            tStateInternal *stateInternal = malloc(sizeof(tStateInternal) * nStatesRepoEvaluate);
            
            // Read state coding repository
            FILE *fp;

            int nVals = 0;
            
            fp = fopen (repoFile, "r");
            if(fp == NULL)
            {
                printf("### fopen failed with file (%s)\n", repoFile);
                
                free(state);
                free(stateInternal);
                tdlFreeStateValues(pHandle);
                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                lookup_free();
                return -6;            
            }
            
            for(i = 0; i < nStatesRepoEvaluate; i++)
            {
                if (fread ((void *) &stateInternal[nVals], sizeof(tStateInternal), 1, fp) != 1)
                {
                    break;
                }
                nVals++;
            }
                        
            fclose(fp);
            
            printf("Read (%i) states from file (%s)\n", nVals, repoFile);
            
            if(calcConsistencies)
            {
                printf("Calculate (action) consistencies with state repo\n");

                if(forceExploitationConsistencies)
                    printf("Using different policy as for learning (greedy actions)\n");
                else
                    printf("Using same policy as for learning (may do random actions)\n");
                
                if(bestActions)
                    printf("Evaluate the best (%i) actions\n", bestActions);
                else if(allActions)
                    printf("Evaluate all actions\n");
                else if(singleActions)
                    printf("Evaluate best actions of single agent\n");
                else if(ensembleActions)
                    printf("Evaluate best actions of ensemble\n");                
                
                int ret = calculateConsistencies(&param, 0, repoErrorsFile, repoActionNoFile, stateInternal, nVals, forceExploitationConsistencies, bestActions, allActions, singleActions, ensembleActions, retries, verbose);
                if(ret)
                    printf("### calculateConsistencies failed (%i)\n", ret);
            }
            else
            {
                // Get state or state-action values (first agent only)
                
                for(i = 0; i < nVals; i++)
                {
                    // type is ignored
                    ret = tetrisInputVCodingGet(&stateInternal[i].tetris, state[i].s);
                    if(ret)
                    {
                        printf("### tetrisInputVCodingGet failed (%i)\n", ret);

                        free(state);
                        free(stateInternal);
                        tdlFreeStateValues(pHandle);
                        for(i = 0; i < param.agents; i++)
                            tdlCleanup(i);
                        
                        lookup_free();
                        return -6;
                    }
                }

                free(stateInternal);
                
                if(tdlGetStateValues(pHandle, 0, state, nVals))
                {
                    printf("### tdlGetStateValues error\n");

                    free(state);
                    tdlFreeStateValues(pHandle);
                    for(i = 0; i < param.agents; i++)
                        tdlCleanup(i);
                    
                    lookup_free();
                    return -6;
                }
                
                // Write state or state-action values to file
                
                fp = fopen (repoValuesFile, "w");
                if(fp == NULL)
                {
                    printf("### fopen failed with file (%s)\n", repoValuesFile);
                    
                    free(state);
                    tdlFreeStateValues(pHandle);
                    for(i = 0; i < param.agents; i++)
                        tdlCleanup(i);
                    
                    lookup_free();
                    return -6;
                }
                
                for(i = 0; i < nVals; i++)
                {
                    char str[100];
                    snprintf(str, 100, "%lf\n", state[i].Vs);
                    
                    fwrite(str, sizeof(char), strlen(str), fp);
                }
                
                fclose(fp);
                        
                printf("Written (%i) state or state-action values to file (%s)\n", nVals, repoValuesFile);                            
            }

            // TODO: Fix up
            tdlFreeStateValues(pHandle);
            free(state);
            
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);

            
            return 0;        
        }
    }
            
    if(benchmark)
    {
        if(nStatesRepo > 0)
        {
            tRepo repo;
            pRepo = &repo;

            pRepo->pStateInternalRepo = (tStateInternal *) malloc(sizeof(tStateInternal) * nStatesRepo);

            pRepo->nStateRepoMax = nStatesRepo;
            pRepo->nStateRepo = 0;

            pStateProbs = (unsigned long *) malloc (sizeof(unsigned long) * nStatesRepo);
            
            if(pStateProbs == NULL)
            {
                printf("### allocate pStateProbs failed\n");
                lookup_free();
                return -9;                
            }
            
            memset(pStateProbs, 0, sizeof(unsigned long) * nStatesRepo);

            pNActions = (unsigned long *) malloc (sizeof(unsigned long) * nStatesRepo);
        
            if(pNActions == NULL)
            {
                printf("### allocate pNActions failed\n");
                lookup_free();
                return -9;                
            }
        
            memset(pNActions, 0, sizeof(unsigned long) * nStatesRepo);            
        }

        // Do not start training, just evaluate the (trained and restored) value function approximator
        int ret = performBenchmarks(&param, retries, pRepo, pStateProbs, pNActions, probToAddState, decideWithRewards, deterministicEnvironment, nStatesPerAgentMax);

        printf("Benchmark done\n");        
        
        if(bestActions)
            printf("Limiting the number of actions to %i actions\n", bestActions);
        
        // Write state coding repository
        if(pRepo != NULL)
        {
            int nStateRepoReal = pRepo->nStateRepo;
            
            int i;
            
            if(minFrequencyState > 1)
            {
                nStateRepoReal = 0;
                for(i = 0; i < pRepo->nStateRepo; i++)
                {
                    if(pStateProbs[i] >= minFrequencyState)
                        nStateRepoReal++;
                }                
            }
            
            printf("pRepo->nStateRepo (%i)\n", pRepo->nStateRepo);
            printf("Number of states with a minimum frequency of (%i): (%i)\n", minFrequencyState, nStateRepoReal);
            
            if(!ret && nStateRepoReal > 0)
            {
                FILE *fp;
                
                fp = fopen (repoFile, "w");
                if (fp != NULL)
                {
                    for(i = 0; i < pRepo->nStateRepo; i++)
                    {
                        if(pStateProbs[i] >= minFrequencyState)
                        {
                            if (fwrite ((const void *) &pRepo->pStateInternalRepo[i], sizeof(tStateInternal), 1, fp) != 1)
                                printf("### fwrite failed\n");
                        }                        
                    }
                    
                    fclose(fp);
                    
                    printf("Written (%i) states to file (%s)\n", nStateRepoReal, repoFile);
                }
                                                
                fp = fopen (repoFileStateProbs, "w");
                if (fp != NULL)
                {
                    for(i = 0; i < pRepo->nStateRepo; i++)
                    {
                        if(pStateProbs[i] >= minFrequencyState)
                        {
                            char str[100];
                            
                            snprintf(str, 100, "%lu\n", pStateProbs[i]);

                            fwrite(str, sizeof(char), strlen(str), fp);
                        }
                    }

                    fclose(fp);
                    
                    printf("Written (%i) state probs to file (%s)\n", nStateRepoReal, repoFileStateProbs);
                }

                fp = fopen (repoFileNumberActions, "w");
                if (fp != NULL)
                {
                    for(i = 0; i < pRepo->nStateRepo; i++)
                    {
                        if(pStateProbs[i] >= minFrequencyState)
                        {
                            char str[100];

                            if(bestActions)
                            {
                                if(pNActions[i] > bestActions)
                                    pNActions[i] = bestActions;
                            }
                            
                            snprintf(str, 100, "%lu\n", pNActions[i]);

                            fwrite(str, sizeof(char), strlen(str), fp);
                        }
                    }

                    fclose(fp);
                    
                    printf("Written (%i) number actions to file (%s)\n", nStateRepoReal, repoFileNumberActions);
                }                
            }
        }
        
        if(!decideWithRewards)
        {
            int i;
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);
        }
        
        if(nStatesRepo > 0)
        {
            free(pRepo->pStateInternalRepo);            
        }        
        
        lookup_free();
        
        if(ret)
        {
            printf("### performBenchmarks failed (%i)\n", ret);
            return -9;
        }
        else
            return 0;        
    }
                
    int z = 0;

    tTetris tetris[param.agents];
    int steps[param.agents];
    tPosition lastPiecePos[param.agents];

    // Initialize the board for all agents
    for(z = 0; z < param.agents; z++)
    {
        // Erase board
        tetrisEraseBoard(&tetris[z]);

        steps[z] = 0;
    }
        
    unsigned long t2, t2max = param.iterations / iterationsPerAgent;
    z = 0;

    for(t2 = 0; t2 < t2max; t2++)
    {
        bool forceSingleDecisionsTmp = false;
        
        if(jointDecisionsEpsilon > 0 && param.agents > 1 && !benchmark)
        {
            double myrand = randValDouble (0, 1);
            if (myrand < jointDecisionsEpsilon)
                forceSingleDecisionsTmp = true;
            else
                forceSingleDecisionsTmp = false;
        }
        
        for(z = 0; z < param.agents; z++)
        {
            unsigned long t3;
            for(t3 = 0; t3 < iterationsPerAgent; t3++)
            {
                unsigned long t = t2 * iterationsPerAgent + t3;
                if(t > 0)
                {
                    // Select piece randomly (stochastic SZ-Tetris)
                    int type = randValLong(0, 1);

                    bool isTerminalState = false;
                    bool exploited = false;
                    double rewardArr[MAX_NUM_ACTIONS];
                    tPosition posArr[MAX_NUM_ACTIONS];
                    double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
                    unsigned long bestStatesIndicesLen = 0;

                    double *pVsTo2Arr = NULL;
                    
                    if(param.learnFromAverageStateValues && param.agents > 1)
                        pVsTo2Arr = VsTo2Arr;

                    if(decreasingDecisionWeight && param.agents > 1 && param.weightedDecisions && param.averageDecision)
                    {
                        double decisionWeight = 1.0 - (double) (t + itOff) / itMax * maxWeightCurrentAgent;
                        tdlSetDecisionWeightCurrentAgent(decisionWeight);
                        if(t % 10000 == 0)
                            printf("Current decision weight: %lf\n", decisionWeight);
                    }
                    
                    /* Observe the current reward, 
                     * choose the (best) action according to the current policy and decision mechanism
                     * and observe the successor state. Then choose the best action in the successor state */
                    
                    int ret = tetrisBestState(&tetris[z], &param, z, false, false, type, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, posArr, VsRealToArr, pVsTo2Arr, false, t, param.iterations, forceSingleDecisionsTmp, false);

                    if(ret)
                    {
                        printf("###  tetrisBestState failed (%i)\n", ret);
                        break;
                    }

                    if(!bestStatesIndicesLen)
                    {
                        printf("### bestStatesIndicesLen is Zero\n");
                        break;
                    }

                    double reward;
                    tPosition pos;
                    double VsRealTo, VsTo2 = 0;

                    // Randomly select one of the best actions
                    unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                    reward = rewardArr[myrand];
                    pos = posArr[myrand];
                    VsRealTo = VsRealToArr[myrand];
                    if(pVsTo2Arr != NULL)
                        VsTo2 = VsTo2Arr[myrand];                        

                    if(verbose)
                        printf("(best) actions (%li), piece (%i), pos.x (%i), pos.y (%i), isHorizontal (%i)\n", bestStatesIndicesLen, type, posArr[myrand].x, posArr[myrand].y, posArr[myrand].isHorizontal);

                    // 1. pos[z] = posFrom, action = actionFrom
                    // 2. pos[z] = posTo, action = actionTo

                    if(exploited)
                    {
                        if(verbose)
                            printf("Exploited action at t = %li\n", t);
                    }
                    else
                    {
                        if(verbose)
                            printf("Explored action at t = %li\n", t);
                    }

                    if(steps[z] > 0)
                    {
                        ret = tetrisLearn(&tetris[z], &param, z, isTerminalState, exploited, reward, type, pos, VsRealTo, VsTo2, lastPiecePos[z], learnExploredActions);
                        if(ret)
                        {
                            printf("### tetrisLearn returned error (%i)\n", ret);
                            break;
                        }
                    }
                    else
                    {
                        if((z > 0) && (z == param.agents - 1) && (!param.updateWeightsImmediate))
                        {
                            int z2;
                            for(z2 = 0; z2 < param.agents; z2++)
                            {
                                if(verbose)
                                    printf("force learning, z2 (%i)\n", z2);

                                if(param.batchSize > 0)
                                {
                                    tdlAddStateDone(z2, true);
                                }
                            }                            
                        }
                        
                        if(verbose)
                            printf("Not learning empty board\n");
                    }

                    lastPiecePos[z] = pos;

                    steps[z]++;
                    
                    if(!isTerminalState)
                    {
                        // Non-terminal state, apply the position

                        ret = tetrisApplyPos(&tetris[z], type, pos);
                        if(ret)
                        {
                            printf("### tetrisApplyPos failed (%i)\n", ret);
                            break;
                        }

                        // Erase the full lines
                        tetrisEraseLines(&tetris[z]);
                    }
                    else
                    {
                        // Terminal state

                        // Erase board
                        tetrisEraseBoard(&tetris[z]);

                        steps[z] = 0;
                    }
                    if(verbose)
                        outputBoard(&tetris[z], BOARD_HEIGHT - 1, 0);
                } /* t > 0 */
            } /* for t3 */
        } /* for z */
    } /* for t2 */

    char filename[100];

    int i;
    for(i = 0; i < param.agents; i++)
    {
        sprintf (filename, "%s_%i", savfile, i);

        printf("Save MLP to file (%s)\n", filename);

        tdlSaveNet (i, filename);
    }

    if(lPieceSequence != NULL)
        freeMatrixType2((void **) lPieceSequence, retries);
    
    for(i = 0; i < param.agents; i++)
        tdlCleanup(i);

    lookup_free();
    
    return 0;
}
