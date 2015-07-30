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
 * \file mazePathPlanning.c
 * \brief Generalized maze environment
 *
 * \author Stefan Fausser
 * 
 * Modification history:
 * 
 * 2012-04-01, S. Fausser - written
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

#undef VERIFY_AVERAGED_STATE_VALUES
            
#define MAX_WIDTH               50
#define MAX_WIDTH_INTERNAL      5
#define MAX_NUM_ACTIONS         4
#define ALPHA_DISCOUNT          1.0

#ifndef INFINITY
#   define INFINITY 999999
#endif

static bool lVerbose = false;
// 
typedef enum
{
    eAction_north = 0,
    eAction_south,
    eAction_east,
    eAction_west,
    eAction_none
} eAction;

typedef union
{
    uint8_t val;
    struct
    {
        unsigned
            actionNorth : 1,
            actionSouth : 1,
            actionEast : 1,
            actionWest : 1;
    } __attribute((packed));
} tMazeActions;

typedef struct
{
    int x;
    int y;
} tPosition;

static int lMlpInputNeurons = 0;

typedef struct
{
    int heightAndWidth;
    tPosition goal;
    bool barrier[MAX_WIDTH_INTERNAL][MAX_WIDTH_INTERNAL];
} tMazeInternal;

typedef struct
{
    int heightAndWidth;
    tPosition goal;
    bool barrier[MAX_WIDTH][MAX_WIDTH];
    int minSteps[MAX_WIDTH][MAX_WIDTH][MAX_NUM_ACTIONS];
    double Vtrue[MAX_WIDTH][MAX_WIDTH][MAX_NUM_ACTIONS];
    tMazeActions piTrue[MAX_WIDTH][MAX_WIDTH];
} tMaze;

typedef struct
{
    int mazeNumber;    
    tPosition pos;
    eAction action;
} __attribute__((packed)) tStateInternal;

typedef struct
{
    int nStateRepoMax;
    
    int nStateRepo;
    
    tStateInternal *pStateInternalRepo;    
} tRepo;

int generateMaze(tMaze *pMaze, int nBarriers)
{
    if(nBarriers > pow(pMaze->heightAndWidth, 2.0))
    {
        printf("### generateMaze: nBarriers (%i) is too large for %i^2 states\n", nBarriers, pMaze->heightAndWidth);
        return -1;
    }    

    int i,j;
    
    for(i = 0; i < pMaze->heightAndWidth; i++)
    {
        for(j = 0; j < pMaze->heightAndWidth; j++)
        {
            pMaze->barrier[i][j] = false;
        }
    }
    
    /* Randomly place barriers in the maze */

    int32_t val1, val2;
    
    for(i = 0; i < nBarriers; i++)
    {
        val1 = randValLong(0, pMaze->heightAndWidth - 1);
        val2 = randValLong(0, pMaze->heightAndWidth - 1);

        if(pMaze->barrier[val2][val1])
        {
            /* We already have placed a barrier at this position.
                * Try again! */
            i--;
            continue;
        }
        
        pMaze->barrier[val2][val1] = true;
    }

    /* Randomly set the goal position */
    do
    {
        pMaze->goal.x = randValLong(0, pMaze->heightAndWidth - 1);
        pMaze->goal.y = randValLong(0, pMaze->heightAndWidth - 1);
    } while(pMaze->barrier[pMaze->goal.x][pMaze->goal.y]);

    return 0;
}

typedef struct
{
    tPosition pos[MAX_WIDTH * MAX_WIDTH];
    unsigned int queueSize;
} tQueue;

int enqueuePos(tQueue *pQueue, tPosition pos)
{
    // FIFO Queue
    pQueue->pos[pQueue->queueSize] = pos;
    
    pQueue->queueSize++;

    return 0;    
}

int dequeuePos(tQueue *pQueue, tPosition *pPos)
{
    // FIFO Queue
    *pPos = pQueue->pos[0];

    int i;
    for(i = 0; i < pQueue->queueSize - 1; i++)
        pQueue->pos[i] = pQueue->pos[i + 1];
    
    pQueue->queueSize--;

    return 0;
}

int searchShortestPath_BreadthFirst(
    tMaze *pMaze,
    tPosition posStart,
    eAction firstAction)
{
    bool visited[pMaze->heightAndWidth][pMaze->heightAndWidth];
    int label[pMaze->heightAndWidth][pMaze->heightAndWidth];

    int x, y;

    // Nothing is possible in a terminal state
    if(pMaze->barrier[posStart.x][posStart.y])
        return -1;

    if(posStart.x == pMaze->goal.x &&
       posStart.y == pMaze->goal.y)
        return 0;

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            visited[x][y] = false;
            label[x][y] = -1;
        }
    }

    tQueue queue;
    queue.queueSize = 0;

    enqueuePos(&queue, posStart);

    bool first = true;
    
    while(queue.queueSize > 0)
    {
        tPosition pos;
        dequeuePos(&queue, &pos);

        if(pos.x == pMaze->goal.x &&
           pos.y == pMaze->goal.y)
            return(label[pos.x][pos.y] + 1);

        eAction a, startA, endA;
        if(pos.x == posStart.x && 
           pos.y == posStart.y && first)
        {
            startA = firstAction;
            endA = firstAction + 1;
            first = false;
        }
        else
        {
            startA = 0;
            endA = 4;
        }

        // Iterate all actions
        for(a = startA; a < endA; a++)
        {
            tPosition posTmp = pos;

            // Perform the action and verify if have reached a terminal state

            if(a == eAction_north)
            {
                if((posTmp.y == 0) ||
                (pMaze->barrier[posTmp.x][posTmp.y-1]))
                {
                    // Do nothing
                }
                else
                {
                    posTmp.y--;
                    if(!visited[posTmp.x][posTmp.y])
                    {
                        visited[posTmp.x][posTmp.y] = true;
                        label[posTmp.x][posTmp.y] = label[pos.x][pos.y] + 1;
                        enqueuePos(&queue, posTmp);
                    }
                }
            }
            else if(a == eAction_south)
            {
                if((posTmp.y >= pMaze->heightAndWidth - 1) ||
                (pMaze->barrier[posTmp.x][posTmp.y+1]))
                {
                    // Do nothing
                }
                else
                {
                    posTmp.y++;
                    if(!visited[posTmp.x][posTmp.y])
                    {
                        visited[posTmp.x][posTmp.y] = true;
                        label[posTmp.x][posTmp.y] = label[pos.x][pos.y] + 1;
                        enqueuePos(&queue, posTmp);
                    }
                }
            }
            else if(a == eAction_east)
            {
                if((posTmp.x >= pMaze->heightAndWidth - 1) ||
                (pMaze->barrier[posTmp.x+1][posTmp.y]))
                {
                    // Do nothing
                }
                else
                {
                    posTmp.x++;
                    if(!visited[posTmp.x][posTmp.y])
                    {
                        visited[posTmp.x][posTmp.y] = true;
                        label[posTmp.x][posTmp.y] = label[pos.x][pos.y] + 1;
                        enqueuePos(&queue, posTmp);
                    }
                }
            }
            else if(a == eAction_west)
            {
                if((posTmp.x == 0) ||
                (pMaze->barrier[posTmp.x-1][posTmp.y]))
                {
                    // Do nothing
                }
                else
                {
                    posTmp.x--;
                    if(!visited[posTmp.x][posTmp.y])
                    {
                        visited[posTmp.x][posTmp.y] = true;
                        label[posTmp.x][posTmp.y] = label[pos.x][pos.y] + 1;
                        enqueuePos(&queue, posTmp);
                    }
                }
            }
        } /* for a */
    } /* while */

    return -1;
}

bool isGoalReachable(
    tMaze *pMaze)
{
    int x, y;
    int maxA = MAX_NUM_ACTIONS, a;

    /* Verify that there is at least one action in a non-terminal state
     * that leads to a path to the goal */

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            // If it is a terminal state then continue
            if(pMaze->barrier[x][y] || (pMaze->goal.x == x && pMaze->goal.y == y))
                continue;

            bool actionToPathToGoal = false;
            for(a = 0; a < maxA; a++)
            {
                if(pMaze->minSteps[x][y][a] > 0)
                {
                    actionToPathToGoal = true;
                    break; /* for a */
                }
            }
            
            if(!actionToPathToGoal)
                return false;
        }
    }
    
    return true;
}

int outputMaze2(
    tMaze *pMaze)
{
    int x, y;
    
    printf("Maze:\n");
    
    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        printf("|");
        
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {

            if(pMaze->goal.x == x &&
               pMaze->goal.y == y)
                printf("X");
            else if(pMaze->barrier[x][y])
                printf("#");
            else
                printf(" ");
        }
        printf("|\n");
    }
    
    return 0;
}

int outputMaze3(
    tMaze *pMaze,
    tPosition pos)
{
    int x, y;
    
    printf("Maze:\n");
    
    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        printf("|");
        
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            if(pMaze->goal.x == x &&
               pMaze->goal.y == y)
                printf("X");
            else if(pMaze->barrier[x][y])
                printf("#");
            else if(pos.x == x &&
                    pos.y == y)
                printf("P");
            else
                printf(" ");
        }
        printf("|\n");
    }
    
    return 0;
}

int outputMaze(
    tMaze *pMaze)
{
    int x, y;
    int minA = 0, maxA = 1;
    
    outputMaze2(pMaze);
    
    printf("Minimum number of required steps:\n");

    eAction a;

    maxA = MAX_NUM_ACTIONS;
    for(a = 0; a < maxA; a++)
    {
        printf("Action (%i)\n", a);
        for(y = 0; y < pMaze->heightAndWidth; y++)
        {
            for(x = 0; x < pMaze->heightAndWidth; x++)
            {
                printf("%02i ", pMaze->minSteps[x][y][a]);
            }
            printf("\n");
        }
    }
    
    printf("True state-values:\n");
    for(a = minA; a < maxA; a++)
    {
        printf("Action (%i)\n", a);
        for(y = 0; y < pMaze->heightAndWidth; y++)
        {
            for(x = 0; x < pMaze->heightAndWidth; x++)
            {
                printf("%04lf ", pMaze->Vtrue[x][y][a]);
            }
            printf("\n");
        }
    }

    printf("Best actions:\n");
    printf("Legend: North                       (1)\n");
    printf("Legend: South                       (2)\n");
    printf("Legend: South, North                (3)\n");
    printf("Legend: East                        (4)\n");
    printf("Legend: East, North                 (5)\n");
    printf("Legend: East, South                 (6)\n");
    printf("Legend: East, South, North          (7)\n");
    printf("Legend: West                        (8)\n");
    printf("Legend: West, North                 (9)\n");
    printf("Legend: West, South                 (A)\n");
    printf("Legend: West, South, North          (B)\n");
    printf("Legend: West, East                  (C)\n");
    printf("Legend: West, East, North           (D)\n");
    printf("Legend: West, East, South           (E)\n");
    printf("Legend: West, East, South, North    (F)\n");
    
    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            printf("%X ", pMaze->piTrue[x][y].val);
        }
        printf("\n");
    }

    return 0;
}

int comparePolicies(
    tMaze *pMaze,
    tMazeActions piTrue[][MAX_WIDTH],
    tMazeActions pi[][MAX_WIDTH])
{
    int x, y;

    int nCorrectActions = 0;
    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            bool policyIsCorrect = true;

            /* We only compare the policies if there is a goal reachable in this
             * given position. Otherwise any action will be counted as correct */
            if(piTrue[x][y].val != 0)
            {
                if(pi[x][y].actionNorth &&
                    !piTrue[x][y].actionNorth)
                    policyIsCorrect = false;
                    
                if(pi[x][y].actionSouth &&
                    !piTrue[x][y].actionSouth)
                    policyIsCorrect = false;
                    
                if(pi[x][y].actionEast &&
                    !piTrue[x][y].actionEast)
                    policyIsCorrect = false;
                    
                if(pi[x][y].actionWest &&
                    !piTrue[x][y].actionWest)
                    policyIsCorrect = false;
            }

            if(policyIsCorrect)
                nCorrectActions++;
        }
    }

    return nCorrectActions;
}

int mazeGetAllActions(
    tMaze *pMaze,
    tPosition pos,
    eAction *pActions,
    int *pNActions,
    int nActionsMax)
{
    // Iterate all actions
    eAction a;
    for(a = 0; a < MAX_NUM_ACTIONS; a++)
    {
        tPosition posTmp = pos;
        bool invalidAction = false;

        // Perform the action, filter out actions that would move the agent out the maze

        if(a == eAction_north)
        {
            if(posTmp.y <= 0)
            {
                // Moving outside is not allowed, keep the actual position
                invalidAction = true;
            }
        }
        else if(a == eAction_south)
        {
            if(posTmp.y >= pMaze->heightAndWidth - 1)
            {
                // Moving outside is not allowed, keep the actual position
                invalidAction = true;
            }
        }
        else if(a == eAction_east)
        {
            if(posTmp.x >= pMaze->heightAndWidth - 1)
            {
                // Moving outside is not allowed, keep the actual position
                invalidAction = true;
            }
        }
        else if(a == eAction_west)
        {
            if(posTmp.x <= 0)
            {
                // Moving outside is not allowed, keep the actual position
                invalidAction = true;
            }
        }

        if(!invalidAction)
        {
            if(*pNActions < nActionsMax)
            {
                pActions[*pNActions] = a;
                *pNActions = *pNActions + 1;
            }
            else
                break; /* a */
        }
    } /* for a */
    
    return 0;
}

int calculatePolicy(
    tMaze *pMaze,
    double pVs[][MAX_WIDTH][MAX_NUM_ACTIONS],
    tMazeActions pPi[][MAX_WIDTH])
{
    int x, y;

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            tPosition pos;
            pos.x = x;
            pos.y = y;

            double bestValue = -INFINITY;

            tMazeActions bestActions;
            bestActions.val = 0;

            eAction a;
            if(pMaze->barrier[pos.x][pos.y] || (pos.x == pMaze->goal.x && pos.y == pMaze->goal.y) )
            {
                // Cannot move anywhere
                bestActions.val = 0;
            }
            else
            {
                eAction actions[MAX_NUM_ACTIONS];
                int nActions = 0;
                
                mazeGetAllActions(pMaze, pos, actions, &nActions, MAX_NUM_ACTIONS);
                
                // Iterate all actions
                int b;
                for(b = 0; b < nActions; b++)
                {
                    a = actions[b];
                    
                    if(pVs[pos.x][pos.y][a] > bestValue)
                    {
//                            printf("%i/%i: %lf\n", x, y, pMaze->Vtrue[posTmp.x][posTmp.y]);
                        bestValue = pVs[pos.x][pos.y][a];

                        // Clear all actions
                        bestActions.val = 0;

                        // Set the first best action

                        if(a == eAction_north)
                            bestActions.actionNorth = 1;
                        else if(a == eAction_south)
                            bestActions.actionSouth = 1;
                        else if(a == eAction_east)
                            bestActions.actionEast = 1;
                        else if(a == eAction_west)
                            bestActions.actionWest = 1;                           
                    }
                    else if(pVs[pos.x][pos.y][a] == bestValue)
                    {
                        // Add another action
//                            printf("Add %i/%i: %lf\n", x, y, pMaze->Vtrue[posTmp.x][posTmp.y]);

                        if(a == eAction_north)
                            bestActions.actionNorth = 1;
                        else if(a == eAction_south)
                            bestActions.actionSouth = 1;
                        else if(a == eAction_east)
                            bestActions.actionEast = 1;
                        else if(a == eAction_west)
                            bestActions.actionWest = 1;                        
                    }
                } /* for b */
            }

            /* Note that bestActions.val can still be Zero, meaning that there 
            * is no goal reachable by our current position */
            pPi[x][y] = bestActions;
        }
    }

    return 0;
}

int calculateTrueStateValues(
    tMaze *pMaze,
    double reward_pos,
    double reward_neg,
    double gamma,
    int maxDepth)
{
    int x, y;
    eAction a;

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            tPosition pos;
            pos.x = x;
            pos.y = y;

            int maxA = MAX_NUM_ACTIONS;
            for(a = 0; a < maxA; a++)
            {
                int depth = searchShortestPath_BreadthFirst(pMaze, pos, a);
                double Vs;
                if(depth > 0)
                    Vs = reward_pos * pow(gamma, depth - 1);
                else
                {
                    // Assign reward_neg if state + action is a terminal state (barrier or goal)
//                    Vs = reward_neg;
                    Vs = -INFINITY;
                }

                pMaze->Vtrue[x][y][a] = Vs;
                pMaze->minSteps[x][y][a] = depth;
            }
        }
    }

    calculatePolicy(pMaze, pMaze->Vtrue, pMaze->piTrue);

    return 0;
}

int saveMazes(tMaze *pMaze, char *savfile, int nMazes)
{
    FILE *fp = fopen(savfile, "w+");
    
    int ret = fwrite(&nMazes, sizeof(nMazes), 1, fp);
    if(ret != 1)
    {
        printf("### Unable to write maze to file (%s)\n", savfile);
        fclose(fp);
        return -1;        
    }

    int i;
    for(i = 0; i < nMazes; i++)
    {
        int ret = fwrite(&pMaze[i], sizeof(tMaze), 1, fp);
        if(ret != 1)
        {
            printf("### Unable to write maze to file (%s)\n", savfile);
            fclose(fp);
            return -1;        
        }
    }
    
    fclose(fp);
    
    return 0;
}

int restoreMazes(tMaze **pMaze, char *savfile, int *pnMazes)
{
    FILE *fp = fopen(savfile, "r");
    
    int ret = fread(pnMazes, sizeof(*pnMazes), 1, fp);
    if(ret != 1)
    {
        printf("### Unable to read maze from file (%s)\n", savfile);
        fclose(fp);
        return -1;
    }

    *pMaze = malloc(sizeof(tMaze) * *pnMazes);
    if(*pMaze == NULL)
    {
        printf("### malloc failed\n");
        return -2;
    }
    
    int i;
    for(i = 0; i < *pnMazes; i++)
    {
        ret = fread(&(*pMaze)[i], sizeof(tMaze), 1, fp);
        if(ret != 1)
        {
            printf("### Unable to read maze from file (%s)\n", savfile);
            fclose(fp);
            return -3;        
        }
    }
    
    fclose(fp);
    
    return 0;
}

typedef enum
{
    eInputCoding_default,
    eInputCoding_default2,
    eInputCoding_default3,
    eInputCoding_default4,
} eInputCoding;

static eInputCoding inputCoding = eInputCoding_default2;

int mazeInputQCodingGet(
    tMaze *pMaze,
    tPosition pos,
    double *x,
    eAction action)
{
    int actionNr = action;

    int i;
    for(i = 0; i < lMlpInputNeurons; i++)
        x[i] = 0;

    if(inputCoding == eInputCoding_default)
    {
        int offset = 0;

        if(actionNr == eAction_north)
            offset = 0;
        else if(actionNr == eAction_east)
            offset = pow(pMaze->heightAndWidth, 2.0) * 3.0;
        else if(actionNr == eAction_south)
            offset = pow(pMaze->heightAndWidth, 2.0) * 3.0 * 2.0;
        else if(actionNr == eAction_west)
            offset = pow(pMaze->heightAndWidth, 2.0) * 3.0 * 3.0;
        
        // Barriers
        for(i = 0; i < pMaze->heightAndWidth; i++)
        {
            int j;
            for(j = 0; j < pMaze->heightAndWidth; j++)
            {
                if(pMaze->barrier[i][j])
                    x[offset] = 1;
                offset++;
            }
        }

        // Goal position
        int goalIndex = pMaze->goal.y * pMaze->heightAndWidth + pMaze->goal.x;
        x[offset + goalIndex] = 1;
        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;

        // Player position
        int playerIndex = pos.y * pMaze->heightAndWidth + pos.x;
        x[offset + playerIndex] = 1;

        return 0;
    }
    else if(inputCoding == eInputCoding_default2)
    {
        int offset = 0;
        
        // Barriers
        for(i = 0; i < pMaze->heightAndWidth; i++)
        {
            int j;
            for(j = 0; j < pMaze->heightAndWidth; j++)
            {
                if(pMaze->barrier[i][j])
                    x[offset] = 1;
                offset++;
            }
        }

        // Goal position
        int goalIndex = pMaze->goal.y * pMaze->heightAndWidth + pMaze->goal.x;
        x[offset + goalIndex] = 1;
        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;

        if(actionNr == eAction_north)
            offset += 0;
        else if(actionNr == eAction_east)
            offset += pow(pMaze->heightAndWidth, 2.0);
        else if(actionNr == eAction_south)
            offset += pow(pMaze->heightAndWidth, 2.0) * 2.0;
        else if(actionNr == eAction_west)
            offset += pow(pMaze->heightAndWidth, 2.0) * 3.0;
        
        // Player position
        int playerIndex = pos.y * pMaze->heightAndWidth + pos.x;
        x[offset + playerIndex] = 1;

        return 0;
    }
    else if(inputCoding == eInputCoding_default3)
    {
        int offset = 0;
        
        // Barriers
        for(i = 0; i < pMaze->heightAndWidth; i++)
        {
            int j;
            for(j = 0; j < pMaze->heightAndWidth; j++)
            {
                if(pMaze->barrier[i][j])
                    x[offset] = 1;
                offset++;
            }
        }

        // Goal position
        int goalIndex = pMaze->goal.y * pMaze->heightAndWidth + pMaze->goal.x;
        x[offset + goalIndex] = 1;
        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;
        
        // Player position
        int playerIndex = pos.y * pMaze->heightAndWidth + pos.x;
        x[offset + playerIndex] = 1;

        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;
        
        // Action
        
        if(actionNr == eAction_north)
            x[offset] = 1;
        else if(actionNr == eAction_east)
            x[offset + 1] = 1;
        else if(actionNr == eAction_south)
            x[offset + 2] = 1;
        else if(actionNr == eAction_west)
            x[offset + 3] = 1;
                
        return 0;
    }
    else if(inputCoding == eInputCoding_default4)
    {
        int offset = 0;
        
        // Barriers
        for(i = 0; i < pMaze->heightAndWidth; i++)
        {
            int j;
            for(j = 0; j < pMaze->heightAndWidth; j++)
            {
                if(pMaze->barrier[i][j])
                    x[offset] = 1;
                offset++;
            }
        }

        // Goal position
        int goalIndex = pMaze->goal.y * pMaze->heightAndWidth + pMaze->goal.x;
        x[offset + goalIndex] = 1;
        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;

        // Player position (current)
        int playerIndex = pos.y * pMaze->heightAndWidth + pos.x;
        x[offset + playerIndex] = 1;
        offset += pMaze->heightAndWidth * pMaze->heightAndWidth;

        // Player position (new position)

        tPosition posNew = pos;

        if(actionNr == eAction_north && posNew.y > 0)
            posNew.y--;
        else if(actionNr == eAction_east && posNew.x < pMaze->heightAndWidth - 1)
            posNew.x++;
        else if(actionNr == eAction_south && posNew.y < pMaze->heightAndWidth - 1)
            posNew.y++;
        else if(actionNr == eAction_west && posNew.x > 0)
            posNew.x--;

        int playerIndexNew = posNew.y * pMaze->heightAndWidth + posNew.x;
        x[offset + playerIndexNew] = 1;
        
        return 0;
    }
    
    return -2;
}

bool isDynamicUpwind(double epsilon)
{
    double myrand = randValDouble(0, 1);
    if(epsilon > myrand)
        return true;
    else
        return false;
}

bool checkTerminalState(
    tMaze *pMaze,
    tConfigParam *pParam,
    tPosition pos,
    double *pReward)
{
    if(pMaze->barrier[pos.x][pos.y])
    {
        // Terminal transition
        *pReward = pParam->reward_lost;
        return true;
    }
    else if((pos.x == pMaze->goal.x) && 
            (pos.y == pMaze->goal.y))
    {
        // Terminal transition
        *pReward = pParam->reward_won;
        return true;
    }

    // reward for being in an empty space and not being in a barrier or being outside
    *pReward = pParam->reward_draw;
    return false;
}

int mazeLearn(
    tMaze *pMaze,
    tConfigParam *pParam,
    int agentNo,
    bool isTerminalState,
    bool isExploited,
    double reward,
    tPosition posFrom,
    eAction actionFrom,
    tPosition posTo,
    eAction actionTo,
    bool isTerminalStateTo,
    double VsRealTo,
    double VsTo2,
    bool learnExploredActions)
{
    // isTerminalState describes if posFrom is a terminal state independent of actionFrom
 
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
        // Terminal states are always zero
        
        return 0;
    }
    else
    {
        if(lVerbose)
            printf("mazeLearn, non-terminal state, agent (%i), exploited (%i)\n", agentNo, isExploited);
        
        if(isExploited || learnExploredActions)
        {
            tState stateFrom;
            tState stateTo;

            stateTo.Vs = VsRealTo;

            tState *pStateToFurtherAgents = NULL;
            tState stateToFurtherAgents;
            if(pParam->learnFromAverageStateValues && pParam->agents > 1 && !isTerminalStateTo)
            {
                pStateToFurtherAgents = &stateToFurtherAgents;
                stateToFurtherAgents.Vs = VsTo2;
    //            printf("mazeLearn: VsTo2 (%lf)\n", VsTo2);
            }

            if(isTerminalStateTo)
            {
                // Check the reward
                if(reward != pParam->reward_won && reward != pParam->reward_lost)
                {
                    printf("### mazeLearn: isTerminalStateTo, Invalid reward (%lf)\n", reward);
                    return -3;
                }
                
                int ret = tdlAddState(agentNo, &stateFrom, NULL, NULL, reward, ALPHA_DISCOUNT);
                if(ret)
                {
                    printf("### mazeLearn: tdlAddState returned error (%i)\n", ret);
                    return -3;
                }

                mazeInputQCodingGet(pMaze, posFrom, stateFrom.s, actionFrom);
            }
            else
            {
                // Check the reward
                if(reward != pParam->reward_draw)
                {
                    printf("### mazeLearn: Invalid reward (%lf)\n", reward);
                    return -3;
                }
                
                int ret = tdlAddState(agentNo, &stateFrom, &stateTo, pStateToFurtherAgents, reward, ALPHA_DISCOUNT);
                if(ret)
                {
                    printf("### mazeLearn: tdlAddState returned error (%i)\n", ret);
                    return -3;
                }

                mazeInputQCodingGet(pMaze, posFrom, stateFrom.s, actionFrom);

                mazeInputQCodingGet(pMaze, posTo, stateTo.s, actionTo);
                
#ifdef VERIFY_AVERAGED_STATE_VALUES
                tState stateToMine;
                    
                if(pStateToFurtherAgents != NULL)
                {
                    // Get the (average) state-value of the to-State

                    void *pHandle;
                    if(tdlGetStateValuesPrepare(false, &pHandle, &stateToMine, 1))
                    {
                        printf("### mazeLearn: tdlGetStateValuesPrepare function returned error.\n");
                        return -4;
                    }
                                
                    if(pHandle == NULL)
                    {
                        printf("### mazeLearn: tdlGetStateValuesPrepare function returned NULL handle\n");
                        return -4;                
                    }
                    
                    mazeInputQCodingGet(pMaze, posTo, stateToMine.s, actionTo);                       
                    
                    double vsSummed = 0;
                        
                    for(z = 0; z < pParam->agents; z++)
                    {                            
                        ret = tdlGetStateValues(pHandle, z, &stateToMine, 1);
                        if(ret)
                        {
                            printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                            tdlFreeStateValues(pHandle);
                            return -4;
                        }
                        
//                            printf("Agent (%i), state value (%lf)\n", z, stateToMine.Vs);

                        if(z == agentNo)
                            continue;
                        
                        vsSummed += stateToMine.Vs;                    
                    }
                            
                    tdlFreeStateValues(pHandle);
                    
                    if(vsSummed != stateToFurtherAgents.Vs)
                    {
                        printf("### mazeLearn: Verifying averaged state values failed. Is (%lf), should be (%lf)\n", stateToFurtherAgents.Vs, vsSummed);
                        return -4;
                    }
                    
//                        printf("vsSummed (%lf), stateToFurtherAgents.Vs (%lf)\n", vsSummed, stateToFurtherAgents.Vs);
                }
#endif                
            }

            if(!pParam->batchSize)
                tdlAddStateDone(agentNo, false);            
        }
        
        for(z = startZ; z < endZ; z++)
        {
//            printf("mazeLearn: non-terminal state, z (%i)\n", z);

            if(!pParam->batchSize)
            {
                if(isTerminalStateTo)
                {
                    double mse = 0;
                    int ret = tdlLearn(z, &mse);
                    if(ret)
                    {
                        printf("### mazeLearn: tdlLearn returned error (%i).\n", ret);
                        return -5;
                    }
    //                printf("mse (%lf)\n",mse);
                }
            }
            else
                tdlAddStateDone(z, true);
        }        
    }

    return 0;
}

int mazeSimulateAction(
    tMaze *pMaze,
    tConfigParam *pParam,
    bool dynamicUpwind,
    tPosition posFrom,
    eAction action,
    tPosition *pPosTo,
    bool *pIsInvalidAction
)
{
    *pIsInvalidAction = false;

    tPosition posTmp = posFrom;

    // Perform the action, filter out actions that would move the agent out the maze

    if(action == eAction_north)
    {
        if(posTmp.y <= 0)
        {
            // Moving outside is not allowed, keep the actual position
            *pIsInvalidAction = true;
        }
        else
        {
            // Moved to empty field
            posTmp.y--;
            // Move further up ?
            if(dynamicUpwind && posTmp.y > 0)
            {
                double r;

                if(!checkTerminalState(pMaze, pParam, posTmp, &r))
                    posTmp.y--;
            }
        }
    }
    else if(action == eAction_south)
    {
        if(posTmp.y >= pMaze->heightAndWidth - 1)
        {
            // Moving outside is not allowed, keep the actual position
            *pIsInvalidAction = true;
        }
        else
        {
            // Moved to empty field
            posTmp.y++;
            // Move further up ?
            if(dynamicUpwind && posTmp.y > 0)
            {
                double r;

                if(!checkTerminalState(pMaze, pParam, posTmp, &r))
                    posTmp.y--;
            }
        }
    }
    else if(action == eAction_east)
    {
        if(posTmp.x >= pMaze->heightAndWidth - 1)
        {
            // Moving outside is not allowed, keep the actual position
            *pIsInvalidAction = true;
        }
        else
        {
            // Moved to empty field
            posTmp.x++;
            // Move further up ?
            if(dynamicUpwind && posTmp.y > 0)
            {
                double r;

                if(!checkTerminalState(pMaze, pParam, posTmp, &r))
                    posTmp.y--;
            }
        }
    }
    else if(action == eAction_west)
    {
        if(posTmp.x <= 0)
        {
            // Moving outside is not allowed, keep the actual position
            *pIsInvalidAction = true;
        }
        else
        {
            // Moved to empty field
            posTmp.x--;
            // Move further up ?
            if(dynamicUpwind && posTmp.y > 0)
            {
                double r;

                if(!checkTerminalState(pMaze, pParam, posTmp, &r))
                    posTmp.y--;
            }
        }
    }

    *pPosTo = posTmp;

    return 0;
}

int mazeBestState(
    tMaze *pMaze,
    tConfigParam *pParam,
    int agentNo,
    bool forceExploitation,
    tPosition posFrom,
    unsigned long *pBestStatesLen,
    bool *pIsTerminalState,
    bool *pExploited,
    double *pReward,
    eAction *pAction,
    double *pVsRealTo,
    double *pVsTo2,
    bool isBenchmark,
    tState *pCachedVals,
    void *pHandleCachedVals,
    bool haveCachedVals)
{
    double reward[MAX_NUM_ACTIONS];
    eAction action[MAX_NUM_ACTIONS];
    tState stateToTmp[MAX_NUM_ACTIONS];
    // reward for being in an empty space and not being in a barrier or being outside
    const double reward_none = pParam->reward_draw;
    double rewardTmp = reward_none;

    void *pHandle = NULL;
    
    tState *stateTo = stateToTmp;

    if(pCachedVals != NULL)
    {
        stateTo = pCachedVals;
        pHandle = pHandleCachedVals;
    }
    else
    {
        // Cached values are not available
        
        haveCachedVals = false;
    }
    
    // Verify if the current position of the agent in the MDP is in a terminal state
    *pIsTerminalState = checkTerminalState(pMaze, pParam, posFrom, &rewardTmp);

    if(*pIsTerminalState)
    {
        // State-Action values: There are no actions possible in a terminal state

        *pBestStatesLen = 1;
        pAction[0] = eAction_none;
        pReward[0] = rewardTmp;
        *pExploited = true;

        return 0;
    }
    
    int nStates = 0;
        
    if(pCachedVals == NULL)
    {
        int ret = tdlGetStateValuesPrepare(true, &pHandle, stateTo, MAX_NUM_ACTIONS);
        if(ret)
        {
            printf("### mazeBestState: tdlGetStateValuesPrepare failed (%i)\n", ret);
            return -1;
        }

        if(pHandle == NULL)
        {
            printf("### mazeBestState: tdlGetStateValuesPrepare returned NULL handle\n");        
            return -1;
        }
    }
    
    // Normal transition

    eAction actions[MAX_NUM_ACTIONS];
    int nActions = 0;
    
    mazeGetAllActions(pMaze, posFrom, actions, &nActions, MAX_NUM_ACTIONS);
    
    eAction a;
    int b;
    // Iterate all actions
    for(b = 0; b < nActions; b++)
    {
        a = actions[b];
        
        // Get feature coding for state (posFrom) and action (a)
        if(!haveCachedVals)        
            mazeInputQCodingGet(pMaze, posFrom, stateTo[nStates].s, a);

        // Get reward
        checkTerminalState(pMaze, pParam, posFrom, &rewardTmp);
        
        action[nStates] = a;
        reward[nStates] = rewardTmp;
        nStates++;
    } /* for a */

    eEnsembleDecision ensembleDecision;

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
                ensembleDecision = eEnsembleDecision_single_agent_decision;
        }
        else
        {
            if(pParam->averageDecision)
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
                ensembleDecision = eEnsembleDecision_single_agent_decision;
        }
    }

    eStateDecision stateDecision;
    if(*pIsTerminalState)
    {
        // Forcing exploitation, i.e. be 100% greedy
        stateDecision = eStateDecision_exploitation;
    }
    else if(forceExploitation)
    {
        stateDecision = eStateDecision_exploitation;
    }
    else
    {
        if(pParam->tau > 0)
        {
//            stateDecision = eStateDecision_softmax;
            stateDecision = eStateDecision_softmax_statistics;
        }
        else
            stateDecision = eStateDecision_epsilon_greedy;
    }

    double VsTo[MAX_NUM_ACTIONS];
    unsigned long bestStatesIndices[MAX_NUM_ACTIONS];
    *pBestStatesLen = 0;

    // Select the best Q(s,a) independent of the reward that will be received in the new state and the new state s'
    int ret = tdlGetBestStateEnsemble(pHandle, agentNo, ensembleDecision, stateTo, NULL, nStates, stateDecision, pParam->tau, pParam->epsilon, pExploited, bestStatesIndices, pBestStatesLen, VsTo, pVsTo2, pVsRealTo, NULL, NULL, haveCachedVals);
    if(ret)
    {
        printf("### mazeBestState: tdlGetBestStateEnsemble failed (%i)\n", ret);

        if(pCachedVals == NULL)
            tdlFreeStateValues(pHandle);

        return -2;
    }

    if(*pBestStatesLen > nStates)
    {
        printf("### mazeBestState: *pBestStatesLen > nStates?\n");
        
        if(pCachedVals == NULL)
            tdlFreeStateValues(pHandle);

        return -3;
    }

    unsigned long i;
    for(i = 0; i < *pBestStatesLen; i++)
    {
        int index = bestStatesIndices[i];
        pAction[i] = action[index];
        pReward[i] = reward[index];
        
    }
        
    if(pCachedVals == NULL)
        tdlFreeStateValues(pHandle);

    return 0;
}

int addStateToRepo(
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,
    double probToAddState,
    int *pNStatesAdded,
    tMaze *pMaze,
    int mazeNumber,
    tPosition pos,
    eAction action)
{
    if(pRepo != NULL)
    {
        if((pRepo->nStateRepoMax > 0) &&
            (pRepo->nStateRepo < pRepo->nStateRepoMax))
        {
            bool add = false;
            if(probToAddState == 1.0)
                add = true;
            else if(randValDouble(0,1) <= probToAddState)
                add = true;
            
            if(add)
            {
                *pNStatesAdded = *pNStatesAdded + 1;                
                
                // Search, if the state is already present
                
                bool present = false;
                unsigned long i = 0;
                for(i = 0; i < pRepo->nStateRepo; i++)
                {
                    if(pRepo->pStateInternalRepo[i].mazeNumber != mazeNumber)
                        continue;
                    
                    // Examine Agent Position

                    if(memcmp(&pRepo->pStateInternalRepo[i].pos, &pos, sizeof(pos)))
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
                    eAction actions[MAX_NUM_ACTIONS];
                    int nActions = 0;
                    
                    mazeGetAllActions(pMaze, pos, actions, &nActions, MAX_NUM_ACTIONS);
                    
                    pNActions[i] = nActions;
                }
                
                if(lVerbose)
                {
                    printf("Adding following maze to from-state:\n");
                    printf("x/y (%i/%i), action (%i)\n", pos.x, pos.y, action);
                    outputMaze2(pMaze);
                }
                
                // ... and add the state to the repository
                pRepo->pStateInternalRepo[pRepo->nStateRepo].mazeNumber = mazeNumber;                                
                pRepo->pStateInternalRepo[pRepo->nStateRepo].pos = pos;
                pRepo->pStateInternalRepo[pRepo->nStateRepo].action = action;                                                        
                
                pRepo->nStateRepo++;
            }
        }        
        return 0;
    }
    
    return -1;
}

int calculateTotalReward(
    tMaze *pMaze,
    int mazeNumber,
    tConfigParam *pParam,
    int agentNo,
    double upwindEpsilon,
    int maxSteps,
    tMazeActions pi[][MAX_WIDTH],
    double *pTotalReward,
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,
    double probToAddState,
    int r,
    bool randomStarts,
    bool notUsingPolicy,
    bool decideWithBF,
    int nStatesPerAgentMax,
    int *pNStatesAdded)
{
    int x, y;

    *pTotalReward = 0.0;

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            if(randomStarts)
            {
                /* Randomly set the start position (non-terminal state) */
                do
                {
                    x = randValLong(0, pMaze->heightAndWidth - 1);
                    y = randValLong(0, pMaze->heightAndWidth - 1);
                } while(pMaze->barrier[x][y] ||
                        (x == pMaze->goal.x && y == pMaze->goal.y));
            }
            
            tPosition posFrom;
            posFrom.x = x;
            posFrom.y = y;
            double reward = pParam->reward_draw;

            if(!randomStarts)
            {
                // Starting position must not be a terminal state (barrier or goal)
                if(checkTerminalState(pMaze, pParam, posFrom, &reward))
                    continue;
            }

            if(lVerbose)
                printf("calculateTotalReward: Start pos (%i/%i)\n", x, y);
            
            bool isTerminalState = false;
            int t;

            tMazeActions piCurrent;
            
            t = 0;            
            
            // Always select the best actions until the agent has arrived at a terminal state
            do
            {
                eAction actionArr[MAX_NUM_ACTIONS];
                int ret;
                
                if(decideWithBF)
                {
                    // Best policy for no dynamic up-wind
                    piCurrent = pMaze->piTrue[posFrom.x][posFrom.y];
                }
                else if(notUsingPolicy)
                {
                    bool exploited = false;
                    double rewardArr[MAX_NUM_ACTIONS];
                    double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
                    unsigned long bestStatesIndicesLen = 0;
                    bool isTerminalState2;
                    
                    ret = mazeBestState(pMaze, pParam, agentNo, true, posFrom, &bestStatesIndicesLen, &isTerminalState2, &exploited, rewardArr, actionArr, VsRealToArr, VsTo2Arr, true, NULL, NULL, false);
                    if(ret)
                    {
                        printf("### calculateTotalReward: mazeBestState failed (%i)\n", ret);
                        return -1;
                    }

                    /* Note that action.val can still be Zero, meaning that there 
                    * is no goal reachable by our current position */

                    piCurrent.val = 0;
                    if(!isTerminalState2)
                    {
                        unsigned long i;
                        for(i = 0; i < bestStatesIndicesLen; i++)
                        {
                            if(actionArr[i] == eAction_north)
                                piCurrent.actionNorth = 1;
                            else if(actionArr[i] == eAction_south)
                                piCurrent.actionSouth = 1;
                            else if(actionArr[i] == eAction_east)
                                piCurrent.actionEast = 1;
                            else if(actionArr[i] == eAction_west)
                                piCurrent.actionWest = 1;
                        }
                    }
                }
                else
                    piCurrent = pi[posFrom.x][posFrom.y];
                
                if(!piCurrent.val)
                {
                    printf("### calculateTotalReward: Reached terminal state? Must not happen\n");
                    return -2;
                }

                bool isInvalidAction = false;

                int nActions = 0;

                if(piCurrent.actionNorth)
                {
                    actionArr[nActions] = eAction_north;
                    nActions++;
                }
                if(piCurrent.actionSouth)
                {
                    actionArr[nActions] = eAction_south;
                    nActions++;
                }
                if(piCurrent.actionEast)
                {
                    actionArr[nActions] = eAction_east;
                    nActions++;
                }
                if(piCurrent.actionWest)
                {
                    actionArr[nActions] = eAction_west;
                    nActions++;
                }

                // Randomly select one of the best actions
                unsigned long myrand = randValLong(0, nActions - 1);
                eAction action = actionArr[myrand];

                addStateToRepo(pRepo, pStateProbs, pNActions, probToAddState, pNStatesAdded, pMaze, mazeNumber, posFrom, action);

                if(pRepo != NULL && nStatesPerAgentMax && *pNStatesAdded >= nStatesPerAgentMax)
                {
                    // We have collected enough states, leave the benchmark immediately
                                        
                    *pTotalReward = 0;
                    
                    return 1;
                }
                
                bool dynamicUpwind = isDynamicUpwind(upwindEpsilon);
                
                if(lVerbose && dynamicUpwind)
                    printf("mazeSimulateAction: dynamic up-wind!\n");
                
                tPosition posTo;
                // Simulate the selected action
                ret = mazeSimulateAction(pMaze, pParam, dynamicUpwind, posFrom, action, &posTo, &isInvalidAction);

                if(ret)
                {
                    printf("### calculateTotalReward: mazeSimulateAction failed (%i)\n", ret);
                    return -1;
                }

                if(isInvalidAction)
                {
                    printf("### calculateTotalReward: isInvalidAction, action (%i) in pos (%i/%i)\n", action, posFrom.x, posFrom.y);

                    return -1;
                }

                if(checkTerminalState(pMaze, pParam, posTo, &reward))
                {
                    isTerminalState = true;                                        
                    break;
                }

                posFrom = posTo;
                t++;
            } while(!isTerminalState && t < maxSteps);

            if(isTerminalState)
            {
                if(lVerbose)
                    printf("calculateTotalReward: Reached terminal state at (%i), reward (%lf)\n", t, reward);                
                *pTotalReward += pow(pParam->gamma, (double) t) * reward;
            }
            
            if(randomStarts)
                return 0;
        } /* for x */
    } /* for y */

    return 0;
}

int retrievePolicy(
    tMaze *pMaze,
    tConfigParam *pParam,
    int agentNo,
    tMazeActions pPi[][MAX_WIDTH])
{
    int x, y;

    for(y = 0; y < pMaze->heightAndWidth; y++)
    {
        for(x = 0; x < pMaze->heightAndWidth; x++)
        {
            tPosition posFrom;
            posFrom.x = x;
            posFrom.y = y;

            bool isTerminalState = false, exploited = false;
            double reward[MAX_NUM_ACTIONS];
            eAction action[MAX_NUM_ACTIONS];
            double VsRealTo[MAX_NUM_ACTIONS], VsTo2[MAX_NUM_ACTIONS];
            unsigned long bestStatesIndicesLen = 0;

            int ret = mazeBestState(pMaze, pParam, agentNo, true, posFrom, &bestStatesIndicesLen, &isTerminalState, &exploited, reward, action, VsRealTo, VsTo2, true, NULL, NULL, false);
            if(ret)
            {
                printf("### retrievePolicy: mazeBestState failed (%i)\n", ret);
                return -1;
            }

            /* Note that action.val can still be Zero, meaning that there 
            * is no goal reachable by our current position */

            pPi[x][y].val = 0;
            if(!isTerminalState)
            {
                unsigned long i;
                for(i = 0; i < bestStatesIndicesLen; i++)
                {
                    if(action[i] == eAction_north)
                        pPi[x][y].actionNorth = 1;
                    else if(action[i] == eAction_south)
                        pPi[x][y].actionSouth = 1;
                    else if(action[i] == eAction_east)
                        pPi[x][y].actionEast = 1;
                    else if(action[i] == eAction_west)
                        pPi[x][y].actionWest = 1;
                }
            }
        }
    }

    return 0;
}

int performBenchmarks(
    tMaze *pMaze,
    tConfigParam *pParam,
    bool totalRewards,                      
    int retries,
    double upwindEpsilon,
    int maxSteps,
    int testStartI,
    int testEndI,
    tRepo *pRepo,
    unsigned long *pStateProbs,
    unsigned long *pNActions,    
    double probToAddState,
    bool randomStarts,
    bool notUsingPolicy,
    bool decideWithBF,
    int nStatesPerAgentMax)
{    
    int nMazesTest = testEndI - testStartI + 1;
    
    int z2, endZ3, m;

    if(pParam->averageDecisionBenchmark || pParam->votingDecisionBenchmark)
        endZ3 = 1;
    else
        endZ3 = pParam->agents;

    printf("Perform benchmark for (%i) agents\n", endZ3);
    
    double averageExpectedTotalReward = 0;
    double averageIdenticalStates = 0;

    int nStatesPerAgentMaxTmp = nStatesPerAgentMax;

    if(nStatesPerAgentMaxTmp)
        nStatesPerAgentMaxTmp /= nMazesTest;
    
    // Iterate the validation set
    for(m = testStartI; m <= testEndI; m++)
    {
        if(!totalRewards)
        {
            // Iterate all agents
            for(z2 = 0; z2 < endZ3; z2++)
            {
                tMazeActions pi[MAX_WIDTH][MAX_WIDTH];

                int ret = retrievePolicy(&pMaze[m], pParam, z2, pi);
                if(ret)
                {
                    printf("### performBenchmarks: retrievePolicy returned error (%i)\n", ret);
                    return -2;
                }

                int nCorrectActions = comparePolicies(&pMaze[m], pMaze[m].piTrue, pi);

                averageIdenticalStates += nCorrectActions;
            } /* for z3 */
        }
        else
        {
            // Iterate all agents
            for(z2 = 0; z2 < endZ3; z2++)
            {
                tMazeActions pi[MAX_WIDTH][MAX_WIDTH];

                if(!notUsingPolicy && !decideWithBF)
                {
                    int ret = retrievePolicy(&pMaze[m], pParam, z2, pi);
                    if(ret)
                    {
                        printf("### performBenchmarks: retrievePolicy returned error (%i)\n", ret);
                        return -3;
                    }
                }

                double totalReward = 0;
                int r;

                int nStatesAdded = 0;
                
                for(r = 0; r < retries; r++)
                {
                    double totalRewardTmp = 0;
                    int ret = calculateTotalReward(&pMaze[m], m, pParam, z2, upwindEpsilon, maxSteps, pi, &totalRewardTmp, pRepo, pStateProbs, pNActions, probToAddState, r, randomStarts, notUsingPolicy, decideWithBF, nStatesPerAgentMaxTmp, &nStatesAdded);
                    if(ret < 0)
                    {
                        printf("### performBenchmarks: calculateTotalReward returned error (%i)\n", ret);
                        return -4;
                    }
                    else if(ret == 1)
                    {
//                        printf("Agent (%i) has collected (%i) states with run (%i), aborting benchmark\n", z2, nStatesAdded, r);
                        break; /* for r */
                    }

                    totalReward += (totalRewardTmp / (double) retries);
                }

                averageExpectedTotalReward += totalReward;
            } /* for z2 */
        }
    } /* for m */

    if(!totalRewards)
    {
        averageIdenticalStates /= (double) (endZ3 * nMazesTest);
    //                        printf ("z = %i, t = %li, Identical by highest rank = %lf/%i\n", z, t, averageIdenticalStates, pMaze[m].heightAndWidth * pMaze[m].heightAndWidth * MAX_NUM_ACTIONS);
        printf ("Identical by highest rank = %lf/%i\n", averageIdenticalStates, pMaze[m].heightAndWidth * pMaze[m].heightAndWidth);
    }
    else
    {
        averageExpectedTotalReward /= (double) (endZ3 * nMazesTest);
    //                        printf ("z = %i, t = %li, Identical by highest rank = %lf/%i\n", z, t, averageIdenticalStates, pMaze[m].heightAndWidth * pMaze[m].heightAndWidth * MAX_NUM_ACTIONS);
        printf ("total reward = %lf\n", averageExpectedTotalReward);
    }
    
    return 0;
}

static int calculateConsistencies(
    tMaze *pMaze,
    tConfigParam *pParam,
    int agentNo,
    char *repoErrorsFile,
    char *repoActionNoFile,
    tStateInternal *pState,
    int nStates,                         
    bool forceExploitation,
    bool allActions,
    bool singleActions,
    bool ensembleActions,    
    int retries,
    double upwindEpsilon,
    bool verbose
    )
{
    /* Erläuterungen bezüglich Dynamic Wind:
     * Ist Dynamic Wind eine Eigenschaft eines Zustandes,
     * so ist Wind / Nicht-Wind bereits vor der Aktion des Agenten
     * festgelegt. Hier könnte man dann entscheiden, ob man diese Eigenschaft des Zustandes
     * dem Agenten vorenthält (Partially Observable) oder dem Agenten mitteilt.
     * In diesem Szenario ist pi(.) jedoch deterministisch, d.h. das eine Aktion
     * mit Wahrscheinlichkeit 1.0 in einen Zustand überführt (und nicht mit 0.7 in s1 und 0.3 in s2).
     * 
     * In unserem Fall ist Dynamic Wind in den Zustandsübergangswahrscheinlichkeiten pi(.)
     * beinhaltet: Wind / Nicht-Wind wird erst nach der Aktion eines Agenten festgelegt.
     */    
    
    unsigned i;
    double *yPredicted = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);
    double *yReal = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);
    int *actionNumber = malloc(sizeof(double) * nStates * MAX_NUM_ACTIONS);
        
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
        tMaze *pMaze2 = &pMaze[pState[i].mazeNumber];
        tPosition posFrom = pState[i].pos;
        
        if(verbose)
        {
            printf("Getting state (%i) from repo\n", i);
            printf("Outputting state:\n");
//            outputMaze2(pMaze2);
            printf("Maze number: (%i)\n", pState[i].mazeNumber);
            printf("Player position: (%i)/(%i)\n", posFrom.x, posFrom.y);
        }
               
        eAction action[MAX_NUM_ACTIONS];
        int nActions = 0;
        
        if(allActions)
            mazeGetAllActions(pMaze2, posFrom, action, &nActions, MAX_NUM_ACTIONS);            
        else
            nActions = 1;
        
        typedef struct
        {
            tPosition pos;
            eAction action;
            double value;
        } tValues;
        
        int a;
        for(a = 0; a < nActions; a++)
        {
            double vsFrom = 0;
            double vsTo = 0;
            double reward = 0;
            
            tValues valRepo[2 * MAX_NUM_ACTIONS];
            int nValsRepo = 0;            
            
            eAction actionFrom = 0;
            
            bool exploited = false;
            bool isTerminalState = false;
            eAction actionArr[MAX_NUM_ACTIONS];
            double rewardArr[MAX_NUM_ACTIONS];
            double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
            unsigned long bestStatesIndicesLen = 0;
            
            if(allActions)
            {
                // Iterate all (valid) actions
                actionFrom = action[a];
            }
            else if(ensembleActions)
            {
                actionFrom = pState[i].action;
            }
            else
            {
                // Get the best action a in the current state s (force exploitation / greedy action selection)
                
                int ret = mazeBestState(pMaze2, pParam, agentNo, true, posFrom, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, actionArr, VsRealToArr, VsTo2Arr, true, NULL, NULL, false);
                if(ret)
                {
                    printf("### calculateConsistencies: mazeBestState failed (%i)\n", ret);            
                    tdlFreeStateValues(pHandle);            
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -2;
                }
                
                if(!bestStatesIndicesLen)
                {
                    printf("### calculateConsistencies: bestStatesIndicesLen is Zero\n");            
                    tdlFreeStateValues(pHandle);            
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -3;
                }

                // Randomly select one of the best actions
                unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                actionFrom = actionArr[myrand];

                if(isTerminalState)
                {
                    printf("### calculateConsistencies: Terminal state is starting state?\n");
                    tdlFreeStateValues(pHandle);            
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -4;
                }
            }
                
            if(verbose)
                printf("Non-terminal state s: Best action (%i)\n", actionFrom);                

            // Get the (average) V(s) or Q(s,a)
            
            // Get the (average) state-value of the from-State
                        
            mazeInputQCodingGet(pMaze2, posFrom, state.s, actionFrom);

            int ret = tdlGetStateValues(pHandle, agentNo, &state, 1);
            if(ret)
            {
                printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                tdlFreeStateValues(pHandle);            
                free(actionNumber);
                free(yPredicted);
                free(yReal);
                return -4;
            }

            vsFrom = state.Vs;
            
            typedef struct
            {
                tPosition pos;
                void *pHandle;
                tState pCachedVals[MAX_NUM_ACTIONS];
                bool haveCachedVals;
            } tValues2;

            tValues2 valsRepo2[2];

            unsigned nValsRepo2 = 0;
            
            unsigned k;
            
            for(k = 0;k < 2; k ++)
            {
                valsRepo2[k].haveCachedVals = false;
                
                ret = tdlGetStateValuesPrepare(false, &valsRepo2[k].pHandle, valsRepo2[k].pCachedVals, MAX_NUM_ACTIONS);
                if(ret)
                {
                    printf("### calculateConsistencies: tdlGetStateValuesPrepare failed (%i)\n", ret);
                    tdlFreeStateValues(pHandle);            
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -4;
                }

                if(valsRepo2[k].pHandle == NULL)
                {
                    printf("### calculateConsistencies: tdlGetStateValuesPrepare returned NULL handle\n");        
                    tdlFreeStateValues(pHandle);            
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -4;
                }
            }            
                        
            int r;
            for(r = 0; r < retries; r++)
            {
                // Get the successor state s' due to action a in state s
                
                bool isInvalidAction = false;

                bool dynamicUpwind = isDynamicUpwind(upwindEpsilon);                

                if(lVerbose)
                {
                    if(dynamicUpwind)
                        printf("calculateConsistencies: dynamic up-wind!\n");
                    else
                        printf("calculateConsistencies: no dynamic up-wind\n");
                }
                                
                tPosition posTo;
                // Simulate the selected action
                int ret = mazeSimulateAction(pMaze2, pParam, dynamicUpwind, posFrom, actionFrom, &posTo, &isInvalidAction);

                if(ret)
                {
                    printf("### calculateConsistencies: mazeSimulateAction failed (%i)\n", ret);
                    tdlFreeStateValues(pHandle);            
                    tdlFreeStateValues(valsRepo2[0].pHandle);
                    tdlFreeStateValues(valsRepo2[1].pHandle);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -1;
                }

                if(isInvalidAction)
                {
                    printf("### calculateConsistencies: isInvalidAction, actionFrom (%i) in pos (%i/%i)\n", actionFrom, posFrom.x, posFrom.y);
                    tdlFreeStateValues(pHandle);            
                    tdlFreeStateValues(valsRepo2[0].pHandle);
                    tdlFreeStateValues(valsRepo2[1].pHandle);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -1;
                }

                if(verbose)
                    printf("Simulated action. New player position: (%i)/(%i)\n", posTo.x, posTo.y);
                
                // Find pos
                
                bool found = false;
                
                for(k = 0;k < nValsRepo2; k ++)
                {
                    if(valsRepo2[k].haveCachedVals)
                    {
                        if(!memcmp(&valsRepo2[k].pos, &posTo, sizeof(tPosition)))
                        {
                            found = true;
                            break; /* for */
                        }
                    }
                }
                
                bool haveCachedVals = false;
                tState *pCachedVals = NULL;
                void *pHandleCachedVals = NULL;
                
                if(found)
                {
                    haveCachedVals = valsRepo2[k].haveCachedVals;
                    pCachedVals = valsRepo2[k].pCachedVals;
                    pHandleCachedVals = valsRepo2[k].pHandle;
                }
                else
                {
                    haveCachedVals = valsRepo2[nValsRepo2].haveCachedVals;
                    pCachedVals = valsRepo2[nValsRepo2].pCachedVals;                    
                    pHandleCachedVals = valsRepo2[nValsRepo2].pHandle;
                }
                
                // Get the best action a in the current state s

                exploited = false;
                isTerminalState = false;
                
                ret = mazeBestState(pMaze2, pParam, agentNo, forceExploitation, posTo, &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, actionArr, VsRealToArr, VsTo2Arr, true, pCachedVals, pHandleCachedVals, haveCachedVals);
                if(ret)
                {
                    printf("### calculateConsistencies: mazeBestState failed (%i)\n", ret);            
                    tdlFreeStateValues(pHandle);            
                    tdlFreeStateValues(valsRepo2[0].pHandle);
                    tdlFreeStateValues(valsRepo2[1].pHandle);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -2;
                }
                
                if(!bestStatesIndicesLen)
                {
                    printf("### calculateConsistencies: bestStatesIndicesLen is Zero\n");            
                    tdlFreeStateValues(pHandle);            
                    tdlFreeStateValues(valsRepo2[0].pHandle);
                    tdlFreeStateValues(valsRepo2[1].pHandle);
                    free(actionNumber);
                    free(yPredicted);
                    free(yReal);
                    return -3;
                }

                if(!found)
                {
                    valsRepo2[nValsRepo2].haveCachedVals = true;
                    valsRepo2[nValsRepo2].pos = posTo;
                    nValsRepo2++;
                }

                haveCachedVals = true;

                eAction actionTo;

                // Randomly select one of the best actions
                unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                reward += (rewardArr[myrand] / (double) retries);
                actionTo = actionArr[myrand];

                if(isTerminalState)
                {
                    // No action is possible, get reward

                    if(verbose)
                        printf("Terminal state s': No action is possible, set V(s') or Q(s',*) to zero, reward (%lf)\n", reward);
                }
                else
                {
                    if(verbose)
                        printf("Non-terminal state s': Best action (%i)\n", actionTo);                
                    
                    // Get the (average) state-value of the to-State
                    
                    int k;
                    bool found = false;
                    for(k = 0; k < nValsRepo; k++)
                    {
                        if(memcmp(&valRepo[k].pos, &posTo, sizeof(posTo)))
                            continue;

                        if(valRepo[k].action != actionTo)
                            continue;

                        found = true;
                        break;
                    }
                        
                    if(found)
                        state.Vs = valRepo[k].value;
                    else
                    {
                        mazeInputQCodingGet(pMaze2, posTo, state.s, actionTo);

                        ret = tdlGetStateValues(pHandle, agentNo, &state, 1);
                        if(ret)
                        {
                            printf("### calculateConsistencies: tdlGetStateValues returned error\n");
                            tdlFreeStateValues(pHandle);            
                            tdlFreeStateValues(valsRepo2[0].pHandle);
                            tdlFreeStateValues(valsRepo2[1].pHandle);
                            free(actionNumber);
                            free(yPredicted);
                            free(yReal);
                            return -4;
                        }
                        
                        valRepo[nValsRepo].value = state.Vs;
                        valRepo[nValsRepo].pos = posTo;
                        valRepo[nValsRepo].action = actionTo;
                        
                        nValsRepo++;
                    }
                        
                    vsTo += (state.Vs / (double) retries);
                }
            } /* for r */
        
            tdlFreeStateValues(valsRepo2[0].pHandle);
            tdlFreeStateValues(valsRepo2[1].pHandle);
        
            double V_s_successor = reward + pParam->gamma * vsTo;
            
            yPredicted[nStatesReal] = vsFrom;
            yReal[nStatesReal] = V_s_successor;            
            actionNumber[nStatesReal] = actionFrom;
            
            if(verbose)            
                printf("calculateConsistencies: gamma (%lf), reward (%lf), V(s) = %lf, V(s') = %lf, delta(s') = %lf\n", pParam->gamma, reward, vsFrom, vsTo, V_s_successor);
            
            nStatesReal++;        
        } /* for a */
    }
    
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

bool isUniqueMaze(tMaze *pMaze, int nMazes, int mazeNo)
{
    // Verify that maze i is a unique maze

    unsigned j;

    for(j = 0; j < nMazes; j++)
    {
        if(j == mazeNo)
            continue;
        
        int x, y;
        
        // Is the goal position identical?
        if(pMaze[mazeNo].goal.x != pMaze[j].goal.x ||
            pMaze[mazeNo].goal.y != pMaze[j].goal.y)
            continue;
            
        // Are the barriers identical?
        bool barriersIdentical = true;
        for(y = 0; y < pMaze->heightAndWidth; y++)
        {
            if(!barriersIdentical)
                break;
            
            for(x = 0; x < pMaze->heightAndWidth; x++)
            {
                if(pMaze[mazeNo].barrier[x][y] != pMaze[j].barrier[x][y])
                {
                    barriersIdentical = false;
                    
                    break;
                }
            }
        }

        if(barriersIdentical)
        {
//            printf("Maze (%i) is identical to maze (%i)\n", mazeNo, j);
            return false;        
        }
    }
    
    return true;                    
}

int selectMazePos(tMaze *pMaze, unsigned int nMazes, unsigned int nMazesTest, int testStartI, int testEndI, bool distributeTrainMazes, int agentNo, int nAgents, tPosition *pPos)
{
    /* Randomly select the maze from the training set */
    int nMazesTrain = nMazes - nMazesTest;
    int offset = 0;
    if(distributeTrainMazes)
    {
        nMazesTrain = nMazesTrain / nAgents;
        offset = nMazesTrain * agentNo;
    }
    
    long r = randValLong(0, nMazesTrain - 1);
    int u = 0, m;
    for(m = 0; m < nMazes; m++)
    {
        // testStartI <= m <= testEndI
        if(m >= testStartI && m <= testEndI)
            continue;
        
        if(u >= r + offset)
            break;
        u++;
    }
    
    if(m >= testStartI && m <= testEndI)
        printf("### training maze is (%i) and in the testing set !\n", m);
    
    /* Randomly set the start position (non-terminal state) */
    do
    {
        pPos->x = randValLong(0, pMaze[m].heightAndWidth - 1);
        pPos->y = randValLong(0, pMaze[m].heightAndWidth - 1);
    } while(pMaze[m].barrier[pPos->x][pPos->y] ||
            (pPos->x == pMaze[m].goal.x && pPos->y == pMaze[m].goal.y));
    
    return m;
}

int main (
    int argc,
    char **argv)
{
    char mazeConfFile[100] = "maze.conf";
    bool loadMaze = false;
    bool createMaze = false;
    char mazeFile[100] = "maze.sav";
    int mazeHeightAndWidth = 5;
    int nBarriersMin = 3;
    int nBarriersMax = 5;
    int maxSteps = 20;
    int nMazes = 0;
    int testStartI = 0;
    int testEndI = 99;
    bool verbose = false;
    double upwindEpsilon = 0.3;
    int retries = 10;
    bool isRetriesGiven = false;
    int iterationsPerAgent = 1;
    bool distributeTrainMazes = false;
    bool benchmark = false;
    char savfile[100];
    bool createInitWeights = false;
    unsigned int seed = 0;
    int nStatesRepo = 0;
    int nStatesRepoEvaluate = 0;
    tRepo *pRepo = NULL;
    unsigned long *pStateProbs = NULL;
    unsigned long *pNActions = NULL;
    char repoFile[100] = "mazeStateRepo";
    char repoValuesFile[100] = "mazeStateRepoValues";
    char repoErrorsFile[100] = "mazeStateRepoErrors";
    char repoActionNoFile[100] = "mazeStateRepoActionNumbers";
    char repoFileStateProbs[100] = "mazeStateRepoProbs";
    char repoFileNumberActions[100] = "mazeStateRepoNumberActions";    
    bool calcConsistencies = false;
    double decisionWeight[MAX_MLPS];
    int nDecisionWeights = 0;
    double probToAddState = 0.5;
    bool learnExploredActions = false;
    bool randomStarts = false;
    bool forceExploitationConsistencies = false;
    bool allActions = false;
    bool singleActions = false;
    bool ensembleActions = false;
    bool notUsingPolicy = false;
    bool decideWithBF = false;
    int nStatesPerAgentMax = 0;
    bool outputCollectedStates = false;
    
    int j;
    for(j = 0; j < MAX_MLPS; j++)
        decisionWeight[j] = 0;
    
    strcpy(savfile, "mazeMlpSav");
    
    for (j = 0; j < argc; j++)
    {
        if (strcmp (argv[j], "--conf") == 0)
            sscanf (argv[j + 1], "%s", mazeConfFile);

        if (strcmp (argv[j], "--load") == 0)
        {
            sscanf (argv[j + 1], "%s", mazeFile);
            loadMaze = true;
        }

        if (strcmp (argv[j], "--generate") == 0)
        {
            sscanf (argv[j + 1], "%s", mazeFile);
            sscanf (argv[j + 2], "%i", &nMazes);
            createMaze = true;
        }

        if (strcmp (argv[j], "--mazeHeightAndWidth") == 0)
            sscanf (argv[j + 1], "%i", &mazeHeightAndWidth);

        if (strcmp (argv[j], "--barriers") == 0)
        {
            sscanf (argv[j + 1], "%i", &nBarriersMin);
            sscanf (argv[j + 2], "%i", &nBarriersMax);
        }

        if (strcmp (argv[j], "--maxSteps") == 0)
            sscanf (argv[j + 1], "%i", &maxSteps);

        if (strcmp (argv[j], "--test") == 0)
        {
            sscanf (argv[j + 1], "%i", &testStartI);
            sscanf (argv[j + 2], "%i", &testEndI);
        }

        if (strcmp (argv[j], "--verbose") == 0)
            verbose = true;

        if (strcmp (argv[j], "--upwindEpsilon") == 0)
            sscanf (argv[j + 1], "%lf", &upwindEpsilon);

        if (strcmp (argv[j], "--retries") == 0)
        {
            sscanf (argv[j + 1], "%i", &retries);
            isRetriesGiven = true;
        }
        
        if (strcmp (argv[j], "--iterationsPerAgent") == 0)
            sscanf (argv[j + 1], "%i", &iterationsPerAgent);
        
        if (strcmp (argv[j], "--distributeTrainMazes") == 0)
            distributeTrainMazes = true;
        
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
        
        if(strcmp (argv[j], "--nStatesPerAgentMax") == 0)
            sscanf (argv[j + 1], "%i", &nStatesPerAgentMax);
        
        if(strcmp (argv[j], "--learnExploredActions") == 0)
            learnExploredActions = true;
        
        if(strcmp (argv[j], "--randomStarts") == 0)
            randomStarts = true;
        
        if(strcmp (argv[j], "--forceExploitationConsistencies") == 0)
            forceExploitationConsistencies = true;
        
        if(strcmp (argv[j], "--allActions") == 0)
            allActions = true;

        if(strcmp (argv[j], "--singleActions") == 0)
            singleActions = true;
        
        if(strcmp (argv[j], "--ensembleActions") == 0)
            ensembleActions = true;
        
        if(strcmp (argv[j], "--notUsingPolicy") == 0)
            notUsingPolicy = true;
        
        if(strcmp (argv[j], "--decideWithBF") == 0)
            decideWithBF = true;
        
        if(strcmp (argv[j], "--outputCollectedStates") == 0)
            outputCollectedStates = true;
    }
    
    if(upwindEpsilon > 0.0 && !isRetriesGiven)
    {
        retries = (int) (upwindEpsilon * 10.0);
        if(retries <= 0)
            retries = 1;
    }

    lVerbose = verbose;
    
    tMaze *pMaze = NULL;
    
    // Immediately load the maze and overwrite the mazeHeightAndWidth
    if(loadMaze)
    {
        int ret = restoreMazes(&pMaze, mazeFile, &nMazes);
        if(ret)
        {
            printf("### Unable to load maze file (%s)\n", mazeFile);
            return -10;
        }

        mazeHeightAndWidth = pMaze[0].heightAndWidth;
        printf("Loaded (%i) mazes (%ix%i) from file (%s)\n", nMazes, mazeHeightAndWidth, mazeHeightAndWidth, mazeFile);
    }
    
    tConfigParam param;

    printf("Opening config file (%s)\n", mazeConfFile);
    
    int ret = tdlParseConfig(&param, mazeConfFile);
    if(ret)
    {
        printf("### Unable to open config file (%s)\n", mazeConfFile);
        if(pMaze != NULL)
            free(pMaze);
        return -1;
    }

    if((nDecisionWeights != 0) && 
        (nDecisionWeights < param.agents))
    {
        printf("### Wrong number of decision weights. Have (%i) but should be minimum (%i)\n", nDecisionWeights, param.agents);
        if(pMaze != NULL)
            free(pMaze);
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
    
    if(mazeHeightAndWidth > MAX_WIDTH || mazeHeightAndWidth > MAX_WIDTH_INTERNAL)
    {
        printf("### Internal error: Increase MAX_WIDTH or MAX_WIDTH_INTERNAL\n");
        if(pMaze != NULL)
            free(pMaze);
        return -4;
    }

    if(testStartI < 0 || testEndI < 0 || testStartI > testEndI || testEndI > nMazes - 1 || testStartI > nMazes - 1)
    {
        printf("### testStartI (%i) or testEndI (%i) invalid, must be between 0 and %i\n", testStartI, testEndI, nMazes - 1);
        if(pMaze != NULL)
            free(pMaze);
        return -5;
    }

    if(nStatesRepo > 0 && !benchmark)
    {
        printf("### createStateRepo given but without benchmark, start again with --createStateRepo N --benchmark\n");
        if(pMaze != NULL)
            free(pMaze);
        return -6;
    }
    
    if(nStatesRepoEvaluate > 0 && !benchmark)
    {
        printf("### nStatesRepoEvaluate given but without benchmark, start again with --nStatesRepoEvaluate N --benchmark\n");
        if(pMaze != NULL)
            free(pMaze);
        return -6;
    }

    if(nStatesRepo > 0 && nStatesRepoEvaluate > 0)
    {
        printf("### either define --nStatesRepo or --nStatesRepoEvaluate but not both\n");
        if(pMaze != NULL)
            free(pMaze);
        return -6;
    }
        
    if(decideWithBF && !benchmark)
    {
        printf("### decideWithBF given but without benchmark, start again with --decideWithBF --benchmark\n");
        return -6;
    }
        
    int nMazesTest = testEndI - testStartI + 1;

    printf("Have (%i) test mazes between index (%i) and (%i)\n", nMazesTest, testStartI, testEndI);
    if(distributeTrainMazes)
        printf("Distribute the train mazes to all agents\n");
    else
        printf("Do not distribute the train mazes to all agents, i.e. same train mazes for all agents\n");
        
    printf("Learning state-action values, i.e. Q(s,a)\n");
    printf("Not learning terminal states\n");
    printf("Using next state for rewards\n");

    if(learnExploredActions)
        printf("On-Policy Learning: Learn from exploited and explored actions\n");
    else
        printf("Off-Policy Learning: Only learn from exploited actions\n");
    
    printf("Only for Ensemble-Policy Learning and for benchmark: Use different policy as learned (equally weighted agents)\n");
                
    printf("Upwind epsilon: (%lf), retries (%i)\n", upwindEpsilon, retries);
    if(randomStarts)
        printf("Do random starts on benchmark (only useful for collecting states), retries defines the number of random restarts per maze\n");
    else
        printf("Do not do random starts on benchmark, retries defines the number of retries per maze with all possible starting positions\n");        
    
    unsigned long micro_max_test = pow(mazeHeightAndWidth, 2.0) * (MAX_NUM_ACTIONS + 1);
    if(inputCoding == eInputCoding_default)
        lMlpInputNeurons = pow(mazeHeightAndWidth, 2.0) * 3.0 * 4.0;
    else if(inputCoding == eInputCoding_default2)
        lMlpInputNeurons = pow(mazeHeightAndWidth, 2.0) * 6.0;
    else if(inputCoding == eInputCoding_default3)
        lMlpInputNeurons = pow(mazeHeightAndWidth, 2.0) * 3.0 + 4;
    else if(inputCoding == eInputCoding_default4)
        lMlpInputNeurons = pow(mazeHeightAndWidth, 2.0) * 4.0;

    printf("Input neurons: (%i)\n", lMlpInputNeurons);
    printf("Iterations per agent (%i)\n", iterationsPerAgent);
    
//    unsigned long micro_max_test = MAX_NUM_ACTIONS;

    unsigned long micro_max;
    if(!param.batchSize)
    {
        micro_max = micro_max_test;
//        micro_max = maxSteps;
    }
    else
        micro_max = param.batchSize;
    
    if(nStatesRepoEvaluate > 0)
        micro_max_test = nStatesRepoEvaluate;
    
    if(seed == 0)
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        seed = tv.tv_sec / 1000000UL + tv.tv_usec;
    }
    
    int i;
    char savfileToLoad[100];
    
    if(!decideWithBF)
    {    
        for(i = 0; i < param.agents; i++)
        {
            unsigned int seedAgent = seed + i;
            
            int ret;
            if(!param.stateValueFunctionApproximation)
            {
                printf("### Learning from state value tables is not supported\n");
                if(pMaze != NULL)
                    free(pMaze);
                return -5;
            }
            else
            {
                int nAgents = 1;

                if(param.agents > 1 && param.learnFromAverageStateValues)
                    nAgents = param.agents;

                if(param.linearApproximation)
                    ret = tdlInit(i, nAgents, param.agents, lMlpInputNeurons, micro_max, micro_max_test, NULL, NULL, seedAgent, eStateValueFuncApprox_linear, true, param.gradientPrimeFactor, param.a, param.trainingMode, 0, 0, param.weightedAverage, param.weightCurrentAgent, nDecisionWeights > 0, decisionWeight[i], param.decisionNoise, param.stateValueImprecision, param.tdcOwnSummedGradient, param.replacingTraces);
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
                if(pMaze != NULL)
                    free(pMaze);
                return -5;
            }
            else
                printf("tdlInit: Agent (%i) initialized successful with seed (%u)\n", i, seedAgent);
        } /* for agents */
    }
    
    srandom (seed);

    if(!decideWithBF)
    {
        for(i = 0; i < param.agents; i++)
        {
            if(i == 0)
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
            
            if(pMaze != NULL)
                free(pMaze);

            return 0;
        }
        
        if(nStatesRepoEvaluate > 0)
        {
            // Get states from repo file
            
            tState state[nStatesRepoEvaluate];
            void *pHandle = NULL;
            
            int ret = tdlGetStateValuesPrepare(false, &pHandle, state, nStatesRepoEvaluate);
            if(ret)
            {
                printf("###tdlGetStateValuesPrepare returned error (%i)\n", ret);

                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                if(pMaze != NULL)
                    free(pMaze);

                return -6;            
            }

            if(pHandle == NULL)
            {
                printf("### tdlGetStateValuesPrepare returned NULL handle\n");

                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                if(pMaze != NULL)
                    free(pMaze);

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
                
                free(stateInternal);
                tdlFreeStateValues(pHandle);
                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
                
                if(pMaze != NULL)
                    free(pMaze);

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
                printf("Calculate consistencies with state repo\n");
                
                if(forceExploitationConsistencies)
                    printf("Using different policy as for learning (greedy actions)\n");
                else
                    printf("Using same policy as for learning (may do random actions)\n");
                
                if(allActions)
                    printf("Evaluate all actions\n");
                else if(singleActions)
                    printf("Evaluate best actions of single agent\n");
                else if(ensembleActions)
                    printf("Evaluate best actions of ensemble\n");                
                
                int ret = calculateConsistencies(pMaze, &param, 0, repoErrorsFile, repoActionNoFile, stateInternal, nVals, forceExploitationConsistencies, allActions, singleActions, ensembleActions, retries, upwindEpsilon, verbose);
                if(ret)
                    printf("### calculateConsistencies failed (%i)\n", ret);
            }
            else
            {
                // Get state or state-action values (first agent only)
                
                for(i = 0; i < nVals; i++)
                {
                    tMaze *pMaze2 = &pMaze[stateInternal[i].mazeNumber];
                    
                    ret = mazeInputQCodingGet(pMaze2, stateInternal[i].pos, state[i].s, stateInternal[i].action);
                    
                    if(ret)
                    {
                        printf("### mazeInputQCodingGet / mazeInputVCodingGet failed (%i)\n", ret);

                        free(stateInternal);
                        tdlFreeStateValues(pHandle);
                        for(i = 0; i < param.agents; i++)
                            tdlCleanup(i);
                        
                        if(pMaze != NULL)
                            free(pMaze);

                        return -6;            
                    }
                }
                
                free(stateInternal);            

                if(tdlGetStateValues(pHandle, 0, state, nVals))
                {
                    printf("### tdlGetStateValues error\n");

                    tdlFreeStateValues(pHandle);
                    for(i = 0; i < param.agents; i++)
                        tdlCleanup(i);
                    
                    if(pMaze != NULL)
                        free(pMaze);
                    
                    return -6;            
                }
                
                // Write state or state-action values to file
                
                fp = fopen (repoValuesFile, "w");
                if(fp == NULL)
                {
                    printf("### fopen failed with file (%s)\n", repoValuesFile);
                    
                    tdlFreeStateValues(pHandle);
                    for(i = 0; i < param.agents; i++)
                        tdlCleanup(i);
                    
                    if(pMaze != NULL)
                        free(pMaze);

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

            tdlFreeStateValues(pHandle);
            
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);

            if(pMaze != NULL)
                free(pMaze);

            return 0;                
        }        
    }    
    
    if(createMaze)
    {
        if(!loadMaze)
        {
            pMaze = malloc(nMazes * sizeof(tMaze));
            if(pMaze == NULL)
            {
                printf("### malloc failed\n");

                if(!decideWithBF)
                {
                    for(i = 0; i < param.agents; i++)
                        tdlCleanup(i);
                }
                return -6;
            }
        }
        
        // Generate (nMazes) unique mazes whose single goal position is reachable
        // starting from any non-terminal state (barriers and the goal are terminal states)
        
        unsigned startI = 0, endI = nMazes;
        
        if(loadMaze)
        {
            startI = testStartI;
            endI = testEndI + 1;
            
            printf("Overwriting the mazes from (%i) to (%i)\n", startI, endI - 1);
        }
        
        for(i = startI; i < endI; i++)
        {
            pMaze[i].heightAndWidth = mazeHeightAndWidth;
            
            int nBarriers = randValLong(nBarriersMin, nBarriersMax);
            
            printf("Number of barriers for maze (%i): %i\n", i, nBarriers);
            
            do
            {
                // Generate maze
                generateMaze(&pMaze[i], nBarriers);

                if(!isUniqueMaze(pMaze, i, i))
                    continue;
                                
                calculateTrueStateValues(&pMaze[i], param.reward_won, param.reward_lost, param.gamma, 1000);
            } while(!isGoalReachable(&pMaze[i]));

            if(verbose)
            {
                printf("Output maze (%i):\n", i);
                outputMaze(&pMaze[i]);
            }
        }
        
        int ret = saveMazes(pMaze, mazeFile, nMazes);
        if(ret)
        {
            printf("### Unable to save maze file (%s)\n", mazeFile);
            if(!decideWithBF)
            {
                for(i = 0; i < param.agents; i++)
                    tdlCleanup(i);
            }
    
            return -7;
        }
    }
    else if(loadMaze)
    {
        int nUniqueMazes = 0;
        int nDuplicatedMazes = 0;
        
        for(i = 0; i < nMazes; i++)
        {
            pMaze[i].heightAndWidth = mazeHeightAndWidth;
            
            // Check if the maze is unique
            
            bool isUnique = isUniqueMaze(pMaze, nMazes, i);
            
            if(!isUnique)
                nDuplicatedMazes++;
            else
                nUniqueMazes++;
            
            if(verbose)
            {
                printf("Output maze (%i):\n", i);
                outputMaze(&pMaze[i]);
            }
        }
        
        printf("Number of unique mazes (%i), number of duplicated mazes (%i)\n", nUniqueMazes, nDuplicatedMazes);        
    }
    else
    {
        printf("### Either specify --generate or --load\n");
        
        if(!decideWithBF)
        {
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);
        }
    
        return -8;
    }
    
    if(benchmark)
    {
        if(nStatesRepo > 0)
        {
            tRepo repo;
            pRepo = &repo;

            pRepo->pStateInternalRepo = (tStateInternal *) malloc(sizeof(tStateInternal) * nStatesRepo);

            if(pRepo->pStateInternalRepo == NULL)
            {
                printf("### allocate pRepo failed\n");
                return -9;                
            }
            
            pRepo->nStateRepoMax = nStatesRepo;
            pRepo->nStateRepo = 0;

            pStateProbs = (unsigned long *) malloc (sizeof(unsigned long) * nStatesRepo);
            
            if(pStateProbs == NULL)
            {
                printf("### allocate pStateProbs failed\n");
                return -9;                
            }
            
            memset(pStateProbs, 0, sizeof(unsigned long) * nStatesRepo);

            pNActions = (unsigned long *) malloc (sizeof(unsigned long) * nStatesRepo);
        
            if(pNActions == NULL)
            {
                printf("### allocate pNActions failed\n");
                return -9;                
            }
        
            memset(pNActions, 0, sizeof(unsigned long) * nStatesRepo);
        }
                
        // Do not start training, just evaluate the (trained and restored) value function approximator
        ret = performBenchmarks(pMaze, &param, true, retries, upwindEpsilon, maxSteps, testStartI, testEndI, pRepo, pStateProbs, pNActions, probToAddState, randomStarts, notUsingPolicy, decideWithBF, nStatesPerAgentMax);
                
        // Write state coding repository
        if(pRepo != NULL)
        {
            printf("pRepo->nStateRepo (%i), pRepo->nStateRepo (%i)\n", pRepo->nStateRepo, pRepo->nStateRepo);
            
            if(!ret && pRepo->nStateRepo > 0)
            {
                FILE *fp;
                
                fp = fopen (repoFile, "w");
                if (fp != NULL)
                {
                    if (fwrite ((const void *) pRepo->pStateInternalRepo, sizeof(tStateInternal), pRepo->nStateRepo, fp) != pRepo->nStateRepo)
                        printf("### fwrite failed\n");
                    
                    fclose(fp);
                    
                    printf("Written (%i) states to file (%s)\n", pRepo->nStateRepo, repoFile);
                }
                
                fp = fopen (repoFileStateProbs, "w");
                if (fp != NULL)
                {
                    if(outputCollectedStates)
                        printf("Outputting collected states:\n");
                    
                    for(i = 0; i < pRepo->nStateRepo; i++)
                    {
                        char str[100];

                        snprintf(str, 100, "%lu\n", pStateProbs[i]);

                        fwrite(str, sizeof(char), strlen(str), fp);
                        
                        if(outputCollectedStates)
                        {
                            printf("Output maze (%i):\n", i);
                            outputMaze3(&pMaze[pRepo->pStateInternalRepo[i].mazeNumber], pRepo->pStateInternalRepo[i].pos);
                        }
                    }

                    fclose(fp);
                    
                    printf("Written (%i) state probs to file (%s)\n", pRepo->nStateRepo, repoFileStateProbs);
                }

                fp = fopen (repoFileNumberActions, "w");
                if (fp != NULL)
                {
                    for(i = 0; i < pRepo->nStateRepo; i++)
                    {
                        char str[100];

                        snprintf(str, 100, "%lu\n", pNActions[i]);

                        fwrite(str, sizeof(char), strlen(str), fp);
                    }

                    fclose(fp);
                    
                    printf("Written (%i) number actions to file (%s)\n", pRepo->nStateRepo, repoFileNumberActions);
                }
            }
        }
        
        if(!decideWithBF)
        {
            for(i = 0; i < param.agents; i++)
                tdlCleanup(i);
        }
        
        free(pMaze);

        if(nStatesRepo > 0)
        {
            free(pRepo->pStateInternalRepo);
            free(pStateProbs);
        }
        
        if(ret)
        {
            printf("### performBenchmarks failed (%i)\n", ret);
            return -9;
        }
        else
            return 0;        
    }
    
    int z = 0;

    int mazeI[param.agents];
    
    tPosition pos[param.agents];
    unsigned long lastGoal[param.agents];

    tPosition posLast[param.agents];
    eAction actionLast[param.agents];
    bool haveLastObservation[param.agents];
    for(z = 0; z < param.agents; z++)
        haveLastObservation[z] = false;
    
    for(z = 0; z < param.agents; z++)
    {
        tPosition posTmp;
        
        mazeI[z] = selectMazePos(pMaze, nMazes, nMazesTest, testStartI, testEndI, distributeTrainMazes, z, param.agents, &posTmp);        
        pos[z] = posTmp;

        lastGoal[z] = 0;

        if(verbose)
            printf("Start pos agent (%i) in maze (%i): %i/%i\n", z, mazeI[z], pos[z].x, pos[z].y);
    }

    unsigned long t2, t2max = param.iterations / iterationsPerAgent;
    z = 0;
    
    for(t2 = 0; t2 < t2max; t2++)
    {
        for(z = 0; z < param.agents; z++)
        {
            unsigned long t3;
            for(t3 = 0; t3 < iterationsPerAgent; t3++)
            {
                unsigned long t = t2 * iterationsPerAgent + t3;
                
                if(t > 0)
                {
                    bool isTerminalState = false;
                    bool exploited = false;
                    double rewardArr[MAX_NUM_ACTIONS];
                    eAction actionArr[MAX_NUM_ACTIONS];
                    double VsRealToArr[MAX_NUM_ACTIONS], VsTo2Arr[MAX_NUM_ACTIONS];
                    unsigned long bestStatesIndicesLen = 0;

                    double *pVsTo2Arr = NULL;
                    
                    if(param.learnFromAverageStateValues && param.agents > 1)
                        pVsTo2Arr = VsTo2Arr;
                    
                    /* Observe the current reward, 
                    * choose the (best) action according to the current policy and decision mechanism
                    * and observe the successor state. Then choose the best action in the successor state */

                    int ret = mazeBestState(&pMaze[mazeI[z]], &param, z, false, pos[z], &bestStatesIndicesLen, &isTerminalState, &exploited, rewardArr, actionArr, VsRealToArr, pVsTo2Arr, false, NULL, NULL, false);
                    if(ret)
                    {
                        goto exitOut;
                        break;
                    }

                    if(!bestStatesIndicesLen)
                    {
                        goto exitOut;
                        break;
                    }

                    double reward;
                    eAction action;
                    double VsRealTo, VsTo2 = 0;

                    // Randomly select one of the best actions
                    unsigned long myrand = randValLong(0, bestStatesIndicesLen - 1);
                    reward = rewardArr[myrand];
                    action = actionArr[myrand];
                    VsRealTo = VsRealToArr[myrand];
                    if(pVsTo2Arr != NULL)
                        VsTo2 = VsTo2Arr[myrand];                        

                    tPosition posTo = pos[z];

                    if(!isTerminalState)
                    {
                        bool isInvalidAction = false;

                        bool dynamicUpwind = isDynamicUpwind(upwindEpsilon);
                        
                        if(lVerbose && dynamicUpwind)
                            printf("mazeSimulateAction: dynamic up-wind!\n");

                        // Simulate the selected action
                        ret = mazeSimulateAction(&pMaze[mazeI[z]], &param, dynamicUpwind, pos[z], action, &posTo, &isInvalidAction);
                        if(ret)
                        {
                            printf("### mazeSimulateAction failed (%i)\n", ret);
                            goto exitOut;
                        }

                        if(isInvalidAction)
                        {
                            printf("### isInvalidAction, action (%i) in pos (%i/%i)\n", action, pos[z].x, pos[z].y);
                            goto exitOut;
                        }
                    }

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
                    
                    if(haveLastObservation[z])
                    {
                        ret = mazeLearn(&pMaze[mazeI[z]], &param, z, false, exploited, reward, posLast[z], actionLast[z], pos[z], action, isTerminalState, VsRealTo, VsTo2, learnExploredActions);
                        if(ret)
                        {
                            printf("### mazeLearn returned error (%i)\n", ret);
                            goto exitOut;
                        }
                    }
                    else if((z > 0) && (z == param.agents - 1) && (!param.updateWeightsImmediate))
                    {
                        int z2;
                        for(z2 = 0; z2 < param.agents; z2++)
                        {
                            if(param.batchSize > 0)
                            {
                                tdlAddStateDone(z2, true);
                            }
                        }
                    }
                    
                    posLast[z] = pos[z];
                    actionLast[z] = action;
                    haveLastObservation[z] = true;
                    
                    pos[z] = posTo;

                    bool isTimeout = false;
                    if(t - lastGoal[z] > maxSteps)
                    {
                        isTimeout = true;
                        ret = tdlCancelEpisode(z);
                        if(ret)
                        {
                            printf("### tdlCancelEpisode failed (%i)\n", ret);
                            goto exitOut;
                        }
                    }                                    
                    
                    if(isTerminalState || isTimeout)
                    {
                        haveLastObservation[z] = false;
                        
                        if(isTerminalState && verbose)
                        {
                            if(reward == param.reward_won)
                                printf("Goal reached at t = %li\n", t);
                            else if(reward == param.reward_lost)
                                printf("Barrier reached at t = %li\n", t);
                        }

                        tPosition posTmp;
                        
                        mazeI[z] = selectMazePos(pMaze, nMazes, nMazesTest, testStartI, testEndI, distributeTrainMazes, z, param.agents, &posTmp);                        
                        pos[z] = posTmp;

                        if(verbose)
                            printf("Agent (%i): Selected maze (%i)\n", z, mazeI[z]);
                        
                        if(!isTerminalState && verbose)
                            printf("lastGoal (%li), t (%li)\n", lastGoal[z], t);

                        lastGoal[z] = t;

                        if(verbose)
                            printf("Start pos agent (%i) in maze (%i): %i/%i\n", z, mazeI[z], pos[z].x, pos[z].y);   
                    }
                } /* t > 0 */                    
            } /* for t3 */
        } /* for z */
    } /* for t2 */

    char filename[100];

exitOut:
    
    for(i = 0; i < param.agents; i++)
    {
        sprintf (filename, "%s_%i", savfile, i);

        printf("Save MLP to file (%s)\n", filename);

        tdlSaveNet (i, filename);
    }
    
    for(i = 0; i < param.agents; i++)
        tdlCleanup(i);

    free(pMaze);

    return 0;
}
