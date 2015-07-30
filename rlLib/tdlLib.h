#include "mlpLib.h"

#define MAX_MLPS    100

typedef struct
{
    /* General parameters for all RL applications (some may be unused) */
    double reward_won, 
           reward_lost, 
           reward_draw;
    double gamma;
    double lambda;
    double alpha;
    unsigned long iterations;
    double epsilon;
    double epsilonBase;
    double tau;
    double epsilonOpp;
    double epsilonBaseOpp;
    double tauOpp;
    double stateValueImprecision;
    bool errorStatistics;
    double errorEpsilon;
    int totalErrorIterations;
    int groupErrorIterations;
    int expectedTotalRewardIterations;
    int bellmanErrorIterations;
    bool stateValueFunctionApproximation;
    double minA;
    double maxA;
    bool linearApproximation;
    double a;
    int trainingMode;
    char conffile[100];
    char savfile[100];
    double beta;
    double gradientPrimeFactor;
    bool normalizeGradientByAgents;
    int batchSize;
    int agents;
    bool updateWeightsImmediate;
    bool weightedAverage;
    bool weightedDecisions;
    bool learnFromAverageStateValues;
    bool averageDecision;
    bool votingDecision;
    bool averageDecisionBenchmark;
    bool votingDecisionBenchmark;
    double decisionNoise;
    uint32_t seed[100];
    int nWorstStates;
    bool normalizeLearningRate;
    double weightCurrentAgent;
    bool tdcOwnSummedGradient;
    bool replacingTraces;
} tConfigParam;

/* state presentation */
typedef struct
{
    /* state coding s */
    void *s;

    /* state value V(s) */
    double Vs;
} tState;

typedef enum
{
    eStateValueFuncApprox_none,
    eStateValueFuncApprox_linear,
    eStateValueFuncApprox_nonlinear_mlp,
} eStateValueFuncApprox;

typedef enum
{
    eEnsembleDecision_no_ensemble,
    eEnsembleDecision_single_agent_decision,
    eEnsembleDecision_average_state_values_decision,
    eEnsembleDecision_voting_decision,
    eEnsembleDecision_average_state_values_decision_weighted,
    eEnsembleDecision_voting_decision_weighted    
} eEnsembleDecision;

typedef enum
{
    eStateDecision_epsilon_greedy,
    eStateDecision_softmax,
    /* Choose states with probability linear to their state-values */
    eStateDecision_linear, /* TODO: Implement */
    eStateDecision_exploitation,
    /* Choose the best state with probability impactOfBestState (0.7) and the rest states 
     * with 1.0 - impactOfBestState (0.3) / number of rest states */
    eStateDecision_exploitation_partial,
    eStateDecision_exploration,
    eStateDecision_softmax_statistics
} eStateDecision;

int tdlParseConfig(tConfigParam *pParam, char *filename);

int tdlInit(int n,
            int nAgentsLearningFromAverageSV,
            int nAgents,
            int m_request,
            int micro_max_request,
            int micro_max_test_request,
            char *configfile,
            char *netfile,
            unsigned int seed,
            eStateValueFuncApprox stateValueFuncApprox,
            bool needs_x_prime,
            double gradientPrimeFactor,
            double weightInterval,
            int trainingMode,
            double minStateValue,
            double maxStateValue,
            bool weightedAverage,
            double weightCurrentAgent,
            bool haveDecisionWeights,
            double decisionWeight,
            double decisionNoise,
            double stateValueImprecision,
            bool tdcOwnSummedGradient,
            bool replacingTraces
           );

int tdlCleanup(int n);

int tdlGetParam(
    int n,
    bool *pNormalizeLearningRate,
    double *pAlpha,
    double *pBeta,
    double *pGamma,
    double *pLambda);

int tdlSetParam(
    int n,
    bool normalizeLearningRate,
    double alpha_request,
    double beta_request,
    double gamma_request,
    double lambda_request);

int tdlSetDecisionWeightCurrentAgent(
    double weightCurrentAgent);

int tdlSetMseIterations(
    int n,
    unsigned long mseIterations_request);

int tdlSaveNet(
    int n,
    char *netfile);

int tdlAddState(
        int n,
        tState *pS,
        tState *pS_prime,
        tState *pS_prime2,
        double reward,
        double alphaDiscount);

int tdlGetNumberStates(
    int n,
    unsigned long *pNStates);

int tdlUpdateAlphaDiscount(
    int n,
    int micro,
    double alphaDiscount);

int tdlAddStateDone(
    int n,
    bool autoLearning);

int tdlLearn(
    int n,
    double *pMse);

int tdlCancelEpisode(
    int n);

int tdlClearGradientTrace(
    int n);

int tdlGetStateValuesPrepare(
    bool useInternalMem,
    void **pHandle,                             
    tState *pS,
    unsigned long micro);

int tdlFreeStateValues(
    void *pHandle);

int tdlGetStateValues(
    void *pHandle,
    int n,
    tState *pS,
    unsigned long micro);

int tdlGetBestState(
    tState *pS,
    double *pSTTV,
    unsigned long micro,
    eStateDecision stateDecision,
    double decisionNoise,
    double stateValueImprecision,
    double tau,
    unsigned long *pI,
    double *pVs,
    double *pVsR,
    double *pMinDiff,
    double *pMaxDiff
    );

int tdlGetBestState2(
    tState *pS,
    double *pSTTV,
    unsigned long micro,
    eStateDecision stateDecision,
    double decisionNoise,
    double stateValueImprecision,
    double tau,
    unsigned long *pI,
    double *pVs,
    double *pVsR,
    unsigned long *pNStates,
    double *pMinDiff,
    double *pMaxDiff
    );

int tdlGetBestStateEnsemble(
    void *pHandle,
    int n,
    eEnsembleDecision ensembleDecision,
    tState *pS,
    double *pSTTV,
    unsigned long micro,
    eStateDecision stateDecision,
    double tau,
    double epsilon,
    bool *pExploited,
    unsigned long *pBestStatesIndices,
    unsigned long *pBestStatesLen,
    double *pVs,
    double *pVs2,
    double *pVsR,
    double *pMinDiff,
    double *pMaxDiff,
    bool haveCachedVals
    );

void tdlCalcTotalError(
    double *VgivenAll,
    double *VcurrentAll,
    bool *Visvalid,
    int nStatesAll,
    int *nStates,
    double *error,
    double *pSummedError,
    double *pLargestDiff, 
    int *pnDiffs,
    double *pVar);
    
/* precision: 10^(-3) */
/* If defined then the total error is much lower at the beginning */
#undef NUMERICAL_FIXES
#define NUMERICAL_PRECISION 0.001

double randValDouble (
    double min,
    double max);

int32_t randValLong (
    int32_t min,
    int32_t max);

double signf(double v);

int bubbleRank(
    double *pV,
    int N,
    int descending,
    int *pIndices,
    int *pRank);
