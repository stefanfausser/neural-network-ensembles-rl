/**
 * \file mlpLib.h
 * \brief m * h * n Multi-Layer Perceptron (MLP) Error Backpropagation library
 *
 * \author Stefan Fausser
 */

#include <stdint.h>

#include "commonNeuralLib.h"

#define MAX_HIDDEN_LAYERS 10

/**
 * weight update mode enumeration
 */
enum weightUpdateMode
{
    ONLINE_MODE = 0,
    BATCH_MODE = 1,
    MOMENTUMTERM_MODE = 2,
    SUPERSAB_MODE = 3,
    RPROPM_MODE = 4,
    QUICKPROP_MODE = 5,
    RPROPP_MODE = 6,
    TD_MODE = 7,
    RG_MODE = 8,
    TDC_MODE = 9
};

/**
 * structure including modifyable mlp parameters (using setMlpParams function)
 */
struct mlp_param
{
    /**
     * maximum number of episode iterations that should be trained
     */
    uint32_t maxIterations;

    /**
     * learning rate for hidden layer(s)
     */
    double eta1;

    /**
     * learning rate for output layer
     */
    double eta2;

    int etaNormalize;
    
    /**
     * maximum error rate
     */
    double epsilon;

    /**
     * parameter in fermi function, has to be > 0
     */
    double beta;

    /**
     * parameter in supersab weight update method, updates learning rate (positive)
     */
    double eta_pos;

    /**
     * parameter in supersab weight update method, updates learning rate (negative)
     */
    double eta_neg;

    /**
     * parameter in supersab weight update method, starting learning rate
     */
    double eta_start;

    /**
     * parameter in supersab weight update method maximum learning rate
     */
    double eta_max;

    /**
     * parameter in supersab weight update method minimum learning rate
     */
    double eta_min;

    /**
     * parameter in momentumterm update method, has to be between 0 and 1
     */
    double alpha;

    /**
     * parameter in fermi function, maximum beta
     */
    double beta_max;

    /**
     * holds weight update mode
     */
    enum weightUpdateMode trainingMode;

    /**
     * turns on verbose output
     */
    int verboseOutput;
};

/**
 * structure including parameters for initializing a mlp
 */
struct mlp_init_values
{
    /**
     * number of hidden layers
     */
    uint16_t nrHiddenLayers;

    /**
     * number of input neurons
     */
    uint16_t m;

    /**
     * number of neurons in hidden layer
     */
    uint16_t h[MAX_HIDDEN_LAYERS];

    /**
     * number of output neurons
     */
    uint16_t n;

    /**
     * transfer function type for hidden layer
     */
    uint8_t transFktTypeHidden;

    /**
     * transfer function type for output layer
     */
    uint8_t transFktTypeOutput;
    
    int hasThresholdOutput;
};

int mlpLibInit (
    uint16_t nrMlps);

int mlpLibDeinit (
    );

/**
 * \brief initialize mlp network
 * \param pMlpInitVals pointer to mlp_init_values structure
 * \param a each weight and bias of each neuron will be initialized between -a and +a
 * \param micro_max maximum number of training samples
 * \param seed seed for initializing the pseudo random number generator
 * \returns mlpfd on success (>=0), any other value otherwise
 */
int initializeMlpNet (
    struct mlp_init_values *pMlpInitVals,
    struct mlp_param *pMlpParam,                      
    double a,
    double aOut,
    int weightNormalizedInitialization,
    int thresholdZeroInitialization,
    uint32_t micro_max,
    unsigned int seed);

/**
 * \brief cleanup mlp network
 * \returns 0 on success, any other value otherwise
 */
int cleanupMlpNet (
    int mlpfd);

/**
 * \brief output mlp network (current implementation does this to standard output)
 * \returns 0 on success, any other value otherwise
 */
int outputWeights (
    int mlpfd);

int outputWeightsStatistics (
    int mlpfd);

/**
 * \brief train mlp network
 * \param mlpfd mlp number
 * \param micro number of training samples
 * \param pData pointer to train_data structure.
 * \n Note that training signals should be placed in y.
 * \returns >=0: MSE, < 0: Error
 */
double mlp (
    int mlpfd,
    uint32_t micro_max,
    struct train_data *pData,
    // Attention: Following arguments are only used in combination with Reinforcement Learning.
    // Currently, this is for the learning modes: RG_MODE, TDC_MODE
    double alpha,
    double alpha2,
    double gamma,
    double lambda,
    bool updateFirstLayer,
    bool updateSecondLayer,
    double *sampleAlphaDiscount,
    double gradientPrimeFactor,
    double normFactor,
    bool checkGradients,
    bool tdcOwnSummedGradient,
    bool replacingTraces);

/**
 * \brief test mlp network
 * \param micro number of training samples
 * \param pData pointer to train_data structure.
 * \returns 0 on success, any other value otherwise
 */
int mlpOutput (
    int mlpfd,
    unsigned long micro_max,
    uint32_t offset,
    struct train_data *pData);

/**
 * \brief read mlp parameters (non-modifyable and modifyable) from file
 * \param filename pointer to filename-string
 * \param pMlpInitVals pointer to mlp_init_values structure
 * \param pMlpP pointer to mlp network parameter structure
 * \param seed [out] pointer to seed
 * \param a [out] pointer to a
 * \returns 0 on success, any other value otherwise
 */
int openMlpParameterFile (
    const char *filename,
    struct mlp_init_values *pMlpInitVals,
    struct mlp_param *pMlpP,
    uint32_t *seed,
    double *a,
    double *aOut,
    int *weightNormalizedInitialization,
    int *thresholdZeroInitialization);

/**
 * \brief restore mlp network state
 *
 * use this function INSTEAD of initializeMlpNet to initialize
 * a prior saved mlp network. mlpRestore also calls setMlpParam
 * to set the from file extracted modifyable mlp parameters.
 *
 * \param filename pointer to filename-string
 * \param micro_max maximum number of training samples
 * \param seed seed for initializing the pseudo random number generator
 * \returns 0 on success, any other value otherwise
 */
int mlpRestore (
    struct mlp_param *pMlpParam,
    const char *filename,
    uint32_t micro_max,
    unsigned int seed);

/**
 * \brief save mlp network state
 * \param filename pointer to filename-string
 * \returns 0 on success, any other value otherwise
 */
int mlpSave (
    int mlpfd,
    const char *filename);

/**
 * \brief set modifyable mlp network parameter
 * \param pMlpP pointer to mlp network parameter structure
 * \returns 0 on success, any other value otherwise
 */
int setMlpParam (
    int mlpfd,
    struct mlp_param *pMlpP);

/**
 * \brief get modifyable mlp network parameter
 * \param pMlpP pointer to mlp network parameter structure
 * \returns 0 on success, any other value otherwise
 */
int getMlpParam (
    int mlpfd,
    struct mlp_param *pMlpP);

/**
 * \brief get non-modifyable mlp network parameter
 * \param pMlpInitVals pointer to mlp_init_values structure
 * \returns 0 on success, any other value otherwise
 */
int getMlpInitValues (
    int mlpfd,
    struct mlp_init_values *pMlpInitVals);

int clearGradientTrace (
    int mlpfd);
