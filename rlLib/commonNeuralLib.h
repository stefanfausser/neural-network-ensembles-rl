typedef enum
    { true = 1, false = 0 } bool;

struct train_data
{
    /**
     * input values
     */
    double **x;

    /**
     * input values
     */
    double **x_prime;

    /**
     * output values / training signal values
     */
    double **y;

    /**
     * output values / training signal values of further agents
     * reserved for ensemble learning
     */
    double **y2;

    double *reward;
    
    double *gamma;

    double *delta;
    
    bool *hasXPrime;
    
    unsigned long samples;
    unsigned long xSize;
    unsigned long ySize;
};

/**
 * \brief allocate memory for training-pairs
 * \param pData pointer to train_data structure.
 * \param micro_max maximum number of training-pairs
 * \param m input dimension
 * \param n output dimension
 * \returns 0 on success, any other value otherwise
 */
int allocateTrainData (
    struct train_data *pData,
    unsigned long micro_max,
    unsigned short m,
    unsigned short n,
    bool allocatePrime);

int getAndAllocateTrainDataFile (
    char *filename,
    struct train_data *pData,
    unsigned short *m,
    unsigned short *n,
    unsigned long *micro_max);

/**
 * \brief free memory for training-pairs
 * \param micro_max maximum number of training-pairs.
 * \n Note that this value MUST be equal to the micro_max value used in allocateRbfData.
 * \param pData pointer to train_data structure.
 * \returns 0 on success, any other value otherwise
 */
int freeTrainData (
    struct train_data *pData);

void outputTrainDataStdout(
    struct train_data *pData);
    
int getTrainDataLen (
    struct train_data *pData,
    unsigned long *len);

int copyTrainData (
    struct train_data *pDst,
    struct train_data *pSrc,
    unsigned long offDst,
    unsigned long offSrc,
    unsigned long len);

int copyTrainDataX (
    struct train_data *pDst,
    struct train_data *pSrc,
    unsigned long offDst,
    unsigned long offSrc,
    unsigned long len);

int copyTrainDataXPrime (
    struct train_data *pDst,
    struct train_data *pSrc,
    unsigned long offDst,
    unsigned long offSrc,
    unsigned long len);

int copyTrainDataY (
    struct train_data *pDst,
    struct train_data *pSrc,
    unsigned long off,
    unsigned long offSrc,
    unsigned long len);
