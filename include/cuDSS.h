#ifndef CUDSS_PUBLIC_HEADER_H
#define CUDSS_PUBLIC_HEADER_H

#include <stdint.h>        // int64_t
#include <library_types.h> // for cudaDataType and cudaDataType_t

#define CUDSS_VERSION_MAJOR 0
#define CUDSS_VERSION_MINOR 1
#define CUDSS_VERSION_PATCH 0
#define CUDSS_VERSION (CUDSS_VERSION_MAJOR * 10000 + \
                       CUDSS_VERSION_MINOR *  100 +  \
                       CUDSS_VERSION_PATCH)

#if !defined(CUDSSAPI)
#    if defined(_WIN32)
#        define CUDSSAPI __stdcall
#    else
#        define CUDSSAPI
#    endif
#endif

extern "C" {

struct cudssContext;
typedef struct cudssContext *cudssHandle_t; // the library context handle 

struct cudssMatrix;
typedef cudssMatrix* cudssMatrix_t; // opaque generic matrix struct (for both dense and sparse matrices)

struct cudssData;
typedef cudssData* cudssData_t; // opaque object type which stores internal data (like LU factors) as well as some user-provided pointers

struct cudssConfig;
typedef cudssConfig* cudssConfig_t; // opaque object type which stores solver settings (e.g., algorithmic knobs)

typedef enum cudssConfigParam_t {
    CUDSS_CONFIG_REORDERING_ALG,
    CUDSS_CONFIG_FACTORIZATION_ALG,
    CUDSS_CONFIG_SOLVE_ALG,
    CUDSS_CONFIG_MATCHING_TYPE,     // to enable/disable matching (only off)
    CUDSS_CONFIG_SOLVE_MODE,        // for transpose, conj transpose solves (only off)
    CUDSS_CONFIG_IR_N_STEPS,
    CUDSS_CONFIG_IR_TOL,
    CUDSS_CONFIG_PIVOT_TYPE,
    CUDSS_CONFIG_PIVOT_THRESHOLD,
    CUDSS_CONFIG_PIVOT_EPSILON,
    CUDSS_CONFIG_MAX_LU_NNZ         // (limited use)
} cudssConfigParam_t;

typedef enum cudssDataParam_t {
    CUDSS_DATA_INFO,	             // (out)
    CUDSS_DATA_LU_NNZ,	             // (out)
    CUDSS_DATA_NPIVOTS,              // (out)
    CUDSS_DATA_INERTIA,              // (out, non-trivial for non-positive-definite matrices)  
    CUDSS_DATA_PERM_REORDER,         // (out)
    CUDSS_DATA_PERM_ROW,             // (out, not supported)
    CUDSS_DATA_PERM_COL,		     // (out, not supported)	
    CUDSS_DATA_DIAG,			     // (out)
    CUDSS_DATA_USER_PERM             // (in) for the user to provide a permutation 
} cudssDataParam_t;

typedef enum cudssPhase_t { // allows combinations like FACTORIZATION | SOLVE with extra optimizations
    CUDSS_PHASE_ANALYSIS               = 1,
    CUDSS_PHASE_FACTORIZATION          = 2,
    CUDSS_PHASE_REFACTORIZATION        = 4,
    CUDSS_PHASE_SOLVE                  = 8,
    CUDSS_PHASE_SOLVE_FWD              = 16,
    CUDSS_PHASE_SOLVE_DIAG             = 32,
    CUDSS_PHASE_SOLVE_BWD              = 64
} cudssPhase_t;

typedef enum cudssStatus_t {
    CUDSS_STATUS_SUCCESS,
    CUDSS_STATUS_NOT_INITIALIZED,
    CUDSS_STATUS_ALLOC_FAILED,
    CUDSS_STATUS_INVALID_VALUE,
    CUDSS_STATUS_NOT_SUPPORTED,
    CUDSS_STATUS_ARCH_MISMATCH,
    CUDSS_STATUS_EXECUTION_FAILED,
    CUDSS_STATUS_INTERNAL_ERROR,
    CUDSS_STATUS_ZERO_PIVOT,
} cudssStatus_t;

typedef enum cudssMatrixType_t {
    CUDSS_MTYPE_GENERAL,
    CUDSS_MTYPE_SYMMETRIC,
    CUDSS_MTYPE_HERMITIAN,
    CUDSS_MTYPE_SPD,
    CUDSS_MTYPE_HPD
} cudssMatrixType_t;

typedef enum cudssMatrixViewType_t {
    CUDSS_MVIEW_FULL,
    CUDSS_MVIEW_LOWER,
    CUDSS_MVIEW_UPPER
} cudssMatrixViewType_t;

typedef enum cudssIndexBase_t {
    CUDSS_BASE_ZERO,
    CUDSS_BASE_ONE
} cudssIndexBase_t;

typedef enum cudssLayout_t {
    CUDSS_LAYOUT_COL_MAJOR,
    CUDSS_LAYOUT_ROW_MAJOR
} cudssLayout_t;

typedef enum cudssAlgType_t {
    CUDSS_ALG_DEFAULT,
    CUDSS_ALG_1,
    CUDSS_ALG_2,
    CUDSS_ALG_3
} cudssAlgType_t;

typedef enum cudssPivotType_t {
    CUDSS_PIVOT_COL,
    CUDSS_PIVOT_ROW,
    CUDSS_PIVOT_NONE
} cudssPivotType_t;

typedef enum cudssMatrixFormat_t {
  CUDSS_MFORMAT_DENSE,
  CUDSS_MFORMAT_CSR,
} cudssMatrixFormat_t;


cudssStatus_t CUDSSAPI cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void *value, size_t sizeInBytes);

cudssStatus_t CUDSSAPI cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param,  void *value, size_t sizeInBytes, size_t *sizeWritten);

cudssStatus_t CUDSSAPI cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param,  void *value, size_t sizeInBytes);

cudssStatus_t CUDSSAPI cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void *value, size_t sizeInBytes, size_t *sizeWritten);

// Main cuDSS routine

cudssStatus_t CUDSSAPI cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs);

// Setting the stream (in the library handle)

cudssStatus_t CUDSSAPI cudssSetStream(cudssHandle_t handle, cudaStream_t stream);

// Create/Destroy APIs (allocating structures + set defaults)

cudssStatus_t CUDSSAPI cudssConfigCreate(cudssConfig_t *solverConfig);
cudssStatus_t CUDSSAPI cudssConfigDestroy(cudssConfig_t solverConfig);

cudssStatus_t CUDSSAPI cudssDataCreate(cudssHandle_t handle, cudssData_t *solverData);
cudssStatus_t CUDSSAPI cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData);

cudssStatus_t CUDSSAPI cudssCreate(cudssHandle_t *handle);
cudssStatus_t CUDSSAPI cudssDestroy(cudssHandle_t handle);

// Versioning

cudssStatus_t CUDSSAPI cudssGetProperty(libraryPropertyType propertyType, int* value);

// Create/Destroy API helpers for matrix wrappers

cudssStatus_t CUDSSAPI cudssMatrixCreateDn(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t ld, void *values, cudaDataType_t valueType,  cudssLayout_t layout);

cudssStatus_t CUDSSAPI cudssMatrixCreateCsr(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t nnz, void *rowStart,void *rowEnd, void *colIndices, void *values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase);

cudssStatus_t CUDSSAPI cudssMatrixDestroy(cudssMatrix_t matrix);

// Setters/Getters API helpers for matrix wrappers

cudssStatus_t CUDSSAPI cudssMatrixGetDn(cudssMatrix_t matrix,  int64_t* nrows, int64_t* ncols, int64_t* ld, void **values, cudaDataType_t* type, cudssLayout_t* layout);

cudssStatus_t CUDSSAPI cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void **rowStart, void **rowEnd, void **colIndices, void **values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase);

cudssStatus_t CUDSSAPI cudssMatrixSetValues(cudssMatrix_t matrix, void *values);

cudssStatus_t CUDSSAPI cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void *rowOffsets, void *rowEnd, void *colIndices, void *values);

cudssStatus_t CUDSSAPI cudssMatrixGetFormat(cudssMatrix_t matrix, cudssMatrixFormat_t* format );

} /* extern "C" */

#endif /* CUDSS_PUBLIC_HEADER_H */
