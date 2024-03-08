#ifndef IP_CUDA_SOLVER_INTERFACE_H
#define IP_CUDA_SOLVER_INTERFACE_H

#ifdef IPOPT_SINGLE
#define CUDANUMBER float
#else
#define CUDANUMBER double
#endif

#include "IpoptConfig.h"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpLibraryLoader.hpp"
#include "IpTypes.h"
#include "cuda_runtime_api.h"
#include "cuDSS.h"

class CUDASolverInterface : public Ipopt::SparseSymLinearSolverInterface
{
public:
    CUDASolverInterface();
    ~CUDASolverInterface();
    static void RegisterOptions(Ipopt::SmartPtr<Ipopt::RegisteredOptions> roptions);
    bool InitializeImpl(const Ipopt::OptionsList& options, const std::string& prefix);
    Ipopt::ESymSolverStatus InitializeStructure(Ipopt::Index dim, Ipopt::Index nonzeros, const Ipopt::Index* ia, const Ipopt::Index* ja);
    Ipopt::Number* GetValuesArrayPtr();
    Ipopt::ESymSolverStatus MultiSolve(bool new_matrix, const Ipopt::Index* ia, const Ipopt::Index* ja, Ipopt::Index nrhs, Ipopt::Number* rhs_vals, bool check_NegEVals, Ipopt::Index numberOfNegEVals);
    Ipopt::Index NumberOfNegEVals() const;
    bool IncreaseQuality();
    bool ProvidesInertia() const;
    Ipopt::SparseSymLinearSolverInterface::EMatrixFormat MatrixFormat() const;
    bool ProvidesDegeneracyDetection() const;
    Ipopt::ESymSolverStatus DetermineDependentRows(const Ipopt::Index* ia, const Ipopt::Index* ja, std::list<Ipopt::Index>& c_deps);
private:
    int numRows; // Square matrix, same as number of columns
    int numNonZeros;
    cudaDataType_t dataType;
    int previousNumRHS;

    cudssHandle_t handle;
    cudaStream_t stream;
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDANUMBER* AValuesHost;
    CUDANUMBER* AValues;
    int* ARowPointers;
    int* AColIndices;
    cudssMatrix_t AMatrix;

    CUDANUMBER* bValues;
    cudssMatrix_t bMatrix;

    CUDANUMBER* xValues;
    cudssMatrix_t xMatrix;
};

#endif