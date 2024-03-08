#include "CUDASolverInterface.h"

CUDASolverInterface::CUDASolverInterface()
{
	numRows = 0;
	numNonZeros = 0;
	dataType = sizeof(CUDANUMBER) == 8 ? CUDA_R_64F : CUDA_R_32F;
	previousNumRHS = 1;

	cudssCreate(&handle);
	cudaStreamCreate(&stream);
	cudssSetStream(handle, stream);
	cudssConfigCreate(&solverConfig);
	cudssDataCreate(handle, &solverData);

	AValuesHost = nullptr;
	AValues = nullptr;
	ARowPointers = nullptr;
	AColIndices = nullptr;
	AMatrix = NULL;

	bValues = nullptr;
	bMatrix = NULL;

	xValues = nullptr;
	xMatrix = NULL;
}

CUDASolverInterface::~CUDASolverInterface()
{
	free(AValuesHost);
	cudaFree(AValues);
	cudaFree(ARowPointers);
	cudaFree(AColIndices);
	cudaFree(bValues);
	cudaFree(xValues);
	
	cudssMatrixDestroy(AMatrix);
	cudssMatrixDestroy(bMatrix);
	cudssMatrixDestroy(xMatrix);

	cudssDataDestroy(handle, solverData);
	cudssConfigDestroy(solverConfig);
	cudaStreamDestroy(stream);
	cudssDestroy(handle);
}

void CUDASolverInterface::RegisterOptions(Ipopt::SmartPtr<Ipopt::RegisteredOptions> roptions)
{
	
}

bool CUDASolverInterface::InitializeImpl(const Ipopt::OptionsList& options, const std::string& prefix)
{
	return true;
}

Ipopt::ESymSolverStatus CUDASolverInterface::InitializeStructure(Ipopt::Index dim, Ipopt::Index nonzeros, const Ipopt::Index* ia, const Ipopt::Index* ja)
{
	numRows = dim;
	numNonZeros = nonzeros;

	AValuesHost = (double*)malloc(numNonZeros * sizeof(CUDANUMBER));
	std::fill_n(AValuesHost, numNonZeros, 1.0); // Fill with 1's for symbolic analysis

	cudaMalloc((void**)&AValues, numNonZeros * sizeof(CUDANUMBER));
	cudaMalloc((void**)&ARowPointers, (numRows + 1) * sizeof(int));
	cudaMalloc((void**)&AColIndices, numNonZeros * sizeof(int));
	cudaMalloc((void**)&xValues, numRows * sizeof(CUDANUMBER));
	cudaMalloc((void**)&bValues, numRows * sizeof(CUDANUMBER));

	cudaMemcpy(AValues, AValuesHost, numNonZeros * sizeof(CUDANUMBER), cudaMemcpyHostToDevice);
	cudaMemcpy(ARowPointers, ia, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(AColIndices, ja, numNonZeros * sizeof(int), cudaMemcpyHostToDevice);

	cudssMatrixCreateCsr(&AMatrix, numRows, numRows, numNonZeros, ARowPointers, NULL, AColIndices, AValues, CUDA_R_32I, dataType, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);

	cudssMatrixCreateDn(&xMatrix, numRows, 1, numRows, xValues, dataType, CUDSS_LAYOUT_COL_MAJOR);
	cudssMatrixCreateDn(&bMatrix, numRows, 1, numRows, bValues, dataType, CUDSS_LAYOUT_COL_MAJOR);

	cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, AMatrix, xMatrix, bMatrix); // Symbolic analysis
	cudaStreamSynchronize(stream);

	return Ipopt::SYMSOLVER_SUCCESS;
}

Ipopt::Number* CUDASolverInterface::GetValuesArrayPtr()
{
	return AValuesHost;
}

Ipopt::ESymSolverStatus CUDASolverInterface::MultiSolve(bool new_matrix, const Ipopt::Index* ia, const Ipopt::Index* ja, Ipopt::Index nrhs, Ipopt::Number* rhs_vals, bool check_NegEVals, Ipopt::Index numberOfNegEVals)
{
	if (nrhs != previousNumRHS)
	{
		cudaFree(bValues);
		cudaMalloc((void**)&bValues, numRows * nrhs * sizeof(CUDANUMBER));
		cudssMatrixCreateDn(&bMatrix, numRows, nrhs, numRows, bValues, dataType, CUDSS_LAYOUT_COL_MAJOR);

		cudaFree(xValues);
		cudaMalloc((void**)&xValues, numRows * nrhs * sizeof(CUDANUMBER));
		cudssMatrixCreateDn(&xMatrix, numRows, nrhs, numRows, xValues, dataType, CUDSS_LAYOUT_COL_MAJOR);
	}
	cudaMemcpy(bValues, rhs_vals, numRows * nrhs * sizeof(CUDANUMBER), cudaMemcpyHostToDevice);
	previousNumRHS = nrhs;

	if (new_matrix)
	{
		cudaMemcpy(AValues, AValuesHost, numNonZeros * sizeof(CUDANUMBER), cudaMemcpyHostToDevice);

		cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, AMatrix, xMatrix, bMatrix);
	}
	
	cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, AMatrix, xMatrix, bMatrix);
	cudaStreamSynchronize(stream);
	
	cudaMemcpy(rhs_vals, xValues, numRows * nrhs * sizeof(CUDANUMBER), cudaMemcpyDeviceToHost);

	return Ipopt::SYMSOLVER_SUCCESS;
}

Ipopt::Index CUDASolverInterface::NumberOfNegEVals() const
{
	throw std::runtime_error("CUDA solver does not implement count for number of negative eigenvalues");
}

bool CUDASolverInterface::IncreaseQuality()
{
	return false;
}

bool CUDASolverInterface::ProvidesInertia() const
{
	return false;
}

Ipopt::SparseSymLinearSolverInterface::EMatrixFormat CUDASolverInterface::MatrixFormat() const
{
	return Ipopt::SparseSymLinearSolverInterface::CSR_Format_0_Offset;
}

bool CUDASolverInterface::ProvidesDegeneracyDetection() const
{
	return false;
}

Ipopt::ESymSolverStatus CUDASolverInterface::DetermineDependentRows(const Ipopt::Index* ia, const Ipopt::Index* ja, std::list<Ipopt::Index>& c_deps)
{
	throw std::runtime_error("CUDA solver does not implement check for linearly dependent rows");
}