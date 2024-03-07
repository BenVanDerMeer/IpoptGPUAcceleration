#include "OptimisationProblem.h"

OptimisationProblem::OptimisationProblem(int numFreedoms_, int numConstraints_)
{
	numFreedoms = numFreedoms_;
	numConstraints = numConstraints_;

	std::vector<double> includedPoint; // All constraints will be forced to have this as a feasible point, guaranteeing a non-zero feasible region
	for (int i = 0; i < numFreedoms; i++)
	{
		includedPoint.push_back(rand() / (double)RAND_MAX);
	}

	for (int i = 0; i < numConstraints; i++)
	{
		std::vector<double> constraintCoefficients;
		for (int j = 0; j < numFreedoms; j++)
		{
			constraintCoefficients.push_back(rand() / (double)RAND_MAX * 20.0 - 10.0);
		}
		constraintsCoefficients.push_back(constraintCoefficients);
	}

	for (int i = 0; i < numConstraints; i++)
	{
		double constant = 0.0;
		for (int j = 0; j < numFreedoms; j++)
		{
			constant += constraintsCoefficients[i][j] * includedPoint[j];
		}
		constant -= rand() / (double)RAND_MAX;
		constraintsConstant.push_back(constant);
	}
}

bool OptimisationProblem::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
{
	n = numFreedoms;
	m = numConstraints;
	nnz_jac_g = numFreedoms * numConstraints;
	nnz_h_lag = 0;
	index_style = TNLP::C_STYLE;
	return true;
}

bool OptimisationProblem::get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u, Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
{
	for (int i = 0; i < n; i++)
	{
		x_l[i] = 0;
		x_u[i] = 1.0;
	}
	
	for (int i = 0; i < m; i++)
	{
		g_l[i] = -2e19;
		g_u[i] = 0.0;
	}
	return true;
}

bool OptimisationProblem::get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda)
{
	for (int i = 0; i < n; i++)
	{
		x[i] = rand() / (double)RAND_MAX;
	}
	return true;
}

bool OptimisationProblem::eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value)
{
	obj_value = 0.0;
	for (int i = 0; i < n; i++)
	{
		obj_value += x[i];
	}
	return true;
}

bool OptimisationProblem::eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f)
{
	for (int i = 0; i < n; i++)
	{
		grad_f[i] = 1.0;
	}
	return true;
}

bool OptimisationProblem::eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g)
{
	for (int i = 0; i < m; i++)
	{
		g[i] = 0.0;
		for (int j = 0; j < n; j++)
		{
			g[i] += constraintsCoefficients[i][j] * x[j];
		}
		g[i] -= constraintsConstant[i];
	}
	return true;
}

bool OptimisationProblem::eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values)
{
	if (values == NULL)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				iRow[i * n + j] = i;
				jCol[i * n + j] = j;
			}
		}
	}
	else
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				values[i * n + j] = constraintsCoefficients[i][j];
			}
		}
	}
	return true;
}

bool OptimisationProblem::eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda, bool new_lambda,
	Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values)
{
	return false;
}

void OptimisationProblem::finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U, Ipopt::Index m,
	const Ipopt::Number* g, const Ipopt::Number* lambda, Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq)
{
	std::cout << "Objective value: " << obj_value << std::endl;
	for (int i = 0; i < n; i++)
	{
		std::cout << "x[" << i << "] = " << x[i] << std::endl;
	}
}

bool OptimisationProblem::intermediate_callback(Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value, Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
	Ipopt::Number d_norm, Ipopt::Number regularization_size, Ipopt::Number alpha_du, Ipopt::Number alpha_pr, Ipopt::Index ls_trials, const Ipopt::IpoptData* ip_data,
	Ipopt::IpoptCalculatedQuantities* ip_cq)
{
	return true;
}