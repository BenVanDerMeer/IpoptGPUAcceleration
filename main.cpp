#include "IpIpoptApplication.hpp"
#include "OptimisationProblem.h"

int main()
{
	Ipopt::SmartPtr<Ipopt::TNLP> objective = new OptimisationProblem(2, 2);
	Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
	app->Options()->SetIntegerValue("max_iter", 500);
	app->Options()->SetStringValue("hessian_approximation", "limited-memory");
	Ipopt::ApplicationReturnStatus status = app->Initialize();
	status = app->OptimizeTNLP(objective);

	return 0;
}