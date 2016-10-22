/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
  title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
  author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
  year={2016},
  institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cmath>
#include "Tools.h"
#include "ImplicitQRSVD.h"

template <class T>
void testAccuracy(const std::vector<Eigen::Matrix<T, 3, 3> >& AA,
    const std::vector<Eigen::Matrix<T, 3, 3> >& UU,
    const std::vector<Eigen::Matrix<T, 3, 1> >& SS,
    const std::vector<Eigen::Matrix<T, 3, 3> >& VV)
{
    T max_UUt_error = 0, max_VVt_error = 0, max_detU_error = 0, max_detV_error = 0, max_reconstruction_error = 0;
    T ave_UUt_error = 0, ave_VVt_error = 0, ave_detU_error = 0, ave_detV_error = 0, ave_reconstruction_error = 0;
    for (size_t i = 0; i < AA.size(); i++) {
        Eigen::Matrix<T, 3, 3> M = AA[i];
        Eigen::Matrix<T, 3, 1> S = SS[i];
        Eigen::Matrix<T, 3, 3> U = UU[i];
        Eigen::Matrix<T, 3, 3> V = VV[i];
        T error;
        error = (U * U.transpose() - Eigen::Matrix<T, 3, 3>::Identity()).array().abs().maxCoeff();
        max_UUt_error = (error > max_UUt_error) ? error : max_UUt_error;
        ave_UUt_error += fabs(error);
        error = (V * V.transpose() - Eigen::Matrix<T, 3, 3>::Identity()).array().abs().maxCoeff();
        max_VVt_error = (error > max_VVt_error) ? error : max_VVt_error;
        ave_VVt_error += fabs(error);
        error = fabs(fabs(U.determinant()) - (T)1);
        max_detU_error = (error > max_detU_error) ? error : max_detU_error;
        ave_detU_error += fabs(error);
        error = fabs(fabs(V.determinant()) - (T)1);
        max_detV_error = (error > max_detV_error) ? error : max_detV_error;
        ave_detV_error += fabs(error);
        error = (U * S.asDiagonal() * V.transpose() - M).array().abs().maxCoeff();
        max_reconstruction_error = (error > max_reconstruction_error) ? error : max_reconstruction_error;
        ave_reconstruction_error += fabs(error);
    }
    ave_UUt_error /= (T)(AA.size());
    ave_VVt_error /= (T)(AA.size());
    ave_detU_error /= (T)(AA.size());
    ave_detV_error /= (T)(AA.size());
    ave_reconstruction_error /= (T)(AA.size());
    std::cout << std::setprecision(10) << " UUt max error: " << max_UUt_error
              << " VVt max error: " << max_VVt_error
              << " detU max error:" << max_detU_error
              << " detV max error:" << max_detV_error
              << " recons max error:" << max_reconstruction_error << std::endl;
    std::cout << std::setprecision(10) << " UUt ave error: " << ave_UUt_error
              << " VVt ave error: " << ave_VVt_error
              << " detU ave error:" << ave_detU_error
              << " detV ave error:" << ave_detV_error
              << " recons ave error:" << ave_reconstruction_error << std::endl;
}

template <class T>
void runImplicitQRSVD(const int repeat, const std::vector<Eigen::Matrix<T, 3, 3> >& tests, const bool accuracy_test)
{
    using namespace JIXIE;
    std::vector<Eigen::Matrix<T, 3, 3> > UU, VV;
    std::vector<Eigen::Matrix<T, 3, 1> > SS;
    JIXIE::Timer timer;
    timer.start();
    double total_time = 0;
    for (int test_iter = 0; test_iter < repeat; test_iter++) {
        timer.click();
        for (size_t i = 0; i < tests.size(); i++) {
            Eigen::Matrix<T, 3, 3> M = tests[i];
            Eigen::Matrix<T, 3, 1> S;
            Eigen::Matrix<T, 3, 3> U;
            Eigen::Matrix<T, 3, 3> V;
            singularValueDecomposition(M, U, S, V);
            if (accuracy_test && test_iter == 0) {
                UU.push_back(U);
                SS.push_back(S);
                VV.push_back(V);
            }
        }
        double this_time = timer.click();
        total_time += this_time;
        std::cout << std::setprecision(10) << "impQR time: " << this_time << std::endl;
    }
    std::cout << std::setprecision(10) << "impQR Average time: " << total_time / (double)(repeat) << std::endl;
    if (accuracy_test)
        testAccuracy(tests, UU, SS, VV);
}

template <class T>
void addRandomCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const T random_range, const int N)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::cout << std::setprecision(10) << "Adding random test cases with range " << -random_range << " to " << random_range << std::endl;
    RandomNumber<T> random_gen(123);
    for (int t = 0; t < N; t++) {
        Eigen::Matrix<T, 3, 3> Z;
        random_gen.fill(Z, -random_range, random_range);
        tests.push_back(Z);
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addIntegerCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int int_range)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::cout << std::setprecision(10) << "Adding integer test cases with range " << -int_range << " to " << int_range << std::endl;
    Eigen::Matrix<T, 3, 3> Z;
    Z.fill(-int_range);
    typename Eigen::Matrix<T, 3, 3>::Index i = 0;
    tests.push_back(Z);
    while (i < Eigen::Matrix<T, 3, 3>::SizeAtCompileTime) {
        if (Z(i) < int_range) {
            Z(i)++;
            tests.push_back(Z);
            i = 0;
        }
        else {
            Z(i) = -int_range;
            i++;
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addPerturbationFromIdentityCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int num_perturbations, const T perturb)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::vector<Eigen::Matrix<T, 3, 3> > tests_tmp;
    Eigen::Matrix<T, 3, 3> Z = Eigen::Matrix<T, 3, 3>::Identity();
    tests_tmp.push_back(Z);
    std::cout << std::setprecision(10) << "Adding perturbed identity test cases with perturbation " << perturb << std::endl;
    RandomNumber<T> random_gen(123);
    size_t special_cases = tests_tmp.size();
    for (size_t t = 0; t < special_cases; t++) {
        for (int i = 0; i < num_perturbations; i++) {
            random_gen.fill(Z, -perturb, perturb);
            tests.push_back(tests_tmp[t] + Z);
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addPerturbationCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int int_range, const int num_perturbations, const T perturb)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::vector<Eigen::Matrix<T, 3, 3> > tests_tmp;
    Eigen::Matrix<T, 3, 3> Z;
    Z.fill(-int_range);
    typename Eigen::Matrix<T, 3, 3>::Index i = 0;
    tests_tmp.push_back(Z);
    while (i < Eigen::Matrix<T, 3, 3>::SizeAtCompileTime) {
        if (Z(i) < int_range) {
            Z(i)++;
            tests_tmp.push_back(Z);
            i = 0;
        }
        else {
            Z(i) = -int_range;
            i++;
        }
    }
    std::cout << std::setprecision(10) << "Adding perturbed integer test cases with perturbation " << perturb << " and range " << -int_range << " to " << int_range << std::endl;
    RandomNumber<T> random_gen(123);
    size_t special_cases = tests_tmp.size();
    for (size_t t = 0; t < special_cases; t++) {
        for (int i = 0; i < num_perturbations; i++) {
            random_gen.fill(Z, -perturb, perturb);
            tests.push_back(tests_tmp[t] + Z);
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

void runBenchmark()
{
    using namespace JIXIE;
    using std::fabs;

    bool run_qr;

    bool test_float;
    bool test_double;
    bool accuracy_test;
    bool normalize_matrix;
    int number_of_repeated_experiments;
    bool test_random;
    int random_range;
    int number_of_random_cases;
    bool test_integer;
    int integer_range;
    bool test_perturbation;
    int perturbation_count;
    float float_perturbation;
    double double_perturbation;
    bool test_perturbation_from_identity;
    int perturbation_from_identity_count;
    float float_perturbation_identity;
    double double_perturbation_identity;
    std::string title;

    // Finalized options
    run_qr = true;

    test_float = true;
    test_double = true;
    normalize_matrix = false;
    int number_of_repeated_experiments_for_timing = 2;

    for (int test_number = 1; test_number <= 1; test_number++) {

        if (test_number == 1) {
            title = "random timing test";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = true, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, integer_range = 3, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
		
		 //if (test_number == 2) {
		//	 title = "SVD 2*2";
			 
			
			
		 //}
        /*if (test_number == 2) {
            title = "integer timing test";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = true; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 3) {
            title = "integer-perturbation timing test: 256 eps";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = true, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 4) {
            title = "identity-perturbation timing test: 1e-3";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 5) {
            title = "identity-perturbation timing test: 256 eps";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation_identity = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed itentity test
        }

        if (test_number == 6) {
            title = "random accuracy test";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = true, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, integer_range = 3, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 7) {
            title = "integer accuracy test";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = true; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 8) {
            title = "integer-perturbation accuracy test: 256 eps";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = true, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 9) {
            title = "identity-perturbation accuracy test: 1e-3";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 10) {
            title = "identity-perturbation accuracy test: 256 eps";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation_identity = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed itentity test
        }*/

        std::cout << " \n========== RUNNING BENCHMARK TEST == " << title << "=======" << std::endl;
        std::cout << " run_qr " << run_qr << std::endl;
        std::cout << " test_float " << test_float << std::endl;
        std::cout << " test_double " << test_double << std::endl;
        std::cout << " accuracy_test " << accuracy_test << std::endl;
        std::cout << " normalize_matrix " << normalize_matrix << std::endl;
        std::cout << " number_of_repeated_experiments " << number_of_repeated_experiments << std::endl;
        std::cout << " test_random " << test_random << std::endl;
        std::cout << " random_range " << random_range << std::endl;
        std::cout << " number_of_random_cases " << number_of_random_cases << std::endl;
        std::cout << " test_integer " << test_integer << std::endl;
        std::cout << " integer_range " << integer_range << std::endl;
        std::cout << " test_perturbation " << test_perturbation << std::endl;
        std::cout << " perturbation_count " << perturbation_count << std::endl;
        std::cout << " float_perturbation " << float_perturbation << std::endl;
        std::cout << " double_perturbation " << double_perturbation << std::endl;
        std::cout << " test_perturbation_from_identity " << test_perturbation_from_identity << std::endl;
        std::cout << " perturbation_from_identity_count " << perturbation_from_identity_count << std::endl;
        std::cout << " float_perturbation_identity " << float_perturbation_identity << std::endl;
        std::cout << " double_perturbation_identity " << double_perturbation_identity << std::endl;

        std::cout << std::setprecision(10) << "\n--- float test ---\n" << std::endl;
        if (test_float) {
            std::vector<Eigen::Matrix<float, 3, 3> > tests;
            if (test_integer)
                addIntegerCases(tests, integer_range);
            if (test_perturbation)
                addPerturbationCases(tests, integer_range, perturbation_count, float_perturbation);
            if (test_perturbation_from_identity)
                addPerturbationFromIdentityCases(tests, perturbation_from_identity_count, float_perturbation_identity);
            if (test_random)
                addRandomCases(tests, (float)random_range, number_of_random_cases);
            if (normalize_matrix) {
                for (size_t i = 0; i < tests.size(); i++) {
                    float norm = tests[i].norm();
                    if (norm > (float)8 * std::numeric_limits<float>::epsilon()) {
                        tests[i] /= norm;
                    }
                }
            }
            std::cout << std::setprecision(10) << "\n-----------" << std::endl;
            if (run_qr)
                runImplicitQRSVD(number_of_repeated_experiments, tests, accuracy_test);

        }

        std::cout << std::setprecision(10) << "\n--- double test ---\n" << std::endl;
        if (test_double) {
            std::vector<Eigen::Matrix<double, 3, 3> > tests;
            if (test_integer)
                addIntegerCases(tests, integer_range);
            if (test_perturbation)
                addPerturbationCases(tests, integer_range, perturbation_count, double_perturbation);
            if (test_perturbation_from_identity)
                addPerturbationFromIdentityCases(tests, perturbation_from_identity_count, double_perturbation_identity);
            if (test_random)
                addRandomCases(tests, (double)random_range, number_of_random_cases);
            if (normalize_matrix) {
                for (size_t i = 0; i < tests.size(); i++) {
                    double norm = tests[i].norm();
                    if (norm > (double)8 * std::numeric_limits<double>::epsilon()) {
                        tests[i] /= norm;
                    }
                }
            }
            std::cout << std::setprecision(10) << "\n-----------" << std::endl;
            if (run_qr)
                runImplicitQRSVD(number_of_repeated_experiments, tests, accuracy_test);

        }
    }
}

template <class T>
void Givens(T x, T y, T* c, T* s){
	T d=x*x+y*y,t;
	*c=1;
	*s=0;
	if(d!=0){
		t=sqrt(d);
		*c=x/t;
		*s=-y/t;
	}
}

template <class T>
void Jacobi(Eigen::Matrix<T,2,2> C, T* c, T* s){
	T t,tau=(C(3)-C(0))/(2*C(1));
	if(std::abs(tau)<10e5){
		if(tau>0)
			t=tau-sqrt(tau*tau+1);
		if(tau<0)
			t=tau+sqrt(1+tau*tau);
	}
	else if(tau>0)
		t=1/(tau+sqrt(1+tau*tau));
	else
		t=1/(-tau+sqrt(1+tau*tau));

	*c=1/sqrt(1+t*t);
	*s=*c*t;
}


template <class T>
void SVD2_2(Eigen::Matrix<T,2,2> FF){
	Eigen::Matrix<T,2,2> C,V,Flip,Sign,U,Sigma,F;
	Sign<<1,0,0,-1;
	Flip<<0,1,1,0;
	F=FF;
	
	C=F.transpose()*F;
	
	T c,s;
							//do Jacobi rotation to get C=V*Sigma^2*V^t
	Jacobi(C,&c,&s);

	V << c,-s,s,c;
	
	C.applyOnTheLeft(V.adjoint());
	C.applyOnTheRight(V);
	
	bool V_flip = false;
	bool U_flip = false;
	
	T sigma1, sigma2;
	if(C(0)>=C(3)){
		sigma1=sqrt(C(0));
		sigma2=sqrt(C(3));
	}
	else{
		V_flip = true;
		
		V.applyOnTheRight(Flip);
		
		sigma1=sqrt(C(3));
		sigma2=sqrt(C(0));
	}
	
	C=F*V;								//A=FV

	Givens(C(0),C(1),&c,&s);
	U<< c,s,-s,c;
	
	T r = C(2)*s+C(3)*c;   				//r_22
	
	if(r<0){
		U_flip=true;
		U.applyOnTheRight(Sign);
		sigma2=-sigma2;
	}
	
	Sigma << sigma1, 0, 0, sigma2;
	F=U*Sigma*V.transpose();
	
	T detF=F(0)*F(3)-F(1)*F(2);
	
	if(detF<0){
		sigma2=-sigma2;
		if(U_flip)
			U.applyOnTheRight(Sign);
		else{
			V.applyOnTheRight(Sign);
		}
	}
	else if(detF>0){
		if(U_flip && V_flip){
			U.applyOnTheRight(Sign);
			V.applyOnTheRight(Sign);
		}
	}
	else if(detF==0){
		if(U_flip&& V_flip){
			U.applyOnTheRight(Sign);
			V.applyOnTheRight(Sign);
		}
		else if(U_flip){
			U.applyOnTheRight(Sign);
			sigma1=-sigma1;
		}
		else if(V_flip){
			V.applyOnTheRight(Sign);
			sigma1=-sigma1;
		}
	}
	
	std::cout << "computed sigma:\n" << sigma1 << "," << sigma2 << std::endl;
	std::cout << "F:\n" << F << std::endl; 
	std::cout << "U:\n" << U << std::endl; 
	std::cout << "V:\n" << V << std::endl; 
	std::cout << "check determinant:" << std::endl;
	std::cout << "detU=" << U(0)*U(3)-U(1)*U(2) << std::endl;
	std::cout << "detV=" << V(0)*V(3)-V(1)*V(2) << std::endl;
	
	Sigma << sigma1, 0, 0, sigma2;
	F=FF-U*Sigma*V.transpose();

	T error=sqrt(F(0)*F(0)+F(1)*F(1)+F(2)*F(2)+F(3)*F(3));    //error in Frobinus norm
		
	std::cout << "SVD error:\n" << error << std::endl;
}

template <class T>
T findMax(T a, T b, T c){					//find max of three inputs
	T max = a;
	if(max<b)
		max=b;
	if(max<c)
		max=c;
	
	return max;
}

template <class T>
T FNorm(Eigen::Matrix<T,3,3> F){			//Frobinus norm of 3*3 matrix
	int n=F.cols();
	T norm=0;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			norm+=F(i,j)*F(i,j);
	norm=sqrt(norm);
	
	return norm;
}

template <class T>
void Polar3_3(Eigen::Matrix<T,3,3> F){
	Eigen::Matrix<T,3,3> R,S,G;
	G.setZero();						
	S=F;
	R<<1,0,0,0,1,0,0,0,1;
	
	int it = 0, max_it = 100;
	T c,s,m,n;
	float tol = 0.001;
	
	while(it<max_it && findMax(std::abs(S(0,1)-S(1,0)),std::abs(S(0,2)-S(2,0)),std::abs(S(1,2)-S(2,1)))>tol){
		for(int i=0;i<2;i++)
			for(int j=i+1;j<3;j++){
				m=S(i,i)+S(j,j),n=S(j,i)-S(i,j);
				Givens(m,n,&c,&s);

				G.setZero();						//get G_ij
				G(i,i)=c;
				G(j,j)=c;
				G(i,j)=s;
				G(j,i)=-s;
				G(3-i-j,3-i-j)=1;
				R.applyOnTheRight(G);
				
				S.applyOnTheLeft(G.transpose());	//update S
			}
		it++;
	}
	std::cout << "total iterations: " << it << std::endl;
	std::cout << "F: \n" << F << std::endl;
	std::cout << "R: \n" << R << std::endl;
	std::cout << "S: \n" << S << std::endl;
	
	F-=R*S;
	T error=FNorm(F);
	std::cout << "Polar error: \n" << error << std::endl;
}


int main()
{
	//std::clock_t start;
	//double duration;
	
	Eigen::Matrix<int,2,2> F1;
	Eigen::Matrix<float,2,2> F2;
	
	Eigen::Matrix<int,3,3> G1;
	Eigen::Matrix<float,3,3> G2;
	
	F1<<1,2,3,5;				//test template <class T>
	SVD2_2(F1);
  	
	//start=std::clock();		//test running time
	F2<<1,2,3,5;
	SVD2_2(F2);
  	//duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
  	//std::cout<<"printf:" <<duration<<'\n';
	
	G1<<1,2,3,4,5,6,7,8,9;
	Polar3_3(G1);
	
	//start=std::clock();
	G2<<1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1;
	Polar3_3(G2);
  	//duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
  	//std::cout<<"printf:" <<duration<<'\n';
	
  	//bool run_benchmark = true;
    //if (run_benchmark) runBenchmark();
}
