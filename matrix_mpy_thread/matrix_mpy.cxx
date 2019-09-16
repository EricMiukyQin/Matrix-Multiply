// Matrix multiplication using C++ threads

//	c++ -std=c++11 -pthread -O2 matrix_mpy.cxx matrix.cxx matrix_mpy_user.cxx ee155_utils.cxx 

#include <iostream>
#include <sstream>
#include <mutex>

using namespace std;
#include "bits.hxx"
#include "ee155_utils.hxx"
#include "matrix.hxx"

// Simple algorithm for multiplying two matrices.
void Matrix::mpy_dumb (const Matrix &A, const Matrix &B) {
    int N = this->N();
    for (int r=0; r<N; ++r)
	for (int c=0; c<N; ++c) {
	    float sum=0.0;
	    for (int k=0; k<N; ++k)
		sum += (A(r,k) * B(k,c));
	    this->data[index(r,c)] = sum;
	}
}

// Wrapper function around Matrix::mpy2(). It just runs ::mpy2() several times
// and checks how long it took.
static void run_mpy2 (int BS, int n_cores, const Matrix &a, const Matrix &b,
					   const Matrix &c, Matrix &d) {
    long int total_time=0;
    for (int i=0; i<4; ++i) {
	auto start = start_time();
	d.mpy2 (a, b, BS, n_cores);
	long int time = delta_usec (start);
	total_time += time;
	c.compare (d);
	cout<<"mpy2 with "<<n_cores<<" cores="<<(time/1000000.0)<<"sec"<<endl;
    }
    LOG ("mpy2 took an average of "<<(total_time/4000000.0)<<"sec\n");
}

main () {
    // Time mpy_dumb() for 1Kx1K.
    LOG ("Timing mpy_dumb() on 1Kx1K matrices");
    int N=1<<10;
    Matrix a(N), b(N), c(N), d(N);
    a.init_cyclic_order();
    b.init_identity();

    auto start = start_time();
    c.mpy_dumb (b, a);
    long int time = delta_usec (start);
    LOG ("1Kx1K mpy_dumb() took "<<(time/1000000.0)<<"sec");

    // Time mpy_dumb(), mpy1() and mpy2() for 2Kx2K.
    N=1<<11;
    a = Matrix(N); b=Matrix(N); c=Matrix(N); d=Matrix(N);
    a.init_cyclic_order();
    b.init_identity();

    start = start_time();
    c.mpy_dumb (b, a);
    time = delta_usec (start);
    LOG ("2Kx2K mpy_dumb() took "<<(time/1000000.0)<<"sec");

    int BS = 128;
    long int total_time=0;
    for (int i=0; i<4; ++i) {
	auto start = start_time();
	d.mpy1 (a, b, BS);
	long int time = delta_usec (start);
	c.compare (d);
	LOG ("2Kx2K mpy1 took "<<(time/1000000.0)<<"sec");
	total_time += time;
    }
    LOG ("2Kx2K mpy1 took an average of "<<(total_time/4000000.0)<<"sec\n");

    // mpy2: using 1, 2, 4, 8 and 16 cores.
    run_mpy2 (BS,  1, a, b, c, d);	// Parameters are BS, # cores, matrices
    run_mpy2 (BS,  2, a, b, c, d);
    run_mpy2 (BS,  4, a, b, c, d);
    run_mpy2 (BS,  8, a, b, c, d);
    run_mpy2 (BS, 16, a, b, c, d);
}
