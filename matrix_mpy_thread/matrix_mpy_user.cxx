#include "matrix.hxx"

using namespace std;
#include <thread>

////////////////////////////////////////////////////////////////
// One thread, blocked. Loop order rB, kB, cB, r, k, c.
// This function is for you to write.
//
void Matrix::mpy1 (const Matrix &A, const Matrix &B, int BS) {
    int NBLK= this->N()/BS;     // An NBLKxNBLK grid of blocks
    assert (this->N() >= BS);
    
    for (int rB = 0; rB < NBLK; rB++)
        for (int kB = 0; kB < NBLK; kB++)
            for (int cB = 0; cB < NBLK; cB++)
                for (int r = 0; r < BS; r++)
                    for (int k = 0; k < BS; k++) {
                        int rr = rB * BS + r;
                        int kk = kB * BS + k;
                        for (int c = 0; c < BS; c++) {
                            int cc = cB * BS + c;
                            if (kk == 0)
                                this->data[this->index(rr, cc)] = A(rr, kk) * B(kk, cc);
                            else
                                this->data[this->index(rr, cc)] += A(rr, kk) * B(kk, cc);
                        }
                    }
}

////////////////////////////////////////////////////////////////
// Multithreaded, blocked version.
//
// This function, th_func2(), does the per-thread work of multithreaded, blocked
// matrix multiplication.
static void th_func2 (Matrix &OUT, const Matrix &A, const Matrix &B, int me, int BS, int NBLK, int N_threads) {
    // Called every thread
    // Thread[0] calculates (0 + i * N_threads) lines in output Matrix
    // Thread[1] calculates (1 + i * N_threads) lines in output Matrix
    // ...
    // Thread[me] calculates (me + i * N_threads) lines in output Matrix

    for (int i = 0; i < NBLK / N_threads; i++) {
        int rB = me + i * N_threads;
        for (int kB = 0; kB < NBLK; kB++) {
            for (int cB = 0; cB < NBLK; cB++) {
                for (int r = 0; r < BS; r++) {
                    for (int k = 0; k < BS; k++) {
                        int rr = rB * BS + r;
                        int kk = kB * BS + k;
                        for (int c = 0; c < BS; c++) {
                            int cc = cB * BS + c;
                            if (kk == 0)
                                OUT(rr, cc) = A(rr, kk) * B(kk, cc);
                            else
                                OUT(rr, cc) += A(rr, kk) * B(kk, cc);
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// This function does multithreaded, blocked matrix multiplication. It is for
// you to write. The parameters:
//	A, B: the input matrices
//	BS: block size; i.e., you should use blocks of BSxBS.
//	n_procs: how many processors to use.
// You must store the output in (*this), which already has its .data array
// allocated (but not necessarily cleared).
// Note that you can find out the size of the A, B and (*this) matrices by
// either looking at the _N member variable, or calling Matrix.N().
void Matrix::mpy2 (const Matrix &A, const Matrix &B, int BS, int n_procs) {
    vector<thread> threads;
    int NBLK= this->N()/BS;     // An NBLKxNBLK grid of blocks
    assert (this->N() >= BS);

    for (int me = 0; me < n_procs; me++)
        threads.push_back(thread(th_func2, ref(*this), ref(A), ref(B), me, BS, NBLK, n_procs));
    for (auto &it : threads)
        it.join();
}

