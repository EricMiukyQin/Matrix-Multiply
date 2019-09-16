#include <vector>
#include <sstream>
#include <iostream>
#include <random> 
using namespace std;
#include "bits.hxx"
#include "ee155_utils.hxx"
#include "matrix.hxx"

Matrix::Matrix (int nRows, int nCols) {
    if (nCols == -1)	// If we're just called as Matrix(5), then
	nCols=nRows;	// build a square 5x5 matrix.

    // Allocate power of 2 # of cols, to make index() faster.
    for (_log2NColsAlc=0;1<<_log2NColsAlc<nCols; ++_log2NColsAlc)
	;

    this->_nRows = nRows;
    this->_nCols = nCols;
    unsigned int n_elements = (1<<_log2NColsAlc) * nRows;
    this->data = vector<float> (n_elements);
}

bool Matrix::operator== (const Matrix &other) const {
    return (this->data == other.data);
}

// Like ==. But: on mismatch, prints the first mismatching element and dies.
void Matrix::compare (const Matrix &M2) const {
    if ((_nRows != M2._nRows) || (_nCols != M2._nCols))
	DIE ("Comparing unequal-sized matrices");

    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    if ((*this)(r,c) != M2(r,c))
		DIE ("M1["<<r<<","<<c<<"]="<<(*this)(r,c)
		     << ", M2["<<r<<","<<c<<"]="<<M2(r,c));
}

void Matrix::init_identity() {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = ((r==c)?1.0F:0.0F);
}

void Matrix::init_cyclic_order() {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float> (r+c);
}

void Matrix::init_count_order () {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float>	(r*this->_nCols + c);
}

void Matrix::init_random (int max) {
    default_random_engine gen;
    uniform_int_distribution<int> dist(0,max);

    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float> (dist(gen));
}

// Printing support.
string Matrix::row_str(int row) const {
    ostringstream os;
    os << "{";
    for (int c=0; c<_nCols; ++c)
	os << (c==0?"":", ") << (*this)(row,c);
    os << "}";
    return (os.str());
}
string Matrix::str() const {
    string s = "{";
    for (int r=0; r<_nRows; ++r)
	s += this->row_str(r);
    s += "}";
    return (s);
}
