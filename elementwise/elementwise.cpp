#include<iostream>
#include<chrono>
using namespace std;
void elementwise_add(const float* A, const float* B, float* C, int size) {
for(int i =0;i<size;i++){
    C[i] = A[i] + B[i];
}
}


int main(){
    int size = 10000000;
    float* A = new float[size]; 
    float* B = new float[size];
    float* C = new float[size];
    for (int i = 0; i < size; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    auto start = chrono::high_resolution_clock::now();
    elementwise_add(A,B,C,size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli> elapsed = end - start;
    cout<<"CPU Time = "<<elapsed.count()<<"ms\n";
    cout << "Result verification C[last] = " << C[size-1] << endl;
    delete[] A; delete[] B; delete[] C;
    return 0;
}

//compilation flags optimizations
//No optimization  ≈ 31.19 ms
//-O1              ≈ 13.03 ms
//-O2              ≈ 13.08 ms
//-O3              ≈ 12.91 ms