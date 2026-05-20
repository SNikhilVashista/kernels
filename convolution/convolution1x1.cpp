#include<iostream>
#include<chrono>
using namespace std;
void conv1x1cpu(float* input, float* filter, float* output,
                int H, int W, int Cin, int Cout) {
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            for(int oc = 0; oc < Cout; oc++) {
                float sum = 0.0f;
                for(int ic = 0; ic < Cin; ic++) {
                    int input_idx = (h * W + w) * Cin + ic;
                    int filter_idx = ic * Cout + oc;
                    sum += input[input_idx] * filter[filter_idx];
                }
                int output_idx = (h * W + w) * Cout + oc;
                output[output_idx] = sum;
            }
        }
    }
}

int main(){
    int h=4,w=4,Cin=64,Cout=64;
    
    float *input = new float[h*w*Cin];
    float *filter = new float[Cin*Cout];
    float *output = new float[h*w*Cout];
    for(int i = 0; i < h*w*Cin; i++) {
    input[i] = 1.0f;
    }

for(int i = 0; i < Cin*Cout; i++) {
    filter[i] = 1.0f;
    }

for(int i = 0; i < h*w*Cout; i++) {
    output[i] = 0.0f;
    }
    auto start = chrono::high_resolution_clock::now();
    conv1x1cpu(input,filter,output,h,w,Cin,Cout);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli> elapsed = end-start;
    cout<<"CPU Time = "<<elapsed.count()<<"ms\n";
    cout << output[0] << std::endl;
    cout << output[h*w*Cout - 1] << std::endl;
    return 0;
}