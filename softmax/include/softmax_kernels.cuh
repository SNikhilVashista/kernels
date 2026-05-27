#pragma once

void launch_softmax_naive(float *d_scores, int M, int N);

void launch_softmax_shared(float *d_scores, int M, int N);

void launch_softmax_online(float *d_scores, int M, int N);