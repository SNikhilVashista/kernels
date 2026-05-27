#pragma once

void launch_scores_naive(
    const float* d_query,
    const float* d_key,
    float* d_scores,
    int M,
    int N,
    int d
);

void launch_scores_tiled(
    const float* d_query,
    const float* d_key,
    float* d_scores,
    int M,
    int N,
    int d
);