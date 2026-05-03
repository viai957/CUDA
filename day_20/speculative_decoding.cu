/*
 * Day 20 - Speculative Decoding Acceptance Kernel
 *
 * Purpose:
 *   Verify a short draft-model token proposal against a target model and emit
 *   the next committed token(s). This is the inference-time algorithm used to
 *   trade cheap draft-model compute for fewer expensive target-model steps.
 *
 * Math / Algorithm:
 *   Draft proposes gamma tokens y_0..y_{gamma-1} with probabilities q_i(.).
 *   Target evaluates the same positions plus one bonus position, producing
 *   p_0(.), ..., p_gamma(.).
 *
 *   For each draft token y_i:
 *     accept_prob_i = min(1, p_i(y_i) / q_i(y_i))
 *     accept y_i iff u_i <= accept_prob_i and all earlier draft tokens accepted.
 *
 *   If the first rejection is at r < gamma:
 *     sample replacement z from normalize(max(p_r - q_r, 0)).
 *   If all gamma tokens are accepted:
 *     sample bonus token z from p_gamma.
 *
 * Inputs / Outputs:
 *   target_probs:    float[B, gamma + 1, V], normalized target distributions.
 *   draft_probs:     float[B, gamma, V], normalized draft distributions.
 *   draft_tokens:    int[B, gamma], draft token ids.
 *   accept_uniforms: float[B, gamma], uniform random numbers in [0, 1).
 *   sample_uniforms: float[B], uniform random numbers in [0, 1).
 *   out_tokens:      int[B, gamma + 1], accepted prefix plus correction/bonus,
 *                    trailing entries set to -1.
 *   accepted_counts: int[B], number of draft tokens accepted.
 *   rejected_at:     int[B], first rejected draft index, or -1 if all accepted.
 *
 * Assumptions:
 *   - Probability tensors are already softmax-normalized and row-major.
 *   - gamma is small (typical speculative decoding uses 4-8 draft tokens).
 *   - V can be large; sampling is parallelized over vocab chunks per batch row.
 *   - q_i(y_i) may be zero in adversarial tests; this is guarded explicitly.
 *   - This single-file teaching implementation targets fp32 correctness. A
 *     production path would consume fp16/bf16 logits, fuse softmax/top-k, and
 *     use Philox RNG inside the kernel.
 *
 * Parallel Strategy:
 *   - One CUDA block per batch row.
 *   - Thread 0 resolves the gamma-length accept prefix (tiny dependent loop).
 *   - All threads cooperatively compute chunk sums over V for residual/bonus
 *     sampling, then one owner thread scans its chunk to find the sampled token.
 *   - Vocab chunking preserves deterministic first-token-by-cumulative-order
 *     behavior while avoiding a serial O(V) sample over the whole vocabulary.
 *
 * Mixed Precision Policy:
 *   - This file uses fp32 probabilities and fp32 accumulation.
 *   - Safe extension: read fp16/bf16 probabilities, cast to fp32 for ratio,
 *     residual mass, and cumulative sampling.
 *
 * Distributed Hooks:
 *   - Speculative verification is embarrassingly data-parallel over batch rows.
 *   - In tensor-parallel serving, target/draft logits must be all-gathered or
 *     sampled from a sharded-vocab sampler before this acceptance kernel.
 *
 * Complexity:
 *   - Acceptance prefix: O(B * gamma).
 *   - Sampling: O(B * V) reads and adds for either residual or bonus distribution.
 *   - Bytes moved per row: approximately (gamma token lookups + V distribution
 *     scan) * sizeof(float), plus small int outputs.
 *
 * Unit Tests:
 *   - Deterministic tiny distributions cover all-accepted, first-token rejected,
 *     and later-token rejected cases.
 *   - Random normalized distributions compare GPU output with CPU reference.
 *
 * Build / Run:
 *   nvcc -O3 -std=c++17 -DUNIT_TEST -arch=sm_80 -o speculative_decoding speculative_decoding.cu
 *   ./speculative_decoding
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
    } while (0)

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

__host__ __device__ static inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

static float distribution_value(
    const float* target_probs,
    const float* draft_probs,
    int b,
    int pos,
    int v,
    int gamma,
    int V,
    bool all_accepted
) {
    const int target_base = (b * (gamma + 1) + pos) * V;
    if (all_accepted) {
        return target_probs[target_base + v];
    }

    const int draft_base = (b * gamma + pos) * V;
    return fmaxf(target_probs[target_base + v] - draft_probs[draft_base + v], 0.0f);
}

static int sample_from_distribution_cpu(
    const float* target_probs,
    const float* draft_probs,
    int b,
    int pos,
    int gamma,
    int V,
    bool all_accepted,
    float u
) {
    float mass = 0.0f;
    for (int v = 0; v < V; ++v) {
        mass += distribution_value(target_probs, draft_probs, b, pos, v, gamma, V, all_accepted);
    }

    if (!(mass > 0.0f) || !std::isfinite(mass)) {
        int best = 0;
        float best_val = -FLT_MAX;
        const int target_base = (b * (gamma + 1) + pos) * V;
        for (int v = 0; v < V; ++v) {
            const float p = target_probs[target_base + v];
            if (p > best_val) {
                best_val = p;
                best = v;
            }
        }
        return best;
    }

    const float threshold = fminf(fmaxf(u, 0.0f), 0.99999994f) * mass;
    float cdf = 0.0f;
    for (int v = 0; v < V; ++v) {
        cdf += distribution_value(target_probs, draft_probs, b, pos, v, gamma, V, all_accepted);
        if (cdf >= threshold) {
            return v;
        }
    }
    return V - 1;
}

void speculative_decode_cpu(
    const std::vector<float>& target_probs,
    const std::vector<float>& draft_probs,
    const std::vector<int>& draft_tokens,
    const std::vector<float>& accept_uniforms,
    const std::vector<float>& sample_uniforms,
    std::vector<int>& out_tokens,
    std::vector<int>& accepted_counts,
    std::vector<int>& rejected_at,
    int B,
    int gamma,
    int V
) {
    std::fill(out_tokens.begin(), out_tokens.end(), -1);

    for (int b = 0; b < B; ++b) {
        int first_reject = gamma;

        for (int i = 0; i < gamma; ++i) {
            const int token = draft_tokens[b * gamma + i];
            const float p = target_probs[(b * (gamma + 1) + i) * V + token];
            const float q = draft_probs[(b * gamma + i) * V + token];
            float accept_prob = 0.0f;
            if (q <= 0.0f) {
                accept_prob = (p > 0.0f) ? 1.0f : 0.0f;
            } else {
                accept_prob = fminf(1.0f, p / q);
            }

            if (accept_uniforms[b * gamma + i] > accept_prob) {
                first_reject = i;
                break;
            }
        }

        const bool all_accepted = first_reject == gamma;
        const int sample_pos = all_accepted ? gamma : first_reject;
        const int sampled = sample_from_distribution_cpu(
            target_probs.data(), draft_probs.data(), b, sample_pos, gamma, V,
            all_accepted, sample_uniforms[b]);

        for (int i = 0; i < first_reject; ++i) {
            out_tokens[b * (gamma + 1) + i] = draft_tokens[b * gamma + i];
        }
        out_tokens[b * (gamma + 1) + first_reject] = sampled;
        accepted_counts[b] = first_reject;
        rejected_at[b] = all_accepted ? -1 : first_reject;
    }
}

__device__ float device_distribution_value(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    int b,
    int pos,
    int v,
    int gamma,
    int V,
    bool all_accepted
) {
    const int target_base = (b * (gamma + 1) + pos) * V;
    if (all_accepted) {
        return target_probs[target_base + v];
    }

    const int draft_base = (b * gamma + pos) * V;
    return fmaxf(target_probs[target_base + v] - draft_probs[draft_base + v], 0.0f);
}

__global__ void speculative_decode_kernel(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const int* __restrict__ draft_tokens,
    const float* __restrict__ accept_uniforms,
    const float* __restrict__ sample_uniforms,
    int* __restrict__ out_tokens,
    int* __restrict__ accepted_counts,
    int* __restrict__ rejected_at,
    int B,
    int gamma,
    int V
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B) {
        return;
    }

    extern __shared__ float smem[];
    float* chunk_sums = smem;                 // [blockDim.x]
    float* prefix_slot = chunk_sums + blockDim.x;
    int* shared_ints = (int*)(prefix_slot + 1);
    int& first_reject = shared_ints[0];
    int& sampled_token = shared_ints[1];
    int& owner_thread = shared_ints[2];
    float& prefix_before_owner = prefix_slot[0];

    if (tid == 0) {
        first_reject = gamma;
        sampled_token = 0;
        owner_thread = 0;

        for (int i = 0; i < gamma; ++i) {
            const int token = draft_tokens[b * gamma + i];
            const float p = target_probs[(b * (gamma + 1) + i) * V + token];
            const float q = draft_probs[(b * gamma + i) * V + token];
            float accept_prob = 0.0f;
            if (q <= 0.0f) {
                accept_prob = (p > 0.0f) ? 1.0f : 0.0f;
            } else {
                accept_prob = fminf(1.0f, p / q);
            }

            if (accept_uniforms[b * gamma + i] > accept_prob) {
                first_reject = i;
                break;
            }
        }
    }
    __syncthreads();

    const bool all_accepted = first_reject == gamma;
    const int sample_pos = all_accepted ? gamma : first_reject;
    const int chunk = ceil_div_int(V, blockDim.x);
    const int begin = tid * chunk;
    const int end = min(begin + chunk, V);

    float local_sum = 0.0f;
    for (int v = begin; v < end; ++v) {
        local_sum += device_distribution_value(
            target_probs, draft_probs, b, sample_pos, v, gamma, V, all_accepted);
    }
    chunk_sums[tid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int t = 0; t < blockDim.x; ++t) {
            total += chunk_sums[t];
        }

        if (!(total > 0.0f) || !isfinite(total)) {
            int best = 0;
            float best_val = -FLT_MAX;
            const int target_base = (b * (gamma + 1) + sample_pos) * V;
            for (int v = 0; v < V; ++v) {
                const float p = target_probs[target_base + v];
                if (p > best_val) {
                    best_val = p;
                    best = v;
                }
            }
            sampled_token = best;
            owner_thread = -1;
        } else {
            const float u = fminf(fmaxf(sample_uniforms[b], 0.0f), 0.99999994f);
            const float threshold = u * total;
            float prefix = 0.0f;
            int owner = blockDim.x - 1;
            float owner_prefix = 0.0f;
            for (int t = 0; t < blockDim.x; ++t) {
                const float next = prefix + chunk_sums[t];
                if (threshold <= next) {
                    owner = t;
                    owner_prefix = prefix;
                    break;
                }
                prefix = next;
            }
            owner_thread = owner;
            prefix_before_owner = owner_prefix;
        }
    }
    __syncthreads();

    if (tid == owner_thread) {
        const float total_before = prefix_before_owner;
        float cdf = total_before;

        float total = total_before;
        for (int t = owner_thread; t < blockDim.x; ++t) {
            total += chunk_sums[t];
        }
        const float threshold = fminf(fmaxf(sample_uniforms[b], 0.0f), 0.99999994f) * total;

        int chosen = min(begin, V - 1);
        for (int v = begin; v < end; ++v) {
            cdf += device_distribution_value(
                target_probs, draft_probs, b, sample_pos, v, gamma, V, all_accepted);
            if (cdf >= threshold) {
                chosen = v;
                break;
            }
        }
        sampled_token = chosen;
    }
    __syncthreads();

    if (tid == 0) {
        const int out_base = b * (gamma + 1);
        for (int i = 0; i < gamma + 1; ++i) {
            out_tokens[out_base + i] = -1;
        }
        for (int i = 0; i < first_reject; ++i) {
            out_tokens[out_base + i] = draft_tokens[b * gamma + i];
        }
        out_tokens[out_base + first_reject] = sampled_token;
        accepted_counts[b] = first_reject;
        rejected_at[b] = all_accepted ? -1 : first_reject;
    }
}

void launch_speculative_decode(
    const float* d_target_probs,
    const float* d_draft_probs,
    const int* d_draft_tokens,
    const float* d_accept_uniforms,
    const float* d_sample_uniforms,
    int* d_out_tokens,
    int* d_accepted_counts,
    int* d_rejected_at,
    int B,
    int gamma,
    int V
) {
    dim3 grid(B);
    dim3 block(BLOCK_DIM);
    const size_t smem_bytes = (BLOCK_DIM + 1) * sizeof(float) + 3 * sizeof(int);
    speculative_decode_kernel<<<grid, block, smem_bytes>>>(
        d_target_probs, d_draft_probs, d_draft_tokens, d_accept_uniforms,
        d_sample_uniforms, d_out_tokens, d_accepted_counts, d_rejected_at,
        B, gamma, V);
    CUDA_CHECK_KERNEL();
}

#ifdef UNIT_TEST
static void normalize_rows(std::vector<float>& probs, int rows, int V) {
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int v = 0; v < V; ++v) {
            probs[r * V + v] = fmaxf(probs[r * V + v], 1.0e-7f);
            sum += probs[r * V + v];
        }
        for (int v = 0; v < V; ++v) {
            probs[r * V + v] /= sum;
        }
    }
}

static void run_case(
    const char* name,
    int B,
    int gamma,
    int V,
    std::vector<float> target_probs,
    std::vector<float> draft_probs,
    std::vector<int> draft_tokens,
    std::vector<float> accept_uniforms,
    std::vector<float> sample_uniforms
) {
    printf("\n=== %s: B=%d gamma=%d V=%d ===\n", name, B, gamma, V);

    std::vector<int> ref_out(B * (gamma + 1), -1);
    std::vector<int> ref_counts(B, 0);
    std::vector<int> ref_reject(B, -1);
    speculative_decode_cpu(target_probs, draft_probs, draft_tokens,
                           accept_uniforms, sample_uniforms,
                           ref_out, ref_counts, ref_reject, B, gamma, V);

    float *d_target = nullptr, *d_draft = nullptr, *d_accept_u = nullptr, *d_sample_u = nullptr;
    int *d_draft_tokens = nullptr, *d_out = nullptr, *d_counts = nullptr, *d_reject = nullptr;

    CUDA_CHECK(cudaMalloc(&d_target, target_probs.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft, draft_probs.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accept_u, accept_uniforms.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_u, sample_uniforms.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft_tokens, draft_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, ref_out.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, ref_counts.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reject, ref_reject.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_target, target_probs.data(), target_probs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft, draft_probs.data(), draft_probs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accept_u, accept_uniforms.data(), accept_uniforms.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_u, sample_uniforms.data(), sample_uniforms.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_tokens, draft_tokens.data(), draft_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));

    launch_speculative_decode(d_target, d_draft, d_draft_tokens, d_accept_u,
                              d_sample_u, d_out, d_counts, d_reject, B, gamma, V);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> got_out(ref_out.size(), -1);
    std::vector<int> got_counts(ref_counts.size(), 0);
    std::vector<int> got_reject(ref_reject.size(), -1);
    CUDA_CHECK(cudaMemcpy(got_out.data(), d_out, got_out.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_counts.data(), d_counts, got_counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_reject.data(), d_reject, got_reject.size() * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    ok = ok && (got_out == ref_out);
    ok = ok && (got_counts == ref_counts);
    ok = ok && (got_reject == ref_reject);

    for (int b = 0; b < B; ++b) {
        printf("  row %d: accepted=%d rejected_at=%d tokens:",
               b, got_counts[b], got_reject[b]);
        for (int i = 0; i < gamma + 1; ++i) {
            printf(" %d", got_out[b * (gamma + 1) + i]);
        }
        printf("\n");
    }
    printf("  CPU vs GPU: %s\n", ok ? "PASS" : "FAIL");

    if (!ok) {
        fprintf(stderr, "Reference mismatch in case: %s\n", name);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_draft));
    CUDA_CHECK(cudaFree(d_accept_u));
    CUDA_CHECK(cudaFree(d_sample_u));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_reject));
}

static void run_tiny_handwritten_tests() {
    const int B = 3;
    const int gamma = 3;
    const int V = 8;

    std::vector<float> target(B * (gamma + 1) * V, 0.01f);
    std::vector<float> draft(B * gamma * V, 0.01f);
    std::vector<int> draft_tokens = {
        2, 3, 4,
        1, 5, 6,
        0, 7, 2,
    };
    std::vector<float> accept_u = {
        0.01f, 0.01f, 0.01f,   // all accepted
        0.99f, 0.01f, 0.01f,   // reject at 0
        0.01f, 0.99f, 0.01f,   // reject at 1
    };
    std::vector<float> sample_u = {0.20f, 0.30f, 0.70f};

    for (int b = 0; b < B; ++b) {
        for (int pos = 0; pos < gamma + 1; ++pos) {
            target[(b * (gamma + 1) + pos) * V + ((pos + b + 1) % V)] = 0.70f;
        }
        for (int pos = 0; pos < gamma; ++pos) {
            draft[(b * gamma + pos) * V + draft_tokens[b * gamma + pos]] = 0.60f;
        }
    }

    // Row-specific ratios force the desired accept/reject decisions.
    target[(0 * (gamma + 1) + 0) * V + 2] = 0.90f;
    target[(0 * (gamma + 1) + 1) * V + 3] = 0.90f;
    target[(0 * (gamma + 1) + 2) * V + 4] = 0.90f;

    target[(1 * (gamma + 1) + 0) * V + 1] = 0.10f;
    draft[(1 * gamma + 0) * V + 1] = 0.80f;

    target[(2 * (gamma + 1) + 0) * V + 0] = 0.90f;
    target[(2 * (gamma + 1) + 1) * V + 7] = 0.10f;
    draft[(2 * gamma + 1) * V + 7] = 0.80f;

    normalize_rows(target, B * (gamma + 1), V);
    normalize_rows(draft, B * gamma, V);
    run_case("handwritten accept/reject", B, gamma, V, target, draft,
             draft_tokens, accept_u, sample_u);
}

static void run_random_test() {
    const int B = 5;
    const int gamma = 5;
    const int V = 4096;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> token_dist(0, V - 1);

    std::vector<float> target(B * (gamma + 1) * V);
    std::vector<float> draft(B * gamma * V);
    std::vector<int> draft_tokens(B * gamma);
    std::vector<float> accept_u(B * gamma);
    std::vector<float> sample_u(B);

    for (float& x : target) x = dist(rng) + 1.0e-4f;
    for (float& x : draft) x = dist(rng) + 1.0e-4f;
    normalize_rows(target, B * (gamma + 1), V);
    normalize_rows(draft, B * gamma, V);

    for (int& x : draft_tokens) x = token_dist(rng);
    for (float& x : accept_u) x = dist(rng);
    for (float& x : sample_u) x = dist(rng);

    run_case("random normalized distributions", B, gamma, V, target, draft,
             draft_tokens, accept_u, sample_u);
}

static void benchmark_random() {
    const int B = 32;
    const int gamma = 5;
    const int V = 32000;
    const int runs = 200;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> token_dist(0, V - 1);

    std::vector<float> target(B * (gamma + 1) * V);
    std::vector<float> draft(B * gamma * V);
    std::vector<int> draft_tokens(B * gamma);
    std::vector<float> accept_u(B * gamma);
    std::vector<float> sample_u(B);

    for (float& x : target) x = dist(rng) + 1.0e-4f;
    for (float& x : draft) x = dist(rng) + 1.0e-4f;
    normalize_rows(target, B * (gamma + 1), V);
    normalize_rows(draft, B * gamma, V);
    for (int& x : draft_tokens) x = token_dist(rng);
    for (float& x : accept_u) x = dist(rng);
    for (float& x : sample_u) x = dist(rng);

    float *d_target = nullptr, *d_draft = nullptr, *d_accept_u = nullptr, *d_sample_u = nullptr;
    int *d_draft_tokens = nullptr, *d_out = nullptr, *d_counts = nullptr, *d_reject = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target, target.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft, draft.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accept_u, accept_u.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_u, sample_u.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft_tokens, draft_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, B * (gamma + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reject, B * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_target, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft, draft.data(), draft.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accept_u, accept_u.data(), accept_u.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_u, sample_u.data(), sample_u.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_tokens, draft_tokens.data(), draft_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        launch_speculative_decode(d_target, d_draft, d_draft_tokens, d_accept_u,
                                  d_sample_u, d_out, d_counts, d_reject, B, gamma, V);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        launch_speculative_decode(d_target, d_draft, d_draft_tokens, d_accept_u,
                                  d_sample_u, d_out, d_counts, d_reject, B, gamma, V);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;

    const double bytes_read = (double)B * (double)V * 2.0 * sizeof(float);
    const double bandwidth_gbs = bytes_read / (ms / 1000.0) / 1.0e9;
    printf("\n=== Benchmark: B=%d gamma=%d V=%d ===\n", B, gamma, V);
    printf("  avg time: %.4f ms, effective prob-read bandwidth: %.2f GB/s\n", ms, bandwidth_gbs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_draft));
    CUDA_CHECK(cudaFree(d_accept_u));
    CUDA_CHECK(cudaFree(d_sample_u));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_reject));
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s, SMs=%d, blockDim=%d\n", prop.name, prop.multiProcessorCount, BLOCK_DIM);

    run_tiny_handwritten_tests();
    run_random_test();
    benchmark_random();

    printf("\nAll Day 20 speculative decoding checks passed.\n");
    return 0;
}
#endif

/*
 * Profiling:
 *   nsys profile --stats=true ./speculative_decoding
 *
 * Tuning tips:
 *   - BLOCK_DIM=256 gives 125 vocab entries/thread for V=32k. Increase to 512
 *     on Hopper when occupancy remains healthy and register pressure is low.
 *   - Fuse target/draft softmax and top-k filtering upstream to avoid materializing
 *     dense fp32 probability tensors for very large vocabularies.
 *   - For tensor-parallel vocab shards, run local residual sums first, all-reduce
 *     the mass, then sample by shard prefix to avoid a full probability gather.
 */
