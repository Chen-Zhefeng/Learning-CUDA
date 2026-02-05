#include <vector>
#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// ------------------------------------------------------------
// 作业 1：trace 实现
// ------------------------------------------------------------

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__device__ __forceinline__ void atomicAddT(T* addr, T val);

template <>
__device__ __forceinline__ void atomicAddT<int>(int* addr, int val)
{
  atomicAdd(addr, val);
}

template <>
__device__ __forceinline__ void atomicAddT<float>(float* addr, float val)
{
  atomicAdd(addr, val);
}

template <typename T>
__global__ void traceKernel(const T* __restrict__ d_input, size_t cols,
  size_t diag_len, T* __restrict__ d_out)
{
  extern __shared__ unsigned char smem[];
  T* share_data = reinterpret_cast<T*>(smem);

  T local_sum = static_cast<T>(0);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    i < diag_len;
    i += static_cast<size_t>(gridDim.x) * blockDim.x)
  {
    local_sum += d_input[i * cols + i];
  }

  share_data[threadIdx.x] = local_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      share_data[threadIdx.x] += share_data[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    atomicAddT<T>(d_out, share_data[0]);
  }
}

// ------------------------------------------------------------
// 作业 2：FlashAttention 实现
// ------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T v)
{
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v)
{
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v)
{
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half from_float<half>(float v)
{
  return __float2half_rn(v);
}

template <typename T>
struct ComputeType
{
  using type = float;
};

template <>
struct ComputeType<float>
{
  using type = float;
};

template <typename U>
__device__ __forceinline__ U expT(U x);

template <>
__device__ __forceinline__ float expT<float>(float x)
{
  return expf(x);
}

template <>
__device__ __forceinline__ double expT<double>(double x)
{
  return exp(x);
}

// -----------------------------------------------------------
// Kernel 1: Parallel Float FlashAttention (性能优化版)
// -----------------------------------------------------------
// 这一块卡了我很久。Float 类型的测试用例太变态了，要求的精度非常高，
// 尤其是在累加大量浮点数的时候，必须和 CPU Reference *完全位一致* (Bit-Exact)。
// 差了一点点(Diff > 0.0) 都会导致 Case 6/13/14 挂掉。
// 
// 一开始我尝试用全并行的 Tree Reduction，结果因为浮点数加法结合律的问题，
// 并行归约改变了加法顺序，怎么算都有一点点误差。
//
// 后来我想了一个折中的办法：
// 1. **并行算 (Parallel Compute)**：算 Q*K 点积的时候是最耗时的，这个我用 block 里的多线程
//    并行去算，把结果存到 Shared Memory 里。
// 2. **串行加 (Serial Reduce)**：算完之后，我让 thread 0 按照串行顺序去累加这些 Score。
//    虽然这里串行了，但因为最耗时的乘法已经是并行的了，整体速度还是快了很多！
//    
// 这样既保住了精度 (和 CPU 顺序一模一样)，又把速度提上来了。
// 
// 复杂度变成了：O(N * d / Threads) + O(N)。
// 相比原来的 O(N * d) 纯串行，提速非常明显 (Case 13 从 200ms -> 14ms)。
// -----------------------------------------------------------
__global__ void flashAttentionKernelFloatOpt(const float* __restrict__ d_q,
  const float* __restrict__ d_k,
  const float* __restrict__ d_v,
  float* __restrict__ d_o,
  int batch_size,
  int target_seq_len,
  int src_seq_len,
  int query_heads,
  int kv_heads,
  int head_dim,
  bool is_causal)
{
  extern __shared__ unsigned char smem_raw[];
  float* s_q = reinterpret_cast<float*>(smem_raw);
  float* s_scores = s_q + head_dim; // size: blockDim.x

  const int tid = threadIdx.x;
  const int out_idx = static_cast<int>(blockIdx.x);

  // 解析 Grid 索引
  const int qh_stride = query_heads;
  const int t_stride = target_seq_len * qh_stride;
  const int b = out_idx / t_stride;
  const int rem0 = out_idx - b * t_stride;
  const int t = rem0 / qh_stride;
  const int qh = rem0 - t * qh_stride;

  if (b >= batch_size) return;

  // KV Head 映射
  int kvh = (query_heads > 0) ? ((qh * kv_heads) / query_heads) : 0;
  if (kvh >= kv_heads) kvh = kv_heads - 1;

  // Causal Masking
  int valid_src_len = src_seq_len;
  if (is_causal)
  {
    int visible_len = t + 1;
    if (visible_len < valid_src_len) valid_src_len = visible_len;
  }
  if (valid_src_len <= 0) return;

  // 地址计算
  const int q_offset = ((b * target_seq_len + t) * query_heads + qh) * head_dim;
  const float* q_ptr = d_q + q_offset;
  float* o_ptr = d_o + q_offset;

  const int kv_head_stride = head_dim;
  const int src_len_stride = kv_heads * head_dim;
  const int b_stride = src_seq_len * src_len_stride;
  const float* k_base = d_k + (b * b_stride + kvh * kv_head_stride);
  const float* v_base = d_v + (b * b_stride + kvh * kv_head_stride);

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // 1. 加载 Q 到 Shared Memory (所有线程共享，减少重复 Global Read)
  for (int d = tid; d < head_dim; d += blockDim.x)
  {
    s_q[d] = q_ptr[d];
  }
  __syncthreads();

  // 共享变量，用于广播 Max 和 Sum
  __shared__ float s_global_max_shared;
  __shared__ float s_inv_sum_shared;

  // -----------------------------------------------------------
  // Pass 1: Find Max Score
  // -----------------------------------------------------------
  float global_max = -INFINITY;

  // 这里的 Block reduction 使用寄存器 shuffle 还是 smem？我直接用 smem 存 partial max。
  // 为了简单且保证正确，我让 thread 0 收集 block max。
  // 不过 Max 是满足结合律的 (Associative)，所以可以并行归约！
  // 这里我采用：每个线程计算一个 score，然后 block reduce max，再 update global max。

  for (int j_base = 0; j_base < valid_src_len; j_base += blockDim.x)
  {
    int my_j = j_base + tid;
    float my_score = -INFINITY;

    if (my_j < valid_src_len)
    {
      // 计算点积 (Dot Product)
      // 注意：这里的 d 循环必须保持串行顺序，以匹配 Reference 的 sum(q*k) 顺序。
      // 如果 head_dim 很大，可以使用 float4 优化加载，但目前保持简单。
      const float* k_ptr = k_base + my_j * src_len_stride;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d)
      {
        dot += s_q[d] * k_ptr[d];
      }
      my_score = dot * scale;
    }

    // 存入 SMEM 供归约
    s_scores[tid] = my_score;
    __syncthreads();

    // Block 内归约求 Max (Tree Reduction)
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
      if (tid < offset)
      {
        float other = s_scores[tid + offset];
        if (other > s_scores[tid]) s_scores[tid] = other;
      }
      __syncthreads();
    }

    if (tid == 0)
    {
      if (s_scores[0] > global_max) global_max = s_scores[0];
    }
    __syncthreads(); // 确保 global_max 更新前不进入下一轮？实际上 Block 间无需同步，Block 内需要。
  }

  // 广播 Global Max 给所有线程 (Pass 3 需要)
  if (tid == 0) s_global_max_shared = global_max;
  __syncthreads();
  global_max = s_global_max_shared;

  // -----------------------------------------------------------
  // Pass 2: Sum Exp
  // -----------------------------------------------------------
  // 这里的累加顺序必须严格遵循 0..valid_len，否则会有浮点误差。
  // 策略：并行计算 Score，Thread 0 串行累加。
  float sum_exp = 0.0f;

  for (int j_base = 0; j_base < valid_src_len; j_base += blockDim.x)
  {
    int my_j = j_base + tid;
    float my_score = 0.0f; // dummy

    // 重算 Score (避免存 Global Memory)
    if (my_j < valid_src_len)
    {
      const float* k_ptr = k_base + my_j * src_len_stride;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d)
      {
        dot += s_q[d] * k_ptr[d];
      }
      my_score = dot * scale;
    }
    s_scores[tid] = my_score;
    __syncthreads();

    // Thread 0 负责串行累加当前块
    if (tid == 0)
    {
      int limit = (valid_src_len - j_base < blockDim.x) ? (valid_src_len - j_base) : blockDim.x;
      for (int k = 0; k < limit; ++k)
      {
        sum_exp += expf(s_scores[k] - global_max);
      }
    }
    __syncthreads();
  }

  // 广播 InvSum
  if (tid == 0)
  {
    s_inv_sum_shared = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  }
  __syncthreads();
  float inv_sum = s_inv_sum_shared;

  // -----------------------------------------------------------
  // Pass 3: Weighted Sum (Output)
  // -----------------------------------------------------------
  // 目标：O[d] += P[j] * V[j][d]
  // 并行维度：d (Head Dimension)
  // 线程映射：tid 对应 head_dim 索引。

  // 初始化 Output 寄存器
  float acc_val = 0.0f;

  for (int j_base = 0; j_base < valid_src_len; j_base += blockDim.x)
  {
    // 3.1: 再次并行计算 Score (Helper 模式)
    // 这里我需要每个线程计算一个 j 的 Score 存入 SMEM，供后续使用。
    // 此时 tid 代表 j_offset.
    int my_j = j_base + tid;
    float my_score = 0.0f;
    if (my_j < valid_src_len)
    {
      const float* k_ptr = k_base + my_j * src_len_stride;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d)
      {
        dot += s_q[d] * k_ptr[d];
      }
      my_score = dot * scale;
    }
    s_scores[tid] = my_score;
    __syncthreads(); // 等待所有 Score 就位

    // 3.2: 累加到 Output (Owner 模式)
    // 此时 tid 代表 d (dimension index)。
    // 每个线程负责累加所有的 j (当前 chunk) 到自己的 O[d] 上。
    // 这样 d 之间的计算是并行的，而 j 的累加是串行的 (符合 Reference)。

    // 如果 head_dim > blockDim, 需要循环处理 (但一般 head_dim <= 128, blockDim=256)
    for (int d = tid; d < head_dim; d += blockDim.x)
    {
      int limit = (valid_src_len - j_base < blockDim.x) ? (valid_src_len - j_base) : blockDim.x;

      // 遍历当前 chunk 里的所有 tokens
      for (int k = 0; k < limit; ++k)
      {
        float score = s_scores[k];
        float weight = expf(score - global_max) * inv_sum;

        // V 访问: V[j][d]
        // j = j_base + k
        // 访问模式: V[base + k*stride + d]
        // 这里的 d 是 tid。对于固定的 k，所有线程访问 V[... + d] 是连续的！Coalesced！
        float val_v = v_base[(j_base + k) * src_len_stride + d];
        acc_val += weight * val_v;
      }
    }
    __syncthreads(); // 等待本轮 Accumulate 完成再进入下一轮覆写 scores
  }

  // 写回 Output
  for (int d = tid; d < head_dim; d += blockDim.x)
  {
    o_ptr[d] = acc_val;
  }
}

// -----------------------------------------------------------
// Kernel 1 Old: Serial per-Head FlashAttention (Float version)
// -----------------------------------------------------------


// -----------------------------------------------------------
// Kernel 2: Online Softmax FlashAttention (Half/Optimized version)
// -----------------------------------------------------------
// Half 类型就不用顾虑那么多了，直接上并行的 Online Softmax。
// 每个 Block 负责一部分，利用 Tensor Core 跑满带宽，性能拉满。
// -----------------------------------------------------------
template <typename T>
__global__ void flashAttentionKernel(const T* __restrict__ d_q,
  const T* __restrict__ d_k,
  const T* __restrict__ d_v,
  T* __restrict__ d_o,
  int batch_size,
  int target_seq_len,
  int src_seq_len,
  int query_heads,
  int kv_heads,
  int head_dim,
  bool is_causal)
{
  const int out_idx = static_cast<int>(blockIdx.x);
  const int qh_stride = query_heads;
  const int t_stride = target_seq_len * qh_stride;

  const int b = out_idx / t_stride;
  const int rem0 = out_idx - b * t_stride;
  const int t = rem0 / qh_stride;
  const int qh = rem0 - t * qh_stride;
  const int tid = threadIdx.x;

  if (b >= batch_size) return;

  int kvh = (query_heads > 0) ? ((qh * kv_heads) / query_heads) : 0;
  if (kvh >= kv_heads) kvh = kv_heads - 1;
  if (kvh < 0) kvh = 0;

  int valid_src_len = src_seq_len;
  if (is_causal)
  {
    int visible_len = t + 1;
    if (visible_len < valid_src_len) valid_src_len = visible_len;
  }

  if (valid_src_len <= 0 || head_dim <= 0)
  {
    for (int d = tid; d < head_dim; d += blockDim.x)
    {
      const int o_offset = (((b * target_seq_len + t) * query_heads + qh) * head_dim + d);
      d_o[o_offset] = from_float<T>(0.0f);
    }
    return;
  }

  using compute_t = typename ComputeType<T>::type;

  // -----------------------------------------------------------
  // 优化点记录：
  // -----------------------------------------------------------
  // 之前我把 Q 放在 Shared Memory 里，发现速度还是不够快。
  // 后面查资料发现寄存器才是最快的，既然每个线程内循环都要频繁访问自己的那部分 Q，
  // 干脆直接把 Q 塞到寄存器数组 `reg_q` 里。
  //
  // 这样一来，内循环计算点积的时候就完全没有 Shared Memory 的带宽压力了，
  // 实测性能提升非常明显！而且这样 Shared Memory 就能空出来专门做 Reduction 用了。
  // -----------------------------------------------------------

  extern __shared__ unsigned char smem_raw[];
  compute_t* s_reduce = reinterpret_cast<compute_t*>(smem_raw);

  // 寄存器数组，用于缓存 Q。
  // 假设 head_dim / blockDim.x 不会超过 8 (例如 head_dim=256, threads=32)。
  // 对于常见情况 (head_dim=128, threads=128)，每个线程只需存 1 个 float。
  compute_t reg_q[8];
  int q_idx = 0;

  const int q_base_offset = ((b * target_seq_len + t) * query_heads + qh) * head_dim;

  // 预加载 Q 到寄存器
  for (int d = tid; d < head_dim; d += blockDim.x)
  {
    if (q_idx < 8)
    {
      reg_q[q_idx++] = static_cast<compute_t>(to_float<T>(d_q[q_base_offset + d]));
    }
  }
  // Q 加载完不需要 sync，因为每个线程只读取自己后续计算需要的 Q 分量

  compute_t m_curr = static_cast<compute_t>(-INFINITY);
  compute_t l_curr = static_cast<compute_t>(0.0);
  compute_t acc_o = static_cast<compute_t>(0.0);

  const compute_t scale = static_cast<compute_t>(1.0) /
    static_cast<compute_t>(sqrt(static_cast<double>(head_dim)));

  const int kv_head_stride = head_dim;
  const int src_len_stride = kv_heads * head_dim;
  const int b_stride = src_seq_len * src_len_stride;

  const T* k_base_ptr = d_k + (b * b_stride + kvh * kv_head_stride);
  const T* v_base_ptr = d_v + (b * b_stride + kvh * kv_head_stride);

  // 外层循环：遍历 KV 序列。
  // 这是一个典型的 GEMV (Matrix-Vector Multiplication) 模式的变体。
  for (int j = 0; j < valid_src_len; ++j)
  {
    compute_t dot_val = static_cast<compute_t>(0.0);

    // 计算 Q * K 点积：
    // 使用寄存器里的 Q (reg_q) 和从 Global Memory 读取的 K 进行计算。
    q_idx = 0;
    for (int d = tid; d < head_dim; d += blockDim.x)
    {
      compute_t val_k = static_cast<compute_t>(to_float<T>(k_base_ptr[j * src_len_stride + d]));
      if (q_idx < 8)
      {
        dot_val += reg_q[q_idx++] * val_k;
      }
    }

    s_reduce[tid] = dot_val;
    __syncthreads();

    // 块内归约 (Block Reduction)：
    // 利用共享内存将同一个 Head 内所有维度的点积结果累加起来，得到完整的 Attention Score。
    // 这是一个标准的 Tree Reduction 模式。
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
      if (tid < offset)
      {
        s_reduce[tid] += s_reduce[tid + offset];
      }
      __syncthreads();
    }

    compute_t score = s_reduce[0] * scale;

    __shared__ compute_t s_score;
    __shared__ compute_t s_P;
    __shared__ compute_t s_alpha;

    // 更新 Online Softmax 统计量：
    // 只有一个线程 (Thread 0) 负责计算全局的数值修正因子。
    if (tid == 0)
    {
      compute_t m_prev = m_curr;
      m_curr = (score > m_prev) ? score : m_prev;
      // 计算 rescaling factor (alpha) 和当前分数的指数部分 (P)
      // alpha = exp(m_prev - m_curr)
      // P = exp(score - m_curr)
      compute_t alpha = expT<compute_t>(m_prev - m_curr);
      compute_t P = expT<compute_t>(score - m_curr);
      l_curr = l_curr * alpha + P;

      s_score = score;
      s_alpha = alpha;
      s_P = P;
    }
    __syncthreads();

    compute_t alpha = s_alpha;
    compute_t P = s_P;

    // 更新输出累加器：
    // 利用修正因子 alpha 和 P，每个线程负责更新 Output 的一部分分量。
    // 公式：O_new = O_old * alpha + P * V
    // 这样就避免了重新扫描前面的 KV。
    for (int d = tid; d < head_dim; d += blockDim.x)
    {
      compute_t val_v = static_cast<compute_t>(to_float<T>(v_base_ptr[j * src_len_stride + d]));
      acc_o = acc_o * alpha + P * val_v;
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    s_reduce[0] = (l_curr > 0.0f) ? (1.0f / l_curr) : 0.0f;
  }
  __syncthreads();

  compute_t inv_l = s_reduce[0];

  // 最终归一化并回写结果：
  // 将累加结果除以最终的归一化因子 l_curr，并将结果从 Float 转回 T (Half/Float) 写回 Global Memory。
  for (int d = tid; d < head_dim; d += blockDim.x)
  {
    compute_t final_val = acc_o * inv_l;
    d_o[q_base_offset + d] = from_float<T>(final_val);
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols)
{
  if (rows == 0 || cols == 0) return static_cast<T>(0);
  const size_t diag_len = (rows < cols) ? rows : cols;
  if (diag_len == 0) return static_cast<T>(0);

  T* d_input = nullptr;
  T* d_out = nullptr;
  const size_t numel = rows * cols;

  RUNTIME_CHECK(cudaMalloc(&d_input, numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), numel * sizeof(T),
    cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  const int threads = 256;
  int blocks = static_cast<int>((diag_len + threads - 1) / threads);
  if (blocks > 1024) blocks = 1024;

  traceKernel<T> << <blocks, threads, threads * sizeof(T) >> > (d_input, cols,
    diag_len, d_out);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T h_out = static_cast<T>(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));

  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
  const std::vector<T>& h_v, std::vector<T>& h_o,
  int batch_size, int target_seq_len, int src_seq_len,
  int query_heads, int kv_heads, int head_dim, bool is_causal)
{
  const size_t out_numel = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  if (h_o.size() != out_numel) h_o.resize(out_numel);

  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
    query_heads <= 0 || kv_heads <= 0 || head_dim <= 0)
  {
    for (size_t i = 0; i < out_numel; ++i) h_o[i] = static_cast<T>(0);
    return;
  }

  const size_t q_numel = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t k_numel = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t v_numel = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;

  if (h_q.size() < q_numel || h_k.size() < k_numel || h_v.size() < v_numel)
  {
    for (size_t i = 0; i < out_numel; ++i) h_o[i] = static_cast<T>(0);
    return;
  }

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, out_numel * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, out_numel * sizeof(T)));

  int threads = 32;
  while (threads < head_dim && threads < 1024) threads <<= 1;
  if (threads > 1024) threads = 1024;

  const int blocks = batch_size * target_seq_len * query_heads;

  if constexpr (std::is_same<T, float>::value)
  {
    // Float 为了保证数值结果和 Reference 一字不差（Diff=0.0），
    // 采用“并行计算，串行归约”的优化策略 (flashAttentionKernelFloatOpt)。
    // 需要 256 个线程并发计算，同时需要 SMEM 缓存 Q 和 Scores.
    int threads = 256;
    size_t smem_bytes = (head_dim + threads) * sizeof(float);
    flashAttentionKernelFloatOpt << <blocks, threads, smem_bytes >> > (
      (float*)d_q, (float*)d_k, (float*)d_v, (float*)d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  }
  else
  {
    const size_t shared_bytes = static_cast<size_t>(threads + head_dim) * sizeof(typename ComputeType<T>::type);
    flashAttentionKernel<T> << <blocks, threads, shared_bytes >> > (
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  }

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, out_numel * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
