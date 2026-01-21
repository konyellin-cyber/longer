# KV Cache 原理详解

## 1. KV Cache 的存储和读取流程

### **传统方式（无 KV Cache）**

```
每次推理（生成新 token）：

输入序列：[token_0, token_1, token_2, ..., token_n]
           ↓
         embedding 层
           ↓
         Transformer 层
           ↓
    计算所有位置的 Q, K, V
           ↓
      做完整 Attention
           ↓
          输出
           
🔴 问题：即使 token_0 到 token_n 没变，也要全部重新计算 Q, K, V
```

### **KV Cache 方式**

```
第一次推理（生成 token_n+1）：

输入：[token_0, token_1, ..., token_n]
      ↓
计算 Q, K, V
      ↓
保存 K, V 到 KV Cache（外存储或显存）
      ↓
生成 token_n+1

第二次推理（生成 token_n+2）：

输入：只有 token_n+1（新 token）
      ↓
计算新 Q, K, V
      ↓
从 KV Cache 读取历史 K, V
      ↓
拼接：K_full = [K_cache, K_new]
      V_full = [V_cache, V_new]
      ↓
做 Attention(Q_new, K_full, V_full)
      ↓
更新 KV Cache，生成 token_n+2
```

### **KV Cache 的存储位置**

根据序列长度和硬件，有三种存储方式：

```
1️⃣ 显存（GPU Memory）- 推荐
   ├─ 位置：GPU 显存（与模型参数同位置）
   ├─ 优点：访问最快，完全利用 GPU 计算能力
   ├─ 缺点：显存有限，超长序列容易溢出
   └─ 使用场景：短到中等序列（<100k tokens）
   
   显存使用量计算：
   KV_memory = 2 × seq_len × hidden_dim × num_layers × batch_size × dtype_size
   
   示例（1000 tokens）：
   = 2 × 1000 × 768 × 50 × 1 × 2bytes
   ≈ 150MB per request

2️⃣ CPU 内存 - 中等
   ├─ 位置：主机 RAM
   ├─ 优点：容量大，可存储超长序列
   ├─ 缺点：CPU-GPU 数据传输开销大
   └─ 使用场景：长序列，但需要频繁访问
   
   需要在每次 Attention 时：
   CPU → GPU 传输（PCIe 3.0: ~16GB/s）

3️⃣ NVMe SSD - 大规模
   ├─ 位置：固态硬盘
   ├─ 优点：容量最大，可处理极长序列（>1M tokens）
   ├─ 缺点：访问延迟最高（ms 级别）
   └─ 使用场景：离线推理，超长序列
   
   需要预取和异步 I/O 来隐藏延迟
```

## 2. 大模型中的 KV Cache 实现

### **vLLM 的实现（业界标准）**

vLLM 是 GPU 推理框架中 KV Cache 管理最优的实现：

```python
# 核心思想：物理块管理（Physical Block 和 Logical Block）

# 1. 物理块分配
class KVCacheManager:
    def __init__(self, num_gpu_blocks, block_size):
        self.gpu_blocks = GPUBlockAllocator(num_gpu_blocks, block_size)
        # block_size 通常是 16 tokens
        # num_gpu_blocks 根据显存自动计算
    
    def allocate(self, seq_len):
        # 分配足够的块
        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = self.gpu_blocks.allocate(num_blocks)
        return blocks

# 2. 物理块映射
# 多个请求可以共享同一块物理块（KV Cache 共享）
request1_kv = [block_1, block_2, block_3]  # 指向物理块
request2_kv = [block_1, block_2, block_4]  # 前两块共享！

# 3. 访问流程
for step in range(num_steps):
    # 获取逻辑地址映射
    logical_blocks = request.kv_cache_blocks
    
    # 转换到物理块地址
    physical_blocks = mapping_table[logical_blocks]
    
    # GPU kernel 直接操作物理块
    attention_kernel(Q, physical_blocks, output)
    
    # 生成新 token 后，分配新块
    new_block = allocate_block()
    request.kv_cache_blocks.append(new_block)
```

### **HuggingFace 的实现**

```python
# 更简单的实现方式

class SimpleKVCache:
    def __init__(self, max_seq_len, hidden_dim):
        # 预先分配固定大小的张量（显存）
        self.key_cache = torch.zeros(
            (num_layers, max_seq_len, hidden_dim),
            device='cuda'
        )
        self.value_cache = torch.zeros(
            (num_layers, max_seq_len, hidden_dim),
            device='cuda'
        )
        self.cur_len = 0  # 当前填充到的位置
    
    def update(self, layer_idx, new_k, new_v):
        # 把新的 K, V 追加到缓存
        self.key_cache[layer_idx, self.cur_len:self.cur_len+new_k.shape[0]] = new_k
        self.value_cache[layer_idx, self.cur_len:self.cur_len+new_v.shape[0]] = new_v
        self.cur_len += new_k.shape[0]
    
    def get(self, layer_idx):
        # 返回当前有效的 K, V
        return (
            self.key_cache[layer_idx, :self.cur_len],
            self.value_cache[layer_idx, :self.cur_len]
        )
```

### **数据流向**

```
推理阶段（每步）：

外存/CPU ──────────┐
                   │
                   ↓
              预取（可选）
                   │
                   ↓
            ┌─────────────┐
            │  显存缓冲区  │  ← 新 token 对应的 K, V
            └─────────────┘
                   │
                   ↓
         ┌──────────────────┐
         │   GPU 显存       │
         ├──────────────────┤
         │  KV Cache 区域   │  ← 历史 K, V（核心！）
         │  (150-300MB)     │
         ├──────────────────┤
         │  模型参数        │
         │  (70B: 140GB)    │
         └──────────────────┘
                   │
                   ↓
            Attention 计算
                   │
                   ↓
              生成下一 token
                   │
                   └──→ 保存到 KV Cache
```

## 3. KV Cache 显存占用分析

### **公式推导**

```
KV Cache 显存 = 每个 KV 对占用

基础计算：
  每个位置的 K 或 V = hidden_dim × dtype_size
  
  1 个 token 的 1 层 KV = 2 × hidden_dim × dtype_size
  
  完整 KV Cache = 2 × seq_len × hidden_dim × num_layers × dtype_size

实际例子（LLaMA 7B 模型）：
  - hidden_dim = 4096
  - num_layers = 32
  - dtype = float16 (2 bytes)
  - seq_len = 2048
  
  KV_cache = 2 × 2048 × 4096 × 32 × 2
           ≈ 1GB per request
```

### **多请求场景下的节省**

```
N 个请求共享同一 User History 的 KV Cache：

传统方式：
  总显存 = N × (KV_user + KV_item)
  
优化方式：
  总显存 = KV_user + N × KV_item
  
节省量（LONGER 的关键）：
  节省 = (N-1) × KV_user
  
示例（50 个候选 items）：
  KV_user ≈ 100MB
  节省 = 49 × 100MB ≈ 4.9GB ✅
```

## 4. 关键指标

### **性能指标对比**

| 指标 | 无 KV Cache | 有 KV Cache | 改进 |
|------|-----------|----------|------|
| **内存占用** | 很大 ❌ | 小 ✅ | -60% |
| **计算量** | O(n²) | O(n) | 线性 ✅ |
| **推理速度** | 慢 | 快 ✅ | 5-10x |
| **延迟** | ~1s | ~10ms | 100x ✅ |

### **推荐最佳实践**

```
✅ 使用 KV Cache 的场景：
  ├─ 文本生成（LLM）
  ├─ 翻译
  ├─ 推荐排序（LONGER）
  └─ 实时对话

❌ 不需要 KV Cache 的场景：
  ├─ 分类任务
  ├─ 一次性推理
  └─ 离线批处理
```

## 参考资源

详细的工程实现指南见：[kvcache_engineering.md](./kvcache_engineering.md)

相关技术文章：
- vLLM：https://github.com/lm-sys/vllm
- FlashAttention：https://github.com/HazyResearch/flash-attention
- HuggingFace Transformers：https://huggingface.co/docs/transformers/
