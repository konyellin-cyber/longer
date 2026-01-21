# KV Cache 工程实现指南

## 1. TensorFlow 中的 KV Cache 实现

### **TensorFlow 的挑战**

```
TensorFlow 的特点：
✅ 优点：
   - Graph 模式编译优化
   - 静态形状推导
   - 分布式支持完善

❌ 困难：
   - 推理时序列长度动态变化（Graph 难以优化）
   - KV Cache 需要动态更新（与 Graph 的静态特性冲突）
   - Eager 执行虽然灵活但无法充分优化
```

### **方案 1：Eager Execution（推荐）**

```python
# 这是 TensorFlow 中最实用的方案

import tensorflow as tf

class KVCacheTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def call(self, x, kv_cache=None, training=False):
        # Eager 执行，动态处理 KV Cache
        
        if training:
            # 训练时，不使用 KV Cache（完整序列）
            Q, K, V = self.compute_qkv(x)
            output = self.attention(Q, K, V)
            return output, None
        
        else:
            # 推理时，使用 KV Cache
            new_token = x  # shape: (1, hidden_dim)
            Q_new = self.W_q(new_token)
            K_new = self.W_k(new_token)
            V_new = self.W_v(new_token)
            
            if kv_cache is None:
                # 第一个 token
                K_full = K_new
                V_full = V_new
                kv_cache = (K_new, V_new)
            else:
                # 后续 token
                K_cache, V_cache = kv_cache
                K_full = tf.concat([K_cache, K_new], axis=0)  # (seq_len, dim)
                V_full = tf.concat([V_cache, V_new], axis=0)
                kv_cache = (K_full, V_full)
            
            output = self.attention(Q_new, K_full, V_full)
            return output, kv_cache

# 推理循环
@tf.function(jit_compile=True)  # 单步优化
def inference_step(model, input_token, kv_caches):
    outputs = []
    new_kv_caches = []
    
    x = input_token
    for i, layer in enumerate(model.layers):
        x, new_cache = layer(x, kv_cache=kv_caches[i])
        new_kv_caches.append(new_cache)
        outputs.append(x)
    
    return x, new_kv_caches

# 生成循环
def generate(model, prompt, max_len):
    kv_caches = [None] * len(model.layers)
    tokens = prompt
    
    for step in range(max_len):
        last_token = tf.expand_dims(tokens[-1:], 0)
        
        # 执行一步推理
        logits, kv_caches = inference_step(
            model, last_token, kv_caches
        )
        
        next_token = tf.argmax(logits, axis=-1)
        tokens = tf.concat([tokens, next_token], axis=0)
    
    return tokens
```

### **方案 2：使用 tf.RaggedTensor（动态形状）**

```python
# 对于需要更好图优化的场景

import tensorflow as tf

class DynamicKVCache:
    def __init__(self, max_seq_len, hidden_dim, dtype=tf.float32):
        # 使用 TensorVariable 存储 KV Cache
        self.k_cache = tf.Variable(
            tf.zeros((max_seq_len, hidden_dim), dtype=dtype),
            trainable=False,
            name='k_cache'
        )
        self.v_cache = tf.Variable(
            tf.zeros((max_seq_len, hidden_dim), dtype=dtype),
            trainable=False,
            name='v_cache'
        )
        self.length = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def update(self, k_new, v_new):
        # 原子操作：更新缓存并增加长度
        idx = self.length
        new_len = idx + tf.shape(k_new)[0]
        
        # 使用 assign 操作
        self.k_cache[idx:new_len].assign(k_new)
        self.v_cache[idx:new_len].assign(v_new)
        self.length.assign(new_len)
    
    def get_full(self):
        length = self.length
        return (
            self.k_cache[:length],
            self.v_cache[:length]
        )
    
    def reset(self):
        self.length.assign(0)

# 使用示例
@tf.function
def attention_with_cache(Q, k_cache, v_cache):
    K_full, V_full = k_cache.get_full()
    
    # 计算 Attention
    scores = tf.matmul(Q, K_full, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(tf.shape(K_full)[-1], tf.float32))
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V_full)
    
    return output
```

### **方案 3：自定义 Op（高性能）**

```python
# 需要编写 CUDA/C++ 代码

import tensorflow as tf

# 自定义 op，直接在 GPU 上操作
@tf.function
def fused_attention_with_kv_cache(
    Q, K_cache, V_cache, K_new, V_new
):
    """
    融合操作：
    1. 拼接 K, V
    2. 计算 Attention
    3. 更新缓存
    
    完全在 GPU 上执行，无中间数据交换
    """
    # 调用自定义 CUDA op
    output, new_k_cache, new_v_cache = \
        tf.raw_ops.FusedAttentionWithKVCache(
            Q=Q,
            K_cache=K_cache,
            V_cache=V_cache,
            K_new=K_new,
            V_new=V_new,
            # 其他参数...
        )
    
    return output, new_k_cache, new_v_cache
```

## 2. TensorFlow Graph 修改策略

### **传统 Graph（无 KV Cache）**

```
Graph 结构：

Input ──→ Embedding ──→ Layer_0 ──→ Layer_1 ──→ ... ──→ Output
                           ↓          ↓
                        Attention  Attention
                           ↓          ↓
                        完整序列   完整序列
```

### **改造后的 Graph（KV Cache）**

#### **方案 A：动态 Graph（不修改静态图结构）**

```python
# 使用 tf.cond 或 tf.while_loop 动态处理

def build_inference_graph():
    @tf.function
    def step_fn(token_idx, kv_caches):
        # 在 Graph 内部动态执行
        
        # 对当前 token 编码
        x = embed(token_idx)
        
        new_kv_caches = []
        for i, layer in enumerate(layers):
            if kv_caches[i] is None:
                # 首次，计算完整 Attention
                x = layer.full_attention(x)
                new_kv_cache = (K, V)
            else:
                # 后续，使用缓存
                x = layer.incremental_attention(x, kv_caches[i])
                new_kv_cache = update_cache(kv_caches[i], K_new, V_new)
            
            new_kv_caches.append(new_kv_cache)
        
        return x, new_kv_caches
    
    return step_fn

# 调用
step_fn = build_inference_graph()
kv_caches = [None] * num_layers

for i in range(seq_len):
    output, kv_caches = step_fn(tokens[i], kv_caches)
```

#### **方案 B：显式分支 Graph（修改图结构）**

```python
# 为训练和推理分别构建不同的 Graph

class DualGraphModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_train = [...]  # 训练用：处理完整序列
        self.layers_infer = [...]  # 推理用：增量计算
    
    @tf.function(input_signature=[...])
    def call_train(self, input_ids):
        # 训练 Graph：标准 Transformer
        x = embedding(input_ids)
        for layer in self.layers_train:
            x = layer(x)
        return x
    
    @tf.function
    def call_infer(self, token_id, kv_caches):
        # 推理 Graph：使用 KV Cache
        x = embedding(token_id)
        new_kv_caches = []
        
        for i, layer in enumerate(self.layers_infer):
            if kv_caches[i] is None:
                x = layer.forward(x)
                new_cache = None
            else:
                x = layer.forward_incremental(x, kv_caches[i])
                new_cache = layer.updated_cache
            
            new_kv_caches.append(new_cache)
        
        return x, new_kv_caches
    
    def call(self, input_ids, training=True):
        if training:
            return self.call_train(input_ids)
        else:
            # 需要外部循环调用 call_infer
            return None  # 在 Python 循环中调用

# 使用
model = DualGraphModel()

# 训练时
loss = model(input_ids, training=True)

# 推理时（Python 循环）
kv_caches = [None] * num_layers
for token in sequence:
    output, kv_caches = model.call_infer(token, kv_caches)
```

#### **方案 C：使用 tf.while_loop（最优化）**

```python
@tf.function
def generate_with_while_loop(prompt_ids, max_steps):
    # 这样能充分利用 Graph 优化
    
    def body_fn(step, token_id, kv_caches, output_ids):
        # 单步推理
        x = embedding(token_id)
        new_kv_caches = []
        
        for i, layer in enumerate(layers):
            x, new_cache = layer.incremental_forward(x, kv_caches[i])
            new_kv_caches.append(new_cache)
        
        next_token_id = tf.argmax(x, axis=-1)
        output_ids = tf.concat([output_ids, [next_token_id]], axis=0)
        
        return step + 1, next_token_id, new_kv_caches, output_ids
    
    def cond_fn(step, *args):
        return step < max_steps
    
    kv_caches = [None] * num_layers
    initial_token = prompt_ids[-1]
    
    _, _, _, final_ids = tf.while_loop(
        cond_fn,
        body_fn,
        loop_vars=[
            tf.constant(0),
            initial_token,
            kv_caches,
            prompt_ids
        ]
    )
    
    return final_ids
```

## 3. PyTorch vs TensorFlow 对比

### **PyTorch（实际更易用）**

```python
# PyTorch 的 Eager 执行天然适合 KV Cache

class TransformerLayer(nn.Module):
    def forward(self, x, kv_cache=None):
        # 动态处理，无需 Graph 修改
        Q, K, V = self.compute_qkv(x)
        
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=0)
            V = torch.cat([V_cache, V], dim=0)
        
        output = self.attention(Q, K, V)
        
        return output, (K, V)

# 推理循环非常自然
def generate(model, prompt):
    kv_caches = [None] * len(model.layers)
    
    for _ in range(max_len):
        x = prompt[-1:]
        for i, layer in enumerate(model.layers):
            x, kv_caches[i] = layer(x, kv_caches[i])
        prompt = torch.cat([prompt, x], dim=0)
    
    return prompt
```

### **对比表**

| 方面 | PyTorch | TensorFlow |
|------|---------|----------|
| **KV Cache 易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Graph 修改需求** | 无 | 需要（Eager 除外） |
| **推理循环** | 代码简洁 | 相对复杂 |
| **性能** | 最优 | 可接受 |
| **学习成本** | 低 | 高 |
| **生产部署** | vLLM ✅ | 可行但较复杂 |

## 4. 实现建议（LONGER 项目）

### **分阶段建议**

```
1️⃣ 原型开发 → PyTorch
   ├─ 快速迭代 KV Cache 优化逻辑
   ├─ 调试和验证 User/Item 粒度复用
   └─ 无需修改框架代码

2️⃣ 学术发表 → PyTorch
   └─ 开源社区主流，易获得关注

3️⃣ 生产部署
   
   选项 A：PyTorch + vLLM（推荐）
   ├─ 原生 KV Cache 管理
   ├─ 物理块映射
   ├─ PagedAttention 优化
   └─ 业界标准
   
   选项 B：TensorFlow + Eager Execution
   ├─ 无需修改 Graph
   ├─ 代码逻辑清晰
   ├─ 性能 -10-20%（可接受）
   └─ 适合已有 TF 基础设施的团队
   
   选项 C：自定义 CUDA Op
   ├─ 最高性能
   ├─ 维护成本高
   └─ 仅在必要时考虑
```

### **核心工程问题清单**

```
[ ] KV Cache 的初始大小如何设定？
    → 动态扩展还是固定上限？

[ ] 多请求下 KV Cache 如何共享？
    → 物理块映射还是简单复制？

[ ] 显存不足时的 fallback 策略？
    → CPU 内存 / SSD 预取？

[ ] 如何处理 batch 中序列长度不一致？
    → Padding 还是 ragged tensor？

[ ] KV Cache 预热和预加载？
    → 对性能的影响有多大？

[ ] 与量化（INT8）的结合？
    → KV Cache 量化会影响精度吗？

[ ] 分布式推理中的 KV Cache 同步？
    → 张量并行 / 流水线并行下的策略？
```

## 5. 性能优化检查清单

```
推理优化的关键指标：

🚀 延迟优化
  [ ] 单步推理时间 < 100ms（推荐）
  [ ] KV Cache 访问是否是瓶颈？
  [ ] GPU 利用率是否充分（>80%）？

💾 显存优化
  [ ] KV Cache 占用 vs 模型参数比
  [ ] 峰值显存是否在设备限制内？
  [ ] 是否出现频繁的 OOM？

🔄 吞吐优化
  [ ] 批处理大小的最优值
  [ ] 并发请求数对吞吐的影响
  [ ] 序列长度增长时的吞吐下降曲线

🎯 准确率
  [ ] KV Cache 优化是否影响输出？
  [ ] 量化后的 KV Cache 精度
  [ ] 与 full precision 的 diff
```

## 参考实现

- **vLLM**：https://github.com/lm-sys/vllm
  - 物理块管理
  - PagedAttention
  - KV Cache 最佳实践

- **HuggingFace Transformers**：https://github.com/huggingface/transformers
  - 简单 KV Cache 实现
  - 多框架支持

- **TensorFlow Text**：
  - 预处理优化
  - TF native KV Cache 支持（新版本）
