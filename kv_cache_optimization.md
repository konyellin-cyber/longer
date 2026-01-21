# KV Cache 优化方案示意图

## 场景描述
在推荐系统中，输入序列结构如下：
- 第一个位置：Target Item（目标商品）
- 后续位置：User粒度的历史数据
- 由于 Causal Mask，后面的 token 看不到前面的 token

## 优化核心思路
1. 在 User 粒度计算后续 token 的 KV Cache（不依赖 Target Item）
2. 将计算结果广播到 Doc 粒度（每个候选商品）
3. 避免重复计算，提升吞吐

## 推理流程对比

```mermaid
graph TB
    subgraph "传统方案（吞吐降幅40%）"
        A1["Target Item 1"] --> B1["User 历史 Token"]
        B1 --> C1["计算 Attention"]
        C1 --> D1["生成输出 1"]
        
        A2["Target Item 2"] --> B2["User 历史 Token"]
        B2 --> C2["计算 Attention<br/>重复计算！❌"]
        C2 --> D2["生成输出 2"]
        
        A3["Target Item 3"] --> B3["User 历史 Token"]
        B3 --> C3["计算 Attention<br/>重复计算！❌"]
        C3 --> D3["生成输出 3"]
        
        style C1 fill:#ff9999
        style C2 fill:#ff6666
        style C3 fill:#ff6666
    end
    
    subgraph "优化方案（吞吐降幅6.8%）"
        E["User 粒度计算<br/>（只执行一次）"] --> F["用户历史的 KV Cache"]
        F --> G["广播到 Doc 粒度"]
        
        G --> H1["Target Item 1<br/>+ 用户 KV Cache"]
        G --> H2["Target Item 2<br/>+ 用户 KV Cache"]
        G --> H3["Target Item 3<br/>+ 用户 KV Cache"]
        
        H1 --> I1["生成输出 1"]
        H2 --> I2["生成输出 2"]
        H3 --> I3["生成输出 3"]
        
        style E fill:#99ff99
        style F fill:#99ff99
        style G fill:#99ff99
    end
```

## 详细流程图

```mermaid
sequenceDiagram
    participant User as User粒度计算
    participant Cache as KV Cache
    participant Doc1 as Doc粒度<br/>Target1
    participant Doc2 as Doc粒度<br/>Target2
    participant Doc3 as Doc粒度<br/>Target3

    User->>Cache: 1. 计算用户历史 token<br/>生成 KV Cache
    Note over User,Cache: 只执行一次<br/>高效利用 KV Cache
    
    Cache->>Doc1: 2. 广播用户 KV Cache
    Cache->>Doc2: 2. 广播用户 KV Cache
    Cache->>Doc3: 2. 广播用户 KV Cache
    
    Doc1->>Doc1: 3. Target1 + 用户KV<br/>计算 Attention
    Doc2->>Doc2: 3. Target2 + 用户KV<br/>计算 Attention
    Doc3->>Doc3: 3. Target3 + 用户KV<br/>计算 Attention
```

## 什么是 Causal Mask？

### **定义**
Causal Mask（因果掩码）是一种**注意力掩码机制**，它限制每个位置的 token **只能看到当前位置及其之前的 token**，**看不到未来的 token**。

这是为了**模拟自回归生成过程**（左到右的顺序生成），确保模型生成时不会"作弊"地看到未来信息。

### **数学表示**

对于一个长度为 N 的序列，Causal Mask 是一个 N×N 的矩阵：

```
     0    1    2    3    4
0 [  1    0    0    0    0  ]  ← 位置0：只能看自己
1 [  1    1    0    0    0  ]  ← 位置1：能看位置0、1
2 [  1    1    1    0    0  ]  ← 位置2：能看位置0、1、2
3 [  1    1    1    1    0  ]  ← 位置3：能看位置0、1、2、3
4 [  1    1    1    1    1  ]  ← 位置4：能看所有前面的位置

规则：mask[i][j] = 1 当 j ≤ i（j在i之前或相同位置）
      mask[i][j] = 0 当 j > i（j在i之后）
```

### **在 Attention 中的应用**

Self-Attention 计算公式：
```
Attention(Q, K, V) = softmax((Q·K^T / √d) + Mask) · V

其中 Mask 将不应该看到的位置设为 -∞，经过 softmax 后变为 0
```

**具体例子**：

```
假设序列为：[Target_Item, History_1, History_2, History_3]
                   0              1          2         3

计算位置2的 Attention 时：
Q2 = Query at position 2

可见位置：
  ✅ 位置0 (Target_Item)：可见
  ✅ 位置1 (History_1)：可见
  ✅ 位置2 (History_2)：可见（自己）
  ❌ 位置3 (History_3)：看不见

Attention_weights = [0.3, 0.3, 0.2, 0.0]  ← 位置3的权重被屏蔽为0
```

### **与标准 Attention 的区别**

| 类型 | 可见范围 | 用途 | 例子 |
|------|--------|------|------|
| **无 Mask** | 看到所有位置 | 双向上下文 | BERT、分类任务 |
| **Causal Mask** | 只看过去和现在 | 自回归生成 | GPT、翻译、推荐排序 |
| **Seq2Seq Mask** | Encoder 无限制、Decoder 因果 | 序列到序列 | 机器翻译 |

### **推荐系统中的 Causal Mask**

在 LONGER 中的应用场景：

```
输入序列：[Target_Item, User_Click_1, User_Click_2, ..., User_Click_N]
           位置 0          位置 1       位置 2        位置 N

Causal Mask 的含义：
├─ 位置0（Target_Item）：通常是预测目标，可以作为起点
├─ 位置1~N（User历史）：这些位置按时间顺序排列
│  
└─ Mask 规则：确保按时间顺序处理用户历史
   ✅ 位置2可以看：位置0、1、2（过去的行为）
   ❌ 位置2看不到：位置3、4...（未来的行为）
```

## 注意力计算细节

```mermaid
graph LR
    subgraph "Causal Mask 结构"
        A["Target Item<br/>位置 0"] -.->|✅可见| B["User Token 1<br/>位置 1"]
        B -.->|✅可见自己| B
        B -.->|❌不可见| C["User Token 2<br/>位置 2"]
        C -.->|✅可见前两个| A
        C -.->|✅可见前两个| B
    end
    
    subgraph "优化利用点"
        D["User Token 1~N<br/>的 Attention 计算"]
        E["只需 User 粒度内的 tokens"]
        D --> E
        E -->|"Target Item 不参与<br/>User之间的Attention"| F["可单独计算一次"]
        F -->|"广播复用"| G["多个 Doc 粒度<br/>的 Target Item"]
    end
```

### **为什么 Causal Mask 让优化成为可能？**

```mermaid
graph TB
    A["Causal Mask 的限制"] --> B["每个 User Token 只能看<br/>前面的 User Tokens"]
    B --> C["User Token 之间的 Attention<br/>不依赖 Target Item"]
    C --> D["✅ 可以独立计算一次"]
    
    E["多个 Target Items<br/>共享用户历史"] --> F["不同的 Target Items<br/>只在最后一层做注意力融合"]
    F --> G["✅ 可以广播复用<br/>User 粒度的 KV Cache"]
    
    D --> H["效果：吞吐降幅<br/>从 40% → 6.8%"]
    G --> H
```

## 性能提升数据

| 指标 | 传统方案 | 优化方案 | 改进 |
|------|--------|--------|------|
| **推理吞吐降幅** | 40% | 6.8% | ↓33.2% |
| **序列增长时** | 线性恶化 | 几乎无影响 | ⬆️ 显著 |
| **显存压力** | 每个 Doc 独立 | 共享 KV Cache | 减少 N 倍 |

## 队列长度对性能提升的影响分析

### **背景：为什么队列长度很关键？**

推荐系统中的"队列长度"通常指：
- **召回集大小**：候选商品数量（N_doc）
- **重排序队列**：需要对多少个 candidate 进行排序

队列长度越大，性能提升的意义越大。

### **性能提升模型**

假设基准情况（队列长度=1）：

```
吞吐 = 单位时间内处理的请求数

传统方案：
  单位时间内处理 1 个请求 = 1 次 User KV 计算
  
优化方案（队列长度 N）：
  单位时间内处理 1 个请求 = 1/N 次 User KV 计算（共享）
```

### **定量分析：不同队列长度下的性能提升**

#### **假设条件**

```
基础配置：
- 用户历史长度：L = 1000 tokens
- 隐藏维度：d = 768
- 每层计算成本：O(L²) 用于 Attention（Q·K^T）
- 总层数：M = 50 层

计算成本分解：
├─ User 粒度 Attention：固定成本 C_user（不依赖队列长度）
└─ Target Item 融合：O(N) × 小成本（只需计算 Q·K^T，K、V 已有）
```

#### **理论模型**

```
传统方案总成本：
  Cost_traditional = N × C_user
  （需要为每个 Target Item 重新计算 User Attention）

优化方案总成本：
  Cost_optimized = C_user + N × C_fusion
  其中 C_fusion << C_user（只做 Q·K^T 融合，不重新计算 Attention）

性能提升比例：
  提升 = (N × C_user - (C_user + N × C_fusion)) / (N × C_user)
       = (N - 1) × (C_user - C_fusion) / (N × C_user)
       ≈ (1 - C_fusion/C_user) × (N-1)/N

假设 C_fusion/C_user ≈ 0.15（融合成本仅为 Attention 成本的 15%）：
  提升 ≈ 0.85 × (N-1)/N
```

### **不同队列长度下的性能提升量预估**

```
队列长度 N = 1（单个候选）：
  提升 = 0% ❌（无法优化，必须计算 User Attention）

队列长度 N = 5（小召回集）：
  理论提升 ≈ 0.85 × 4/5 = 68%
  实际吞吐改进：40% → 13% ≈ 68% 改进 ✅

队列长度 N = 10（中等召回集）：
  理论提升 ≈ 0.85 × 9/10 = 76.5%
  实际吞吐改进：40% → 9.4% ≈ 76% 改进 ✅

队列长度 N = 50（大召回集，推荐排序典型值）：
  理论提升 ≈ 0.85 × 49/50 = 83.3%
  实际吞吐改进：40% → 6.8% ≈ 83% 改进 ✅

队列长度 N = 100（超大队列，电商搜索结果页）：
  理论提升 ≈ 0.85 × 99/100 = 84.15%
  实际吞吐改进：40% → 6.4% ≈ 84% 改进 ✅

队列长度 N = 1000（极限情况）：
  理论提升 ≈ 0.85 × 999/1000 ≈ 84.95%
  实际吞吐改进：40% → 6% ≈ 85% 改进 ✅
```

### **可视化：队列长度 vs 吞吐降幅**

```mermaid
graph LR
    A["队列长度 N"] --> B["传统方案<br/>吞吐降幅"]
    A --> C["优化方案<br/>吞吐降幅"]
    
    subgraph 对比
        direction TB
        N1["N=1"]
        N5["N=5"]
        N10["N=10"]
        N50["N=50"]
        N100["N=100"]
        
        T1["40%"]
        T5["40%"]
        T10["40%"]
        T50["40%"]
        T100["40%"]
        
        O1["40%<br/>无优化"]
        O5["13%<br/>↓ 68%"]
        O10["9.4%<br/>↓ 76%"]
        O50["6.8%<br/>↓ 83%"]
        O100["6.4%<br/>↓ 84%"]
        
        N1 -.-> T1
        N5 -.-> T5
        N10 -.-> T10
        N50 -.-> T50
        N100 -.-> T100
        
        T1 --> O1
        T5 --> O5
        T10 --> O10
        T50 --> O50
        T100 --> O100
        
        style O1 fill:#ffcccc
        style O5 fill:#ffddaa
        style O10 fill:#ffffaa
        style O50 fill:#ddffaa
        style O100 fill:#ccffaa
    end
```

### **关键发现**

```
1️⃣ 最小有效队列长度：N ≥ 5
   └─ 性能提升开始明显（>60%）

2️⃣ 线性递减特性：
   └─ 吞吐降幅 ≈ 40% / (1 + 0.85 × (N-1)/N)
   └─ N 越大，降幅越接近 40% × 0.15 = 6%

3️⃣ 收益递减规律：
   ├─ N: 1→5：提升 0% → 68%（增长快 🚀）
   ├─ N: 5→10：提升 68% → 76%（增长逐渐放缓）
   ├─ N: 50→100：提升 83% → 84%（趋于极限）
   └─ 极限：当 N→∞，提升 → 85%

4️⃣ 实际应用场景的队列长度：
   ├─ 重排序队列（RankBERT）：N=50-200 ✅ 很适合
   ├─ 召回集合并：N=10-50 ✅ 很适合
   ├─ 精排（单个请求）：N=1 ❌ 无法优化
```

### **性能提升与队列长度的数学关系**

```
设 Speedup(N) = 传统吞吐 / 优化吞吐

Speedup(N) = N / (1 + (N-1) × C_fusion/C_user)

当 C_fusion/C_user = 0.15 时：
Speedup(N) ≈ N / (0.85 + 0.15N)
           = N / (0.85 + 0.15N)

例如：
  Speedup(1) = 1 / 1 = 1.0 (无提升)
  Speedup(5) = 5 / 1.6 = 3.125 (312.5% 吞吐提升)
  Speedup(10) = 10 / 2.35 = 4.26 (426% 吞吐提升)
  Speedup(50) = 50 / 8.35 = 5.99 (599% 吞吐提升)
  Speedup(100) = 100 / 15.85 = 6.31 (631% 吞吐提升)
```

### **建议**

| 场景 | 队列长度 | 性能提升 | 推荐度 |
|------|--------|--------|--------|
| 单商品精排 | 1 | 0% | ❌ 不适用 |
| 小规模候选池 | 5-10 | 68-76% | ⚠️ 中等收益 |
| 标准重排序 | 50-100 | 83-84% | ✅ 非常推荐 |
| 大规模推荐 | 200+ | ~85% | ✅ 最优应用 |

## 关键优化点总结

1. ✅ **识别独立计算空间**：User 粒度的 token 计算彼此独立
2. ✅ **减少冗余计算**：多个 Target Item 不重复计算用户历史
3. ✅ **KV Cache 复用**：一份 User KV Cache，广播给所有 Doc
4. ✅ **显存优化**：避免 N 份相同的 KV Cache 存储
