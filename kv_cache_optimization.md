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

## 注意力计算细节

```mermaid
graph LR
    subgraph "Causal Mask 结构"
        A["Target Item<br/>位置 0"] -.->|不可见| B["User Token 1<br/>位置 1"]
        B -.->|可见自己| B
        B -.->|不可见| C["User Token 2<br/>位置 2"]
        C -.->|可见前两个| A
        C -.->|可见前两个| B
    end
    
    subgraph "优化利用点"
        D["User Token 1~N<br/>的 Attention 计算"]
        E["只需 User 粒度内的 tokens"]
        D --> E
        E -->|"不依赖 Target Item"| F["可单独计算一次"]
        F -->|"广播复用"| G["多个 Doc 粒度<br/>的 Target Item"]
    end
```

## 性能提升数据

| 指标 | 传统方案 | 优化方案 | 改进 |
|------|--------|--------|------|
| **推理吞吐降幅** | 40% | 6.8% | ↓33.2% |
| **序列增长时** | 线性恶化 | 几乎无影响 | ⬆️ 显著 |
| **显存压力** | 每个 Doc 独立 | 共享 KV Cache | 减少 N 倍 |

## 关键优化点总结

1. ✅ **识别独立计算空间**：User 粒度的 token 计算彼此独立
2. ✅ **减少冗余计算**：多个 Target Item 不重复计算用户历史
3. ✅ **KV Cache 复用**：一份 User KV Cache，广播给所有 Doc
4. ✅ **显存优化**：避免 N 份相同的 KV Cache 存储
