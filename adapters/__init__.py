"""框架适配层：把本地张量翻译成全局坐标。

已实现：``dtensor`` —— FSDP2 / DTensor 的 ``placements`` → :class:`ShardSource`。
计划中：``megatron``（PP 层号 / EP 专家号 / 融合参数）、``stateful``（优化器状态 → FQN）。
"""
