# 速度控制指令
# VLM 作为小车大脑，直接输出线速度和角速度

# 时间戳
builtin_interfaces/Time stamp

# 线速度 (米/秒)
# 正值前进，负值后退
# 范围: -1.0 ~ 1.0
float32 linear_velocity

# 角速度 (弧度/秒)
# 正值左转，负值右转
# 范围: -1.0 ~ 1.0
float32 angular_velocity

# 置信度 (0.0 ~ 1.0)
float32 confidence

# 决策描述 (可选)
string description
