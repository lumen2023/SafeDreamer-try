import wandb
import pandas as pd

api = wandb.Api()

# 列出项目下所有的 runs
runs = api.runs("lyz_1023-southeastern-university/safedreamer-highway-go")
for run in runs:
    print(run.name, run.id)  # 检查正确的 run_id


# 初始化 W&B 运行
run = wandb.init()

api = wandb.Api()

# 获取运行数据
run_path = "lyz_1023-southeastern-university/safedreamer-highway-go/20250306-193503_osrp_vector_highway-fast_New-gym_safe-highway-fast-v0_0"
run = api.run(run_path)

# 获取历史数据
history = run.history(keys=None, samples=10000)

# 将历史数据转换为 DataFrame
df = pd.DataFrame(history)

# 保存为 CSV 文件
df.to_csv('wandb_history_data.csv', index=False)

print("数据已成功导出到 wandb_history_data.csv")

