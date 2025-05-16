import os
import subprocess
import time

print("\\n🚀 [1/3] 正在启动模型训练...")
start_time = time.time()
subprocess.run(["python", "train_bfn.py"])
print(f"✅ 模型训练完成！耗时 {time.time() - start_time:.1f} 秒\\n")

print("🎯 [2/3] 采样结构中...")
subprocess.run(["python", "sample_fiber.py"])
print("✅ 已生成结构样本图\\n")

print("📊 [3/3] 批量评估结构...")
subprocess.run(["python", "scripts/batch_sample_eval.py"])
print("✅ 批量结构评估结果已保存为 CSV（batch_eval_results.csv）\\n")

print("🏁 完整流程结束！可查看结构图像、评估结果和 WandB 可视化结果。")