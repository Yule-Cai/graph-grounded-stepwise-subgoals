import rclpy
from stable_baselines3 import PPO
from llm_rl_nav.envs.turtlebot_env import TurtleBotEnv
from llm_rl_nav.utils import latest_successful_ppo_path
import time
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()

def main():
    rclpy.init()
    
    # 路径检查
    model_path = latest_successful_ppo_path(project_root())
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}")
        return

    env = TurtleBotEnv()
    
    print(f"🧠 加载模型: {model_path}")
    model = PPO.load(str(model_path))
    
    print("🤖 开始测试 (按 Ctrl+C 退出)")
    obs = env.reset()
    
    try:
        for i in range(10000):
            # deterministic=True 让 AI 不再随机探索，而是执行最优策略
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            # 打印关键信息
            dist_to_goal = obs[60] # 60 laser bins + distance + heading error
            print(f"Step {i}: 动作={action} | 剩余距离={dist_to_goal:.2f}m")
            
            if done:
                print("--- 回合结束 (撞墙或到达) ---")
                obs = env.reset()
                time.sleep(1.0) # 休息一下
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("测试停止")
    
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
