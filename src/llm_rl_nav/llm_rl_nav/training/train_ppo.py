import rclpy
from stable_baselines3 import PPO
import os
from pathlib import Path

from llm_rl_nav.envs.turtlebot_env import TurtleBotEnv


def project_root():
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()

def main():
    rclpy.init()

    root = project_root()
    log_dir = root / "logs" / "training" / "ppo"
    models_dir = root / "models" / "PPO"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("⚡ 正在初始化环境...")
    env = TurtleBotEnv()

    print("🧠 初始化 PPO 算法 (高探索模式 ent_coef=0.01)...")
    # ent_coef=0.01 强迫 AI 尝试新动作，防止它死守停车策略
    model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log=str(log_dir), 
                learning_rate=0.0003,
                ent_coef=0.01)

    # 2. 开始训练
    # 50,000 步足够让它学乖了
    TOTAL_TIMESTEPS = 50000 
    print(f"🤖 开始训练！目标步数: {TOTAL_TIMESTEPS}")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        # 保存模型
        save_path = models_dir / "ppo_turtlebot_final"
        model.save(str(save_path))
        print(f"✅ 训练完成！模型已保存至: {save_path}")
        
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断，正在保存当前模型...")
        model.save(str(models_dir / "ppo_turtlebot_interrupted"))
    
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
