import rclpy
from stable_baselines3 import SAC
import os
from pathlib import Path

from llm_rl_nav.envs.turtlebot_env import TurtleBotEnv


def project_root():
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()

def main():
    rclpy.init()

    root = project_root()
    log_dir = root / "logs" / "training" / "sac_upgrade"
    models_dir = root / "models" / "SAC_Upgrade"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("⚡ 初始化环境 (60线激光鹰眼版)...")
    env = TurtleBotEnv()

    print("🧠 初始化 SAC (Deep Network 版)...")
    
    # --- 【关键修改】定义大模型网络 ---
    # net_arch=[256, 256] 意味着有两个隐藏层，每层 256 个神经元
    # 这比默认的 [64, 64] 强太多了，能拟合更复杂的函数
    policy_kwargs = dict(net_arch=[256, 256])

    model = SAC("MlpPolicy", env, verbose=1, 
                tensorboard_log=str(log_dir),
                buffer_size=50000,
                learning_rate=3e-4,
                batch_size=256,
                ent_coef='auto',
                gamma=0.99,
                tau=0.005,
                policy_kwargs=policy_kwargs) # 【注入大模型参数】

    # 大脑大了，训练需要的步数也要稍微多一点才能喂饱
    TOTAL_TIMESTEPS = 60000 
    print(f"🤖 开始大模型训练！目标步数: {TOTAL_TIMESTEPS}")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
        model.save(str(models_dir / "sac_turtlebot_deep"))
        print("✅ 训练完成！")
        
    except KeyboardInterrupt:
        print("⚠️ 中断保存...")
        model.save(str(models_dir / "sac_turtlebot_interrupted"))
    
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
