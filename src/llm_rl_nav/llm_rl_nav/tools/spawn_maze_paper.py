import os
import time
import random

def spawn_cylinder(name, x, y):
    """
    生成圆柱体障碍物 (模拟论文中的圆形障碍)
    使用 'beer' 模型作为替代，因为它在 Gazebo 中是标准的圆柱体。
    """
    # 论文中障碍物尺寸约为 0.4m 
    cmd = f"ros2 run gazebo_ros spawn_entity.py -entity {name} -database beer -x {x} -y {y} -z 0.0"
    # 后台运行，不阻塞
    os.system(cmd + " > /dev/null 2>&1 &")

def spawn_goal(x, y):
    """生成终点标记 (用纸箱表示)"""
    cmd = f"ros2 run gazebo_ros spawn_entity.py -entity goal_marker -database cardboard_box -x {x} -y {y} -z 0.0"
    os.system(cmd + " > /dev/null 2>&1 &")

def clear_all():
    print("🧹 正在清理旧环境 (删除 obs_* 和 goal_marker)...")
    # 批量删除可能存在的旧障碍物
    for i in range(60):
        cmd = f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \"{{name: 'obs_{i}'}}\""
        os.system(cmd + " > /dev/null 2>&1")
    
    cmd = "ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \"{name: 'goal_marker'}\""
    os.system(cmd + " > /dev/null 2>&1")
    
    # 等待物理引擎清理完毕
    time.sleep(2)

def create_map_5():
    print("🏗️ 正在构建论文 Map 5 (45个障碍物，15x15m)...")
    
    obstacles = []
    
    # === 根据论文 Table 1 原始数据重建布局 ===
    # 论文中 Map 5 的坐标数据呈现出一定的线性排列和散布特征 
    
    # 1. 左下区域的密集阵列 (模拟 X: 1~3, Y: 1~4 区域)
    # 这会逼迫机器人一开始就要绕路
    for i in range(5):
        obstacles.append((1.5, 1.0 + i*0.8))
        obstacles.append((3.0, 1.5 + i*0.8))

    # 2. 中间的斜向封锁线 (模拟 X: 4~8, Y: 4~8 区域)
    # 这是一个典型的 U 型陷阱结构
    for i in range(8):
        obstacles.append((4.5 + i*0.6, 4.5 + i*0.6)) # 对角线墙
        
    # 3. 上方的干扰集群 (模拟 X: 8~10, Y: 10~12)
    obstacles.append((8.0, 10.0))
    obstacles.append((8.0, 11.0))
    obstacles.append((9.0, 10.5))
    obstacles.append((9.0, 11.5))
    
    # 4. 右侧的狭窄通道 (模拟 X: 12~14)
    # 终点前的最后考验
    for i in range(6):
        obstacles.append((12.0, 5.0 + i*1.5))
        
    # 5. 补充随机障碍物以达到 45 个总数 (增加环境噪声)
    # 论文提到总数为 45 
    current_count = len(obstacles)
    target_count = 45
    
    random.seed(2026) # 固定随机种子，保证每次生成一样
    
    while len(obstacles) < target_count:
        rx = random.uniform(1, 14)
        ry = random.uniform(1, 14)
        
        # 简单防重叠：不要生成在起点(0,0)和终点(15,15)附近
        if (rx < 2 and ry < 2) or (rx > 13 and ry > 13):
            continue
            
        obstacles.append((rx, ry))

    # === 执行生成 ===
    for idx, (x, y) in enumerate(obstacles):
        spawn_cylinder(f"obs_{idx}", x, y)
        # 稍微延时，防止服务调用过快导致 Gazebo 丢包
        time.sleep(0.1)

    # 生成终点
    spawn_goal(15.0, 15.0)
    
    print(f"✅ Map 5 构建完成！共生成 {len(obstacles)} 个障碍物。")
    print(f"🚀 起点: (0, 0) -> 终点: (15, 15)")

def main():
    clear_all()
    create_map_5()


if __name__ == "__main__":
    main()
