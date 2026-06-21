import os

# ==========================================
# 🗺️ 5大经典算法测试陷阱地图库 (0=空地, 1=墙壁)
# ==========================================

MAPS = {
    "1_u_shape": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    "2_narrow": [
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ],
    "3_maze": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    "4_clutter": [
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    ],
    "5_spiral": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
}

# ==========================================
# ⚙️ 配置区 (在这里修改你要测试的地图！)
# ==========================================
# 可选值: "1_u_shape", "2_narrow", "3_maze", "4_clutter", "5_spiral"
CHOOSE_MAP = "1_u_shape"  

RESOLUTION = 1.0  # 每个方格代表的真实物理大小 (单位: 米)
WALL_HEIGHT = 1.0 # 墙的高度 (单位: 米)

def generate_world_file(grid_key, grid, filename="custom_grid.world"):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # 1. 写入 SDF 头
    world_content = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
"""

    # 2. 遍历矩阵生成墙体模型
    block_count = 0
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == 1:
                # 坐标转换：由于图片/矩阵通常原点在左上角，物理世界通常在左下或中间，这里做了镜像和缩放
                real_x = x * RESOLUTION
                real_y = (rows - 1 - y) * RESOLUTION 
                
                block_sdf = f"""
    <model name='wall_block_{block_count}'>
      <pose>{real_x} {real_y} {WALL_HEIGHT/2} 0 0 0</pose>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{RESOLUTION} {RESOLUTION} {WALL_HEIGHT}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{RESOLUTION} {RESOLUTION} {WALL_HEIGHT}</size></box></geometry>
          <material><ambient>0.4 0.2 0.6 1</ambient></material> </visual>
      </link>
    </model>"""
                world_content += block_sdf
                block_count += 1

    # 3. 写入尾部
    world_content += "\n  </world>\n</sdf>"

    with open(filename, 'w') as f:
        f.write(world_content)
    
    print("=" * 40)
    print(f"✅ 生成成功: {filename}")
    print(f"🗺️  当前选用地图: {grid_key}")
    print(f"🧱 共生成 {block_count} 个障碍方块")
    print("=" * 40)

def main():
    if CHOOSE_MAP in MAPS:
        # 直接输出固定名称，方便 launch 文件统一读取
        generate_world_file(CHOOSE_MAP, MAPS[CHOOSE_MAP], "custom_grid.world")
    else:
        print(f"❌ 错误: 找不到地图 '{CHOOSE_MAP}'，请检查配置名是否拼写正确。")


if __name__ == "__main__":
    main()
