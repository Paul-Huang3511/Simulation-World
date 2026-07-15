
import time
import pybullet as p
import pybullet_data

# ── 场景物体配置 ──
SCENE_OBJECTS = {
    "bottle": {"color": [0.2, 0.6, 1.0, 1.0], "shape": "cylinder", "size": [0.03, 0.12]},
    "apple":  {"color": [1.0, 0.2, 0.1, 1.0], "shape": "sphere",   "size": [0.04]},
    "cup":    {"color": [0.9, 0.8, 0.2, 1.0], "shape": "cylinder", "size": [0.04, 0.09]},
    "book":   {"color": [0.5, 0.3, 0.8, 1.0], "shape": "box",      "size": [0.12, 0.09, 0.02]},
    "phone":  {"color": [0.2, 0.2, 0.2, 1.0], "shape": "box",      "size": [0.07, 0.14, 0.01]},
}

OBJECT_POSITIONS = {
    "bottle": [0.5,  0.1,  0.82],
    "apple":  [0.5, -0.1,  0.80],
    "cup":    [0.6,  0.0,  0.81],
    "book":   [0.4, -0.2,  0.80],
    "phone":  [0.55, 0.2,  0.80],
}

# 全局变量
object_ids = {}
robot_id = None
num_joints = 0


def create_table():
    """创建桌子"""
    table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.35, 0.38])
    table_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.4, 0.35, 0.38],
        rgbaColor=[0.76, 0.6, 0.42, 1.0])
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_shape,
        baseVisualShapeIndex=table_visual,
        basePosition=[0.5, 0.0, 0.38])


def create_object(name, cfg, pos):
    """创建一个物体"""
    shape_type = cfg["shape"]
    size = cfg["size"]
    color = cfg["color"]

    if shape_type == "cylinder":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=size[0], height=size[1])
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color)
    elif shape_type == "sphere":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=size[0], rgbaColor=color)
    else:  # box
        half = [s / 2 for s in size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)

    body_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos)
    
    object_ids[name] = body_id


def place_objects():
    """批量放置桌面物体"""
    for name, cfg in SCENE_OBJECTS.items():
        pos = OBJECT_POSITIONS.get(name, [0.5, 0.0, 0.82])
        create_object(name, cfg, pos)


def go_home():
    """机械臂归位"""
    home_angles = [0, 0.4, 0, -1.0, 0, 1.2, 0]
    for i in range(min(num_joints, 7)):
        p.resetJointState(robot_id, i, home_angles[i])


def main():
    global robot_id, num_joints

    # 1. 连接仿真引擎（GUI 模式，弹出窗口）
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0.4, 0.0, 0.5])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 2. 设置物理参数
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    # 3. 创建地面
    p.loadURDF("plane.urdf")

    # 4. 创建桌子
    create_table()

    # 5. 加载机械臂
    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0.0, 0.0, 0.76],
        useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    go_home()

    # 6. 放置桌面物体
    place_objects()

    print("✅ 仿真世界已初始化")
    print(f"   桌面物体: {list(object_ids.keys())}")
    print("\n🎮 仿真窗口已打开，Ctrl+C 退出")

    # 7. 保持窗口运行
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 60)
    except KeyboardInterrupt:
        print("\n👋 关闭仿真")
        p.disconnect()


if __name__ == "__main__":
    main()


