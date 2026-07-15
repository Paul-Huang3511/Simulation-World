import os, sys, time, math, random
import numpy as np
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


class SimWorld:
    def __init__(self, headless=False):
        self.headless = headless
        self.object_ids = {}
        self.picked_items = set()

    def setup(self):
        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.4, 0.0, 0.5],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # 地面
        p.loadURDF("plane.urdf")

        # 桌子
        table_shape  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.35, 0.38])
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.35, 0.38],
            rgbaColor=[0.76, 0.6, 0.42, 1.0])
        p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=table_shape,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0.0, 0.38])

        # 机械臂
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0.0, 0.0, 0.76],
            useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        self._go_home()

        # 场景物体
        self._place_objects()

        print("✅ 仿真世界已初始化")
        print(f"   桌面物体: {list(self.object_ids.keys())}")
        return self

    def _place_objects(self):
        for name, cfg in SCENE_OBJECTS.items():
            shape_type = cfg["shape"]
            size = cfg["size"]
            color = cfg["color"]
            pos = OBJECT_POSITIONS.get(name, [0.5, 0.0, 0.82])

            if shape_type == "cylinder":
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=size[0], height=size[1])
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color)
            elif shape_type == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=size[0], rgbaColor=color)
            else:
                half = [s / 2 for s in size]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)

            body_id = p.createMultiBody(baseMass=0.1,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos)
            self.object_ids[name] = body_id

    def _go_home(self):
        self.HOME_ANGLES = [0, 0.4, 0, -1.0, 0, 1.2, 0]
        for i in range(min(self.num_joints, 7)):
            p.resetJointState(self.robot_id, i, self.HOME_ANGLES[i])

    def list_objects(self):
        return [n for n in self.object_ids if n not in self.picked_items]

    def close(self):
        p.disconnect()

    def keep_alive(self):
        print("\n🎮 仿真窗口已打开，Ctrl+C 退出")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 60)
        except KeyboardInterrupt:
            print("\n👋 关闭仿真")
            self.close()


if __name__ == "__main__":
    world = SimWorld()
    world.setup()
    world.keep_alive()
