#!/usr/bin/env python3

import sys

import cv2

from sim_world import SimWorld
from yolo_detector import YoloDetector


def configure_utf8_output(stream=sys.stdout):
       if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def main():
    configure_utf8_output()
    print("⏳ 初始化仿真世界……")
    world = SimWorld()
    world.setup()

    print("⏳ 加载 YOLO……")
    detector = YoloDetector("yolov8n.pt", conf_threshold=0.15)

    try:
        print("📸 拍摄仿真场景……")
        rgb_image = world.capture()
        print(f"   图片尺寸：{rgb_image.shape}")

        print("🔍 开始检测……")
        detections = detector.detect(rgb_image)

        if detections:
            print(f"✅ YOLO 检测到 {len(detections)} 个目标：")
            for item in detections:
                print(
                    f"   {item['class']:8s} "
                    f"conf={item['conf']:.2f} "
                    f"bbox={item['bbox']}"
                )
            detector.draw_boxes(rgb_image, detections)
        else:
            print("⚠️ YOLO 没有识别出目标")
            print(f"   仿真真值：{world.list_objects()}")

            raw_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("sim_capture_raw.png", raw_bgr)
            print("📸 原始图片已保存：sim_capture_raw.png")
    finally:
        world.close()


if __name__ == "__main__":
    main()
