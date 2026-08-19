import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloDetector:
    """把 YOLO 输出转换字典列表。"""

    COCO_TO_SIM = {
        "bottle": "bottle",
        "apple": "apple",
        "cup": "cup",
        "book": "book",
        "cell phone": "phone",
    }

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.15):
        self.model = None
        self.conf_threshold = conf_threshold

        if YOLO is None:
            print("⚠️ 未安装 ultralytics，YOLO 暂不可用")
            print("   安装命令：pip install ultralytics")
            return

        try:
            print("⏳ 正在加载 YOLO……")
            self.model = YOLO(model_path)
            print(f"✅ YOLO 模型已加载：{model_path}")
        except Exception as error:
            print(f"⚠️ YOLO 模型加载失败：{error}")

    def detect(self, rgb_image: np.ndarray) -> list[dict]:
        """检测一张 RGB 图片，返回类别、置信度和检测框。"""
        if self.model is None:
            return []

        result = self.model.predict(
            source=rgb_image,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = result.names[class_id]
            object_name = self.COCO_TO_SIM.get(label)
            if object_name is None:
                continue

            confidence = float(box.conf.item())
            bbox = box.xyxy[0].cpu().tolist()
            detections.append({
                "class": object_name,
                "label": label,
                "conf": round(confidence, 3),
                "bbox": [round(value) for value in bbox],
            })
        return detections

    @staticmethod
    def draw_boxes(rgb_image, detections, save_path="sim_detect.png"):
        """在图片上绘制检测框并保存。"""
        import cv2

        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            text = f"{item['class']} {item['conf']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(str(save_path), image)
        print(f"📸 检测图片已保存：{save_path}")
        return image
