import os, json, re

LLM_AVAILABLE = False
try:
    from dashscope import Generation
    import dashscope
    api_key = os.environ.get("DASHSCOPE_API_KEY", "你的API Key")
    if api_key:
        dashscope.api_key = api_key
        LLM_AVAILABLE = True
        print("✅ 通义千问已连接\n")
except ImportError:
    print("⚠️  未安装 dashscope，将使用关键词解析\n")

SYSTEM_PROMPT = """你是一个机械臂控制器。将用户的自然语言指令解析为 JSON。
只输出 JSON，不要任何解释。

JSON 格式：
{"action": "pick"|"place"|"stop"|"unknown", "target": "物体描述"}

规则：
- "抓/拿/取/捡"某物 → pick
- "放/放下" → place
- "停/停止/别动" → stop
- 暗示需求但未指明物体 → 推理最合理物体，action=pick
- 无法理解 → unknown

桌面物体：瓶子(bottle)、苹果(apple)、杯子(cup)、书(book)、手机(phone)

示例：
"把瓶子抓起来" → {"action":"pick","target":"瓶子"}
"我渴了" → {"action":"pick","target":"杯子"}
"我饿了" → {"action":"pick","target":"苹果"}
"停下" → {"action":"stop","target":null}"""


def parse_with_llm(text):
    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            result_format="message",
        )
        reply = response.output.choices[0].message.content.strip()
        reply = re.sub(r"^```(?:json)?\s*", "", reply)
        reply = re.sub(r"\s*```$", "", reply)
        return json.loads(reply)
    except Exception as e:
        return {"action": "error", "error": str(e), "raw": text}


def simple_parse(text):
    """关键词解析备选方案"""
    if "停" in text or "别动" in text:
        return {"action": "stop", "target": None}
    if "重置" in text:
        return {"action": "reset", "target": None}
    if "放" in text:
        return {"action": "place", "target": None}
    
    intent_map = {
        "渴": "cup", "饿": "apple", "看书": "book",
        "打电话": "phone", "充电": "phone",
    }
    for kw, obj in intent_map.items():
        if kw in text:
            return {"action": "pick", "target": obj}
    
    obj_map = {
        "瓶子": "bottle", "苹果": "apple", "杯子": "cup",
        "书": "book", "手机": "phone",
    }
    for kw in ["抓", "拿", "取", "捡", "帮我", "把"]:
        if kw in text:
            for zh, en in obj_map.items():
                if zh in text:
                    return {"action": "pick", "target": zh}
    for zh, en in obj_map.items():
        if zh in text:
            return {"action": "pick", "target": zh}
    
    return {"action": "unknown", "target": None, "raw": text}


def main():
    # 准备一组测试指令
    test_cases = [
        "帮我拿苹果",          # 直接指令
        "我渴了",              # 隐含意图（需要推理）
        "帮我打个电话",        # 动作关联物体
    ]

    print("--- 开始测试 LLM 解析能力 ---")
    for text in test_cases:
        if LLM_AVAILABLE:
            command = parse_with_llm(text)
            print(f"'{text}' → {json.dumps(command, ensure_ascii=False)}")
        else:
            command = simple_parse(text)
            print(f"⚙️ 关键词解析: '{text}' → {json.dumps(command, ensure_ascii=False)}")
        print()

if __name__ == "__main__":
    main()
