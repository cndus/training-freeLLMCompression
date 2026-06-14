import os
from openai import OpenAI

client = OpenAI(
    # 新加坡和北京地域的API Key不同。获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    api_key="sk-e4bca3558ebf46fb95f8850dc5caf152",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen2.5-omni-7b",
    messages=[{"role": "user", "content": "你是谁"}],
    # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
    modalities=["text", "audio"],
    audio={"voice": "Ethan", "format": "wav"},
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if chunk.choices:
        print(chunk.choices[0].delta)
    else:
        print(chunk.usage)