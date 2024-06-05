# --*-- conding:utf-8 --*--
# @Time : 2024/1/11 17:21
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : Model_API.py
# @Software : PyCharm
# @Description :  各个模型的API接口
import os
import openai
import requests
import urllib
import json
import time
from http import HTTPStatus
import dashscope
import random
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

class API():
    def __init__(self, api_key_list: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0,
                 max_tokens: int = 1024):
        self.api_key_list = api_key_list
        self.model_name = model_name  # 新的model, 支持1w+
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.llm = LLM("/data/home-old/mguan/MyProject/ChatWK-main/Qwen-merge-lora/pt-chat-1-9-lora",tokenizer_mode='auto',
        #       trust_remote_code=True,
        #       enforce_eager=True,
        #       enable_prefix_caching=True)
        self.llm = LLM("/data/home-old/mguan/MyProject/ChatWK-main/Qwen1.5-14B-Chat", tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              enable_prefix_caching=True)
        self.sampling_params = SamplingParams(temperature=0.85, top_p=0.8, max_tokens=512)
        self.tokenizer = AutoTokenizer.from_pretrained("/data/home-old/mguan/MyProject/ChatWK-main/Qwen1.5-14B-Chat", trust_remote_code=True)

    # GPT系列
    def send_request_turbo(self, prompt, question):
        """
        """
        
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}

        messages = [zero_shot_prompt_message]
        # messages = []
        question = f"输入：\n问题是：\n{question}输出：\n"
        message = {"role": "user", "content": question}
        messages.append(message)
        for m in messages:
            print(m["role"])
            print(m["content"])
        output = {}
        model_output = ""
        try:
            if 'qwen1.5-14b-chat' in self.model_name:
                model_output = self.qwen15_14b_chat_api(messages)
        except Exception as e:
            print('Exception:', e)
        return model_output

    # 多轮会话
    def send_request_chat(self, prompt, share_content, questions, question_type="A3+A4"):
        """
        reference_lists: 里面的列表元素是每个问题对应的参考信息列表
        title_lists
        """
        if question_type == "A3+A4":
            question_chose = "请根据相关的知识和案例的内容，选出唯一一个正确的选项\n"
        else:
            question_chose = "请根据相关的知识，选出在共享答案中唯一一个正确的选项\n"
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}
        messages = [zero_shot_prompt_message]
        # messages = []
        i = 0
        if question_type == "A3+A4":
            messages.append({
                'role': 'user',
                'content': f"案例是：{share_content}"
            })
        else:
            messages.append({
                'role': 'user',
                'content': f"{share_content}"
            })
            # share_content_prompt = f"{share_content}"
        model_output_list = []
        while i < len(questions):
            sub_question = questions[i]['sub_question']
            question = f"输入：问题是：\n问题{sub_question}{question_chose}输出：\n"
            message = {"role": "user", "content": question}
            messages.append(message)
            for m in messages:
                print(m["role"])
                print(m["content"])
            model_output = ""
            try:
                if 'qwen1.5-14b-chat' in self.model_name:
                    model_output = self.qwen15_14b_chat_api(messages)
            except Exception as e:
                print('Exception:', e)
            messages.append({
                "role": "assistant",
                "content": model_output
            })
            model_output_list.append(model_output)
            i += 1
        # print(model_output)
        return model_output_list

    def qwen15_14b_chat_api(self, messages):
        cnt = 0
        while (cnt < 10):
            text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            outputs = self.llm.generate(text, self.sampling_params)
            for output in outputs:
                # model_output = output.outputs[0].text.split("<|endoftext|>")[0]
                model_output = output.outputs[0].text
            if model_output != "" and model_output != "答案：":
                break
            cnt += 1
            print('*'*50, cnt, '*'*50, f"Generated text: {model_output!r}")
        return model_output




