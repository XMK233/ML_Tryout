import random, os, tqdm, time, json
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

import sys, datetime

sys.path.append("../../../../")

random.seed(618)
np.random.seed(907)

new_base_path = os.path.join(
    "/Users/minkexiu/Downloads/",
    "/".join(
        os.getcwd().split("/")[-1 * (len(sys.path[-1].split("/")) - 1):]
    ),
)
print("storage dir:", new_base_path)
print("code dir:", os.getcwd())

## 创建文件夹。
if not os.path.exists(new_base_path):
    os.makedirs(
        new_base_path
    )
if not os.path.exists(os.path.join(new_base_path, "preprocessedData")):
    os.makedirs(
        os.path.join(new_base_path, "preprocessedData")
    )
if not os.path.exists(os.path.join(new_base_path, "originalData")):
    os.makedirs(
        os.path.join(new_base_path, "originalData")
    )
if not os.path.exists(os.path.join(new_base_path, "trained_models")):
    os.makedirs(
        os.path.join(new_base_path, "trained_models")
    )


def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)


def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)


def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)


def millisec2datetime(timestamp):
    time_local = time.localtime(timestamp / 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)


def run_finish():
    # 假设你的字体文件是 'myfont.ttf' 并且位于当前目录下
    font = FontProperties(fname="/Users/minkexiu/Documents/GitHub/ML_Tryout/SimHei.ttf", size=24)
    # 创建一个空白的图形
    fig, ax = plt.subplots()
    ax.imshow(
        plt.imread("/Users/minkexiu/Downloads/wallhaven-dgxpyg.jpg")
    )
    # 在图形中添加文字
    ax.text(
        ax.get_xlim()[1] * 0.5,
        ax.get_ylim()[0] * 0.5,
        f"程序于这个点跑完：\n{millisec2datetime(time.time() * 1000)}", fontproperties=font, ha="center", va="center",
        color="red"
    )
    # 设置图形的布局
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_color("blue")
    # 显示图形
    plt.show()


tqdm.tqdm.pandas()  ## 引入这个，就可以在apply的时候用progress_apply了。

import IPython


def kill_current_kernel():
    '''杀死当前的kernel释放内存空间。'''
    IPython.Application.instance().kernel.do_shutdown(True)

def wait_flag(saved_flag_path, time_interval_sec=10):
    print("waiting for", saved_flag_path)
    time_count = 0
    while True:
        if os.path.exists(saved_flag_path):
            break
        time.sleep(time_interval_sec)
        time_count += time_interval_sec
        print(time_count, end=" ")
    print("finish!!")

class TimerContext:
    def __enter__(self):
        self.start_time = str(datetime.now())
        print("start time:", self.start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("start time:", self.start_time)
        print("end time", str(datetime.now()))

# -*- coding: utf-8 -*-
import json
import os
import time
import logging
import argparse
from tqdm import tqdm
from add_prompt import PromptCreate
from llm_api.my_model import MyModelAPI

output_file = create_preprocessedData_path("xx_A_result_14B.jsonl")
data_path = create_originalData_path("初赛/初赛/test")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = MyModelAPI(logger)
pc = PromptCreate()

def run(model_api, in_file, out_file):
    questions = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    fd = open("error.txt", "w", encoding="utf-8")
    with open(out_file, "a+", encoding="utf-8") as fw:
        for question in tqdm(questions):
            # 添加prompt模板
            try:
                prompt = pc.create(key=question["task"], **question)
                # print(prompt)
                content = model_api.chat_generate(prompt)
                question["answer"] = content
                logger.info(question)
                fw.writelines(json.dumps(question, ensure_ascii=False))
                fw.writelines("\n")
                fw.flush()
                # break
            except:
                fd.write("{}, {}".format(in_file, question["task"]))
                continue
    fd.close()

if os.path.exists(output_file):
    os.remove(output_file)
start = time.time()
for src, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".jsonl"):
            in_path = os.path.join(src, file)
            run(model, in_path, output_file)
print(f"总计耗时：{(time.time()-start)/3600} h")
