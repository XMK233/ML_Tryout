import sys
import os
parent_path = os.path.dirname(sys.path[0])
print(parent_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)
# from LLAMAAPI import LlamaAPI
from Model_API import API
from bench_function import export_distribute_json, export_union_json
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="parameter of TCM_EDU_LLM")
    parser.add_argument(
        "--data_path",
        type=str,
        default='data/A_evaluate/evaluate',
        help="测试数据",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='qwen',
        help="The LLM name.",
    )
    parser.add_argument(
        "--sys_prompt",
        type=str,
        default='A1-2_prompt.json',
        help="选择不同测试题类型的指令.",
    )
    parser.add_argument(
        "--start_num",
        type=int,
        default=0,
        help="保存文档的起始id",
    )
    args = parser.parse_args()
    return args

# 测试主函数, 一个问题+一个答案 --》 bench_function

if __name__ == "__main__":
    args = parse_args()
    with open(f"{args.sys_prompt}", "r", encoding="utf-8") as f:
        data = json.load(f)['examples']
    f.close()
    for i in range(len(data)):
        directory = args.data_path
        model_name = args.model_name
        if 'qwen1.5-14b-chat' in model_name:
            print("model_name", model_name)
            api = API("", model_name=model_name)
        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print(keyword)
        print(question_type)
        export_distribute_json(
            api,
            model_name,
            directory,
            keyword,
            zero_shot_prompt_text,
            question_type,
            args,
            parallel_num=100,
        )

        export_union_json(
            directory,
            model_name,
            keyword,
            zero_shot_prompt_text,
            question_type
        )
    

# export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# export JRE_HOME=${JAVA_HOME}/jre
# export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
# export PATH=${JAVA_HOME}/bin:$PATH

