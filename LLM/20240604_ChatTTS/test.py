import random, os, tqdm, time, json, re, IPython, zhdate, sys
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from datetime import datetime

# sys.path.append("../../")

# from train_predict_tools_lgbm import *
# from train_predict_tools import * 
# from perf_eval_tools import * 
# from fea_verification import *
# from third_party_data_verify import * 

random.seed(618)
np.random.seed(907)

tqdm.tqdm.pandas() ## 引入这个，就可以在apply的时候用progress_apply了。

sys.path.append("../../../")
new_base_path = os.path.join(
    "/mnt/d/forCoding_data",
    "/".join(
        os.getcwd().split("/")[-1*(len(sys.path[-1].split("/")) - 1):]
    ),
)
print("storage dir:", new_base_path)
print("code dir:", os.getcwd(), "\n")

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

def load_data_from_newbasepath(filename, dirname = new_base_path, foldername = "originalData", fmt = "csv"):
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        cmd = f'pd.read_{fmt}("{file_path}", quoting=3, lineterminator="\\n")'
    else:
        cmd = f'pd.read_{fmt}("{file_path}")'
    print(cmd)
    return eval(cmd)
def store_data_to_newbasepath(df, filename, dirname = new_base_path, foldername = "preprocessedData", fmt = "parquet", index=False):
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    cmd = f'df.to_{fmt}("{file_path}", index={index})'
    print(cmd)
    eval(cmd)
    print("data saved.")
    return file_path
def load_data_from_newbasepath__waitUntilDownloaded(filename, dirname = new_base_path, foldername = "originalData", fmt = "csv"):
    flag_path = os.path.join(dirname, foldername, filename + "---downloan_finish_flag.txt")
    print("Downloading, please wait a moment...")
    while True:
#         print(flag_path)
        if os.path.exists(flag_path):
            print("Downloading finished.")
            break
        time.sleep(10)
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        cmd = f'pd.read_{fmt}("{file_path}", quoting=3, lineterminator="\\n")'
    else:
        cmd = f'pd.read_{fmt}("{file_path}")'
    print(cmd)
    return eval(cmd)
def load_data_from_originalData(filename, dirname = new_base_path, foldername = "originalData", fmt = "csv"):
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        cmd = f'pd.read_{fmt}("{file_path}", quoting=3, lineterminator="\\n")'
    else:
        cmd = f'pd.read_{fmt}("{file_path}")'
    print(cmd)
    return eval(cmd)
def load_data_from_preprocessedData(filename, dirname = new_base_path, foldername = "preprocessedData", fmt = "parquet", use_cols = None):
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", quoting=3, lineterminator="\\n")'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", usecols = {use_cols}, quoting=3, lineterminator="\\n")'
    else:
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}")'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", columns={use_cols})'
    print(cmd)
    return eval(cmd)
def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)
def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)
def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)

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
        f"程序于这个点跑完：\n{millisec2datetime(time.time()*1000)}", fontproperties=font, ha="center", va="center", color="red"
    )  
    # 设置图形的布局  
    # ax.set_xlim(0, 1)  
    # ax.set_ylim(0, 1)  
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.patch.set_color("blue")
    # 显示图形  
    plt.show()

def kill_current_kernel():
    '''杀死当前的kernel释放内存空间。'''
    IPython.Application.instance().kernel.do_shutdown(True) 
    
def simply_show_data(df1):
    print(df1.shape)
    # display(df1.head())
    
def wait_flag(saved_flag_path, time_interval_sec=10):
    print("waiting for", saved_flag_path)
    time_count = 0
    while True:
        if os.path.exists(saved_flag_path):
            break
        time.sleep(time_interval_sec)
        time_count+=time_interval_sec
        print(time_count, end=" ")
    print("finish!!")

def parallelly_run_multiple_similar_python_code(codes, nb_workers = 4):
    '''
    codes是多条相似的python代码。
    这个函数的作用就是将其平行地跑，每一条python代码就对应一个线程。或许可以后续优化，比如固定线程数为一个特定值。
    nb_workers 如果赋值为
    '''
    assert (isinstance(nb_workers, int)), "`nb_workers' should be int."
    df_sqls = pd.DataFrame(
        {
            "func": codes

        }
    )
    # display(df_sqls)
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers = df_sqls.shape[0] if nb_workers<0 else nb_workers, progress_bar = True)
    def run_sql_prlly(row):
        try: 
            cmd = f'{row["func"]}'
            print(cmd, "\n")
            eval(cmd)
            return "0-success"
        except Exception as e:
            return e
    df_sqls["run_rsts"] = df_sqls.parallel_apply(lambda row: run_sql_prlly(row), axis = 1)
    # display(df_sqls)
    
class TimerContext:  
    def __enter__(self):  
        self.start_time = str(datetime.now())
        print("start time:", self.start_time)
        return self  
    def __exit__(self, exc_type, exc_val, exc_tb):  
        print("start time:", self.start_time)
        print("end time", str(datetime.now()))

def three_num_get_gua(a, b, c):
    '''梅花易数三数起卦，以取本、互、变。'''
    bagua = ["111", "110", "101", "100", "011", "010", "001", "000"]
    guatu = {
        "111": ("☰", "天", "乾金"), 
        "110": ("☱", "泽", "兑金"),
        "101": ("☲", "火", "离火"),
        "100": ("☳" , "雷", "震木"),
        "011": ("☴", "风", "巽木"),
        "010": ("☵", "水", "坎水"),
        "001": ("☶", "山", "艮土"),
        "000": ("☷", "地", "坤土"),
    }
    print(
        "先天八卦数:", ", ".join([f"{i}{guatu[j][-1][0]}"for i, j in zip(range(1,9), bagua)])
    )
    ## https://zhuanlan.zhihu.com/p/457104350
    gua_64 = "天天乾，天风姤，天山遁，天地否，风地观，山地剥，火地晋，火天大有，水水坎，水泽节，水雷屯，水火既济，泽火革，雷火丰，地火明夷，地水师，山山艮，山火贲，山天大畜，山泽损，火泽睽，天泽履，风泽中孚，风山渐，雷雷震，雷地豫，雷水解，雷风恒，地风升，水风井，泽风大过，泽雷随，风风巽，风天小畜，风火家人，风雷益，天雷无妄，火雷噬嗑，山雷顾，山风蛊，火火离，火山旅，火风鼎，火水未济，山水蒙，风水涣，天水松，天火同人，地地坤，地雷复，地泽临，地天泰，雷天大壮，泽天夬，水天需，水地比，泽泽兑，泽水困，泽地萃，泽山咸，水山蹇，地山谦，雷山小过，雷泽归妹"
    gua_64_dict = {x[:2]: x[2:]for x in gua_64.split("，")}
    
    shanggua_idx = 7 if (a % 8 == 0) else (a % 8 - 1)
    xiagua_idx = 7 if (b % 8 == 0) else (b % 8 - 1)
    bianyao_idx = 5 if (c % 6 == 0) else (c % 6 - 1)
    print("变爻:", bianyao_idx+1)
    bengua = bagua[xiagua_idx] + bagua[shanggua_idx]
    hugua = bengua[1:-1][:3] + bengua[1:-1][1:]
    biangua = list(bengua)
    biangua[bianyao_idx] = str(1 - int(biangua[bianyao_idx]))
    biangua = "".join(biangua)
    df = pd.DataFrame([[
        guatu[bengua[3:]][0]+guatu[bengua[3:]][2], guatu[hugua[3:]][0]+guatu[hugua[3:]][2], guatu[biangua[3:]][0]+guatu[biangua[3:]][2], 
    ],[
        guatu[bengua[:3]][0]+guatu[bengua[:3]][2], guatu[hugua[:3]][0]+guatu[hugua[:3]][2], guatu[biangua[:3]][0]+guatu[biangua[:3]][2], 
    ]], index=["上卦", "下卦"], columns = [
        guatu[bengua[3:]][1] + guatu[bengua[:3]][1] + gua_64_dict[guatu[bengua[3:]][1] + guatu[bengua[:3]][1]],
        guatu[hugua[3:]][1] + guatu[hugua[:3]][1] + gua_64_dict[guatu[hugua[3:]][1] + guatu[hugua[:3]][1]],
        guatu[biangua[3:]][1] + guatu[biangua[:3]][1] + gua_64_dict[guatu[biangua[3:]][1] + guatu[biangua[:3]][1]],
    ])
    # display(df)
    return bengua, hugua, biangua
    
def easy_start_gua():
    """用公历的日、时、分来起卦。"""
    n1, n2, n3 = str(datetime.now())[8:10], str(datetime.now())[11:13], str(datetime.now())[14:16]
    print(n1, n2, n3)
    return three_num_get_gua(int(n1), int(n2), int(n3))
# easy_start_gua()

def easy_start_gua_lunar():
    '''用农历的月、日、时辰来起卦。'''
    time_now = datetime.now()
    zh_date_str = str(zhdate.ZhDate.from_datetime(time_now))
    zh_date_str_1 = datetime.strftime(
        datetime(
            *[int(x) for x in re.findall("\d+", zh_date_str)]
        ),
        '%Y-%m-%d'
    )
    zh_hour = (time_now.hour + 1)//2%12+1
    zh_hour_dizhi = "子、丑、寅、卯、辰、巳、午、未、申、酉、戌、亥".split("、")[zh_hour-1]
    
    n1, n2, n3 = zh_date_str_1[5:7], zh_date_str_1[8:10], zh_hour
    print(n1, n2, n3, f"{zh_hour_dizhi}时")
    return three_num_get_gua(int(n1), int(n2), int(n3))
# easy_start_gua_lunar()


import argparse
parser = argparse.ArgumentParser("For test the parser")
parser.add_argument('-since')
parser.add_argument('-till')
args = parser.parse_args()



import ChatTTS
import torch
import torchaudio
import scipy

import sounddevice as sd
from epub2txt import epub2txt

chat = ChatTTS.Chat()
chat.load(
    device="cuda",
    compile=True, 
    source="custom", 
    custom_path=create_trained_models_path("chatTTS")
)

chunk_size = 50

# from a url to epub
# url = "https://github.com/ffreemt/tmx2epub/raw/master/tests/1.tmx.epub"
# res = epub2txt(url)

# from a local epub file
filepath = r'/mnt/c/Users/Administrator/Desktop/翦商：殷周之变与华夏新生 (李硕) (Z-Library).epub'
res = epub2txt(filepath)

# output as a list of chapters
ch_list = epub2txt(filepath, outputlist=True)
# chapter titles will be available as epub2txt.content_titles if available

ori_str = "。".join(ch_list).replace(
    "\n", " "
).replace(
    "\xa0", " "
).replace(
    "1", "一"
).replace(
    "2", "二"
).replace(
    "3", "三"
).replace(
    "4", "四"
).replace(
    "5", "五"
).replace(
    "6", "六"
).replace(
    "7", "七"
).replace(
    "8", "八"
).replace(
    "9", "九"
).replace(
    "0", "零"
).replace(
    ". ", "。"
)

def split_into_chunks_by_sentences(text, chunk_size=150):
    """
    将字符串按照句号拆分，并进一步组合成长度大致为chunk_size的若干子串。

    参数:
    text (str): 要处理的字符串。
    chunk_size (int): 每个子串的大致长度，默认为150。

    返回:
    list: 包含长度大致为chunk_size的子串的列表。
    """
    # 使用句号拆分字符串为句子列表（考虑句号后可能有空格或换行符）
    sentences = [sentence.strip() for sentence in text.split('。') if sentence.strip()]
    
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for sentence in sentences:
        # 计算如果添加当前句子后子串的长度
        new_chunk_length = current_chunk_length + len(sentence) + 1  # +1 是为了考虑添加句号作为分隔符
        
        # 如果添加后长度超过chunk_size，则保存当前子串并开始新的子串
        if new_chunk_length > chunk_size:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_chunk_length = new_chunk_length - 1  # 不再需要+1，因为我们在最后会添加句号作为分隔符（但在实际拼接时不加在最后一个句子后）
    
    # 添加最后一个子串（如果有的话）
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks

def find_flat_period(data, threshold, window_size):
    """
    找到波形图中最后一段平缓期并返回其开始位置。

    参数:
    data (list or numpy array): 一维波形数据。
    threshold (float): 用于判断平缓期的标准差阈值。
    window_size (int): 用于计算标准差的窗口大小。

    返回:
    int: 平缓期开始的位置索引。
    """
    n = len(data)

    thr = -1
    for i in range(n - window_size + 1, -1, -1000):
        window_data = data[i:i + window_size]
        std_dev = np.std(window_data)
        if std_dev <= threshold:
            thr=i
        else:
            break
    # 如果没有找到平缓期，返回-1
    return thr

## https://github.com/2noise/ChatTTS/issues/705

# 默认种子
DEFAULT_AUDIO_SEED_INPUT = 47
DEFAULT_TEXT_SEED_INPUT = 24
DEFAULT_SEED = 425

def generate_audio_file(chat, texts=None, output_path="word_level_output.wav",
                        audio_seed=DEFAULT_AUDIO_SEED_INPUT,
                        text_seed=DEFAULT_TEXT_SEED_INPUT,
                        seed=DEFAULT_SEED):

    torch.manual_seed(seed)

    temperature = 0.3
    top_P = 0.7
    top_K = 20
    refine_text_flag = False

    torch.manual_seed(audio_seed)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K)
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='',) ## [oral_2][laugh_0][break_6]

    torch.manual_seed(text_seed)
    if refine_text_flag:
        text = chat.infer(texts,
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    else:
        text = texts

    wavs = chat.infer(text,
                      skip_refine_text=True,
                      params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code,
                     )
    
    return wavs

strs = split_into_chunks_by_sentences(ori_str, chunk_size)
len(strs)

bounds = []
interval = 10
for idx, x in enumerate(range(0, len(strs), interval)):
    bounds.append((idx, x, x+interval))
print(len(bounds))

for bound in bounds[int(args.since):int(args.till)]:
    to_see = [x.replace(
        "，", "[uv_break]"
    ).replace(
        "。", "[lbreak]"
    ).replace(
        "、", "[uv_break]"
    ).replace(
        " ", "[uv_break]"
    ) for x in strs[bound[1]:bound[2]]]
    print(bound, end=" ")
    # print(to_see)
    wavs = generate_audio_file(
        chat, 
        to_see, 
    )
    scipy.io.wavfile.write(
        filename=create_preprocessedData_path(f"part_{bound}.wav"), 
        rate=24_000, 
        data=np.concatenate(
            [
                wavs[i].T[:find_flat_period(wavs[i].T, 0.01, 5000)] for i in range(len(to_see))
            ] 
        ) 
    )