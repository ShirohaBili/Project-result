"""
    使用前述的API接口，实现具体的类别划分。
    我们采用了人为打标签的方法，为数据进行预分类。
"""
import argparse
import os
import base64
import time

import requests
import shutil

API_KEY = "."
SECRET_KEY = "."

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def get_file_content_as_base64(path):
    """
    获取文件base64编码
    :param path: 文件路径
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf8")

# 将API返回的文字标签转为数字标签
def word_to_label(word:str):
    if '鱼' in word or '鲨' in word or '鲤' in word or '鱥' in word or '鲷' in word \
            or '鲫' in word or '豚' in word or '凤尾龙睛' in word or '鹅头红' in word or '琉金' in word or '鲟' in word\
            or '长嘴鳄' in word or '鲢' in word or '福鳄' in word or '鲃' in word or '鲉' in word or '鳗' in word \
            or '清道夫' in word or '三道鳞' in word or '鳊' in word or '鱤' in word or '鲀' in word or '鳐' in word\
            or '鲹' in word or '鱵' in word or '鳝' in word:
        return 0
    if '蝶' in word:
        return 1
    if '狗' in word or '犬' in word or '獒' in word or '梗' in word or '㹴' in word \
            or '纳瑞' in word or '拉布拉多' in word or '大丹' in word or '博美' in word or '可卡' in word\
            or '马尔济斯' in word or '狆' in word or '喜乐蒂' in word or '京巴' in word or '比格' in word\
            or '柯基' in word:
        return 2
    if '鸟' in word or '蜂虎' in word or '雁' in word or '鹑' in word or '雀' in word \
            or '鹧鸪' in word or '鹰' in word or '鹭' in word or '鸦' in word or '鹤' in word or '燕' in word or '鹦鹉' in word\
            or '莺' in word or '鸬鹚' in word or '鹧鸪' in word or '鸫' in word or '鹫' in word or '鸥' in word or '鹟' in word\
            or '鹬' in word or '鸨' in word or '鹃' in word or '鸻' in word :
        return 3
    if '蜥' in word or '蜴' in word or '石龙子' in word or '中国水龙' in word:
        return 4
    if '蛇' in word or '蟒' in word or '竹叶青' in word or '蝰' in word or '铜头蝮' in word:
        return 5
    if '猴' in word or '猩' in word or '狒' in word or '猿' in word or '狨' in word or '山魈' in word:
        return 6
    return '9//'+word


# 根据API的返回得到某个照片的标签
def getLabel(label):
    left = 0
    right = 0
    for i in range(len(label)):
        if label[i] == '[':
            left = i
        elif label[i] == ']':
            right = i
    label = label[left + 1:right]
    possibility = []
    label_split = []
    counter = 0
    temp = ""

    # print(label)
    i = 0
    while i < len(label) - 1:
        if label[i] == '\"':
            counter += 1

        if counter == 3:
            i = i + 1
            left = i
            while label[i] != '\"':
                i += 1
            right = i
            counter += 1

            temp = label[left:right]
            possibility.append(temp)
            #print(possibility)
            temp = ""

        if counter == 7:
            i = i + 1
            left = i
            while label[i] != '\"':
                i += 1
            right = i
            counter += 1

            temp = label[left:right]
            label_split.append(temp)
            #print(label_split)
            temp = ""

            if counter == 8:
                counter = 0
        i += 1
    return word_to_label(label_split[0])



url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal?access_token=" + get_access_token()

parser = argparse.ArgumentParser()

parser.add_argument('source', type=str, help='the source image folder')
parser.add_argument('target', type=str, help='target folder to save the classified images')

opt = parser.parse_args()
source = opt.source
target = opt.target



def run(source,target):
    source_floder = './/' + source
    target_folder = './/' + target
    folders = []
    folders.append(source_floder)
    while len(folders) > 0:
        folder = folders.pop(0)

        for file in os.listdir(folder):
            img = folder + '//' +file
            if os.path.isdir(img):
                folders.append(img)
            else:
                base64_data = get_file_content_as_base64(img)
                payload = {"image": base64_data}
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
                while True:
                    try:
                        # response = requests.get(url, headers=headers, timeout=(30, 50), verify=False)
                        response = requests.request("POST", url, headers=headers, data=payload)
                        break
                    except:
                        print("Connection refused by the server..")
                        print("Let me sleep for 5 seconds")
                        print("ZZzzzz...")
                        time.sleep(5)
                        print("Was a nice sleep, now let me continue...")
                        continue

                label = getLabel(response.text)
                time.sleep(0.1)
                print(label)
                dirs = target_folder + '//' +str(label)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                shutil.copy(img,dirs)
                # time.sleep(0.6)

if __name__ == '__main__':
    run(source, target)





