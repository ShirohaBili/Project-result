"""
    本代码实现了访问API并获得标签的功能
    目前这个代码并不全面，出于隐私，我们省去了我们的API接口的Key值，防止被盗用。
"""
import base64
import requests

API_KEY = "."
SECRET_KEY = "."


def main():
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal?access_token=" + get_access_token()

    path = r"imagePath"
    base64_data = get_file_content_as_base64(path)

    payload = {"image": base64_data}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    label = requests.request("POST", url, headers=headers, data=payload,top_num = 50)
    print(label.text)

    getLabel(str(label.text))


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
    #print(label)

    temp = ""

    print(label)
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


    print(possibility)
    print(label_split)




def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()

def get_file_content_as_base64(path):
    """
    获取文件base64编码
    :param path: 文件路径
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf8")


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
