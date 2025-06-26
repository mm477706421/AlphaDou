import requests
import json

def test_sandou_act():
    """
    测试出牌API
    """
    headers = {'Content-Type': 'application/json',
               'Accept': 'application/json'}

    data = {
        'player_position': 'landlord_down',  # 地主是'landlord' 地主下是'landlord_down' 地主上是'landlord_up'
        'player_hand_cards': '[3, 4, 5, 6, 6, 7, 8, 8, 8, 10, 10, 10, 11, 12, 12, 13, 17]',  # 自己当前的手牌 文本型
        'three_landlord_cards': '[8, 9, 9]',  # 三张地主的牌 文本型
        'card_play_action_seq': '[[9, 9, 9, 3]]',  # 出牌历史 只有一手是'[[9, 9, 9, 3]]' 多手比如'[[9, 9, 9, 3], [10, 10, 10, 5]]' 还没有出牌就是'[]' 文本型
        'bid_info': '[3, -1, -1]',  # 每个人的叫分  比如第一家不叫 第二家叫2分 第三家叫不叫 就是[0, 2, -1] 文本型
        'passwd': '12345678'  # 固定12345678  文本型
    }

    url = "http://localhost:5000/sandou_act"

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        print("出牌API响应状态码:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("出牌API响应结果:", result)
        else:
            print("出牌API请求失败:", response.text)
    except Exception as e:
        print("出牌API请求异常:", str(e))


def test_sandou_bid():
    """
    测试叫地主API
    """
    headers = {'Content-Type': 'application/json',
               'Accept': 'application/json'}

    data = {
        'player_position': 'third',  # 第几个叫牌 第一个是'first' 第二个是'second' 第三个是'third' 文本型
        'player_hand_cards': '[3, 3, 4, 4, 5, 5, 6, 8, 10, 10, 11, 11, 12, 12, 13, 13, 17]',  # 自己当前的手牌 文本型
        'bid_info': '[-1, -1, -1]',  # 每个选手的叫牌分数 如果没轮到他就是-1  比如第1家不叫 第2家叫3分 就是[0, 3, -1] 文本型
        'bid_model': 0,  # 固定为0
        'passwd': '12345678'  # 固定12345678  文本型
    }

    url = "http://localhost:5000/sandou_bid"

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        print("叫地主API响应状态码:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("叫地主API响应结果:", result)
        else:
            print("叫地主API请求失败:", response.text)
    except Exception as e:
        print("叫地主API请求异常:", str(e))


if __name__ == "__main__":
    print("开始测试出牌API...")
    test_sandou_act()
    
    print("\n开始测试叫地主API...")
    test_sandou_bid() 