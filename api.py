from flask import Flask, request, jsonify
import torch
import numpy as np
import json
import ast
from douzero.evaluation.deep_agent import DeepAgent, SupervisedModel
from collections import Counter

app = Flask(__name__)

# 加载模型
# 出牌模型
landlord_model = DeepAgent('landlord', 'baseline/best/landlord.ckpt')
landlord_up_model = DeepAgent('landlord_up', 'baseline/best/landlord_up.ckpt')
landlord_down_model = DeepAgent('landlord_down', 'baseline/best/landlord_down.ckpt')

# 叫牌模型
bid_model = SupervisedModel()

# 密码验证
PASSWD = "12345678"

class InfoSet(object):
    """
    根据douzero/env/game.py中的InfoSet类定义的简化版本
    只包含API所需的必要信息
    """
    def __init__(self, player_position):
        # 玩家位置：landlord, landlord_down, landlord_up, first, second, third
        self.player_position = player_position
        # 当前玩家的手牌
        self.player_hand_cards = []
        # 每个玩家剩余的牌数
        self.num_cards_left_dict = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        # 三张地主牌
        self.three_landlord_cards = []
        # 叫牌信息
        self.bid_info = [-1, -1, -1]
        # 出牌历史
        self.card_play_action_seq = []
        # 其他玩家的手牌（联合）
        self.other_hand_cards = []
        # 当前合法动作
        self.legal_actions = []
        # 最后一手牌
        self.last_move = []
        # 最近两手牌
        self.last_two_moves = []
        # 各位置最后出的牌
        self.last_move_dict = {}
        # 已出的牌
        self.played_cards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        # 所有玩家的手牌
        self.all_handcards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        # 最后出有效牌的玩家
        self.last_pid = None
        # 炸弹数量
        self.bomb_num = 0
        # 叫分
        self.bid_count = 0
        # 是否春天
        self.spring = False
        # 叫牌阶段是否结束
        self.bid_over = True
        # 出牌位置
        self.play_card_position = None


def create_infoset_for_act(player_position, player_hand_cards, three_landlord_cards, card_play_action_seq, bid_info):
    """
    为出牌API创建InfoSet对象
    """
    infoset = InfoSet(player_position)
    infoset.player_hand_cards = player_hand_cards
    infoset.three_landlord_cards = three_landlord_cards
    infoset.card_play_action_seq = card_play_action_seq
    infoset.bid_info = bid_info
    infoset.bid_over = True
    
    # 设置最后一手牌
    if card_play_action_seq:
        infoset.last_move = card_play_action_seq[-1]
        if len(card_play_action_seq) >= 2:
            infoset.last_two_moves = card_play_action_seq[-2:]
    
    # 计算合法动作
    infoset.legal_actions = _get_legal_card_play_actions(player_hand_cards, infoset.last_move)
    
    # 设置其他必要信息
    infoset.other_hand_cards = []  # 在实际应用中应该推断其他玩家的手牌
    
    # 设置所有玩家的手牌（简化处理）
    infoset.all_handcards[player_position] = player_hand_cards
    
    # 设置已出的牌和最后出牌的玩家（简化处理）
    positions = ['landlord', 'landlord_up', 'landlord_down']
    current_pos_index = positions.index(player_position)
    
    # 模拟出牌顺序
    for i, action in enumerate(card_play_action_seq):
        if action:  # 不是pass
            pos_index = (current_pos_index + i) % 3
            pos = positions[pos_index]
            infoset.played_cards[pos].extend(action)
            infoset.last_pid = pos
            infoset.last_move_dict[pos] = action
    
    # 设置剩余牌数
    for pos in positions:
        if pos == player_position:
            infoset.num_cards_left_dict[pos] = len(player_hand_cards)
        else:
            # 简化处理，假设其他玩家有15张牌
            infoset.num_cards_left_dict[pos] = 15
    
    return infoset


def create_infoset_for_bid(player_position, player_hand_cards, bid_info):
    """
    为叫地主API创建InfoSet对象
    """
    infoset = InfoSet(player_position)
    infoset.player_hand_cards = player_hand_cards
    infoset.bid_info = bid_info
    infoset.bid_over = False
    
    # 设置叫分的合法动作
    infoset.legal_actions = _get_legal_bid_actions(bid_info)
    
    # 设置所有玩家的手牌（简化处理）
    infoset.all_handcards[player_position] = player_hand_cards
    
    return infoset


def _get_legal_card_play_actions(player_hand_cards, last_move):
    """
    获取出牌的合法动作
    简化版本，实际应使用douzero/env/move_generator.py中的MovesGener类
    """
    try:
        from douzero.env.move_generator import MovesGener
        
        # 创建MovesGener对象
        mg = MovesGener(player_hand_cards)
        
        # 获取所有可能的动作
        all_moves = mg.gen_moves()
        
        if not last_move:
            # 如果是第一手牌，可以出任何牌
            legal_actions = all_moves
        else:
            # 如果不是第一手牌，需要根据上一手牌筛选合法动作
            # 这里需要实现根据last_move筛选合法动作的逻辑
            legal_actions = [[]]  # 暂时只返回"不出"
            last_move_type = _get_move_type(last_move)
            for move in all_moves:
                if _is_valid_response(move, last_move, last_move_type):
                    legal_actions.append(move)
        
        if not legal_actions:  # 如果没有合法动作，返回pass
            return [[]]
            
        return legal_actions
    except Exception as e:
        print(f"Error generating legal actions: {e}")
        # 如果出错，返回空列表和pass
        return [[]]


def _get_move_type(move):
    """
    获取出牌类型
    """
    if not move:
        return "pass"
    
    # 统计牌的数量
    counter = Counter(move)
    if len(move) == 2 and 20 in move and 30 in move:
        return "rocket"
    elif len(set(move)) == 1 and len(move) == 4:
        return "bomb"
    elif len(move) == 1:
        return "single"
    elif len(move) == 2 and len(set(move)) == 1:
        return "pair"
    elif len(move) == 3 and len(set(move)) == 1:
        return "triple"
    # 其他类型暂时都返回None
    return None


def _is_valid_response(response_move, last_move, last_move_type):
    """
    判断response_move是否是对last_move的合法响应
    """
    if not response_move:
        return True  # 可以选择不出
    
    response_type = _get_move_type(response_move)
    
    # 火箭可以打任何牌
    if response_type == "rocket":
        return True
    
    # 炸弹可以打非火箭的牌
    if response_type == "bomb" and last_move_type != "rocket":
        if last_move_type == "bomb":
            return max(response_move) > max(last_move)
        return True
    
    # 其他情况必须牌型相同，且大小更大
    if response_type == last_move_type:
        return max(response_move) > max(last_move)
    
    return False


def _get_legal_bid_actions(bid_info):
    """
    获取叫分的合法动作
    """
    legal_actions = []
    # 找出当前最高分
    max_bid = max([b for b in bid_info if b >= 0] + [0])
    
    # 可以叫比当前最高分更高的分数
    for i in range(max_bid + 1, 4):
        legal_actions.append([i])
    
    # 始终可以选择不叫
    legal_actions.append([0])
    
    return legal_actions


@app.route('/sandou_act', methods=['POST'])
def sandou_act():
    """
    出牌API
    """
    try:
        # 获取请求数据
        data = request.json
        
        # 验证密码
        if data.get('passwd') != PASSWD:
            return jsonify({'code': 1, 'message': '密码错误'})
        
        # 解析请求数据
        player_position = data.get('player_position')
        player_hand_cards = ast.literal_eval(data.get('player_hand_cards'))
        three_landlord_cards = ast.literal_eval(data.get('three_landlord_cards'))
        card_play_action_seq = ast.literal_eval(data.get('card_play_action_seq'))
        bid_info = ast.literal_eval(data.get('bid_info'))
        
        # 创建InfoSet
        infoset = create_infoset_for_act(
            player_position=player_position,
            player_hand_cards=player_hand_cards,
            three_landlord_cards=three_landlord_cards,
            card_play_action_seq=card_play_action_seq,
            bid_info=bid_info
        )
        
        # 根据玩家位置选择模型
        if player_position == 'landlord':
            model = landlord_model
        elif player_position == 'landlord_up':
            model = landlord_up_model
        elif player_position == 'landlord_down':
            model = landlord_down_model
        else:
            return jsonify({'code': 1, 'message': '玩家位置无效'})
        
        # 使用模型预测动作
        action = model.act(infoset)
        
        # 返回预测结果
        return jsonify({
            'code': 0,
            'action': json.dumps(action)
        })
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({'code': 1, 'message': str(e), 'traceback': traceback_str})


@app.route('/sandou_bid', methods=['POST'])
def sandou_bid():
    """
    叫地主API
    """
    try:
        # 获取请求数据
        data = request.json
        
        # 验证密码
        if data.get('passwd') != PASSWD:
            return jsonify({'code': 1, 'message': '密码错误'})
        
        # 解析请求数据
        player_position = data.get('player_position')
        player_hand_cards = ast.literal_eval(data.get('player_hand_cards'))
        bid_info = ast.literal_eval(data.get('bid_info'))
        
        # 创建InfoSet
        infoset = create_infoset_for_bid(
            player_position=player_position,
            player_hand_cards=player_hand_cards,
            bid_info=bid_info
        )
        
        # 使用叫牌模型预测动作
        action = bid_model.act(infoset)
        
        # 返回预测结果
        return jsonify({
            'code': 0,
            'action': json.dumps(action)
        })
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({'code': 1, 'message': str(e), 'traceback': traceback_str})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 