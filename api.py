from flask import Flask, request, jsonify
import torch
import numpy as np
import json
import ast
from douzero.evaluation.deep_agent import DeepAgent, SupervisedModel
from copy import deepcopy
from douzero.env import move_detector as md, move_selector as ms
from douzero.env.move_generator import MovesGener

# 导入game_eval中的InfoSet类
from game_eval import InfoSet, bombs, EnvCard2RealCard, RealCard2EnvCard

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


def create_infoset_for_act(player_position, player_hand_cards, three_landlord_cards, card_play_action_seq, bid_info):
    """
    为出牌API创建InfoSet对象，使用game_eval中的InfoSet类
    """
    infoset = InfoSet(player_position)
    infoset.player_hand_cards = player_hand_cards
    infoset.three_landlord_cards = three_landlord_cards
    infoset.bid_info = bid_info
    infoset.bid_over = True
    
    # 设置出牌历史
    infoset.card_play_action_seq = []
    positions = ['landlord', 'landlord_up', 'landlord_down']
    position_index = positions.index(player_position)
    
    # 处理出牌历史
    for i, action in enumerate(card_play_action_seq):
        actor_index = (position_index - len(card_play_action_seq) + i) % 3
        actor = positions[actor_index]
        infoset.card_play_action_seq.append((actor, action))
    
    # 设置最后一手牌
    infoset.last_move = []
    if infoset.card_play_action_seq:
        if len(infoset.card_play_action_seq[-1][1]) == 0 and len(infoset.card_play_action_seq) > 1:
            infoset.last_move = infoset.card_play_action_seq[-2][1]
        else:
            infoset.last_move = infoset.card_play_action_seq[-1][1]
    
    # 设置最近两手牌
    infoset.last_two_moves = [[], []]
    for card in infoset.card_play_action_seq[-2:]:
        infoset.last_two_moves.insert(0, card[1])
        infoset.last_two_moves = infoset.last_two_moves[:2]
    
    # 设置已出的牌和最后出牌的玩家
    infoset.played_cards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
    infoset.last_move_dict = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
    
    for pos, action in infoset.card_play_action_seq:
        if action:  # 不是pass
            infoset.played_cards[pos].extend(action)
            infoset.last_move_dict[pos] = action
            infoset.last_pid = pos
    
    # 设置剩余牌数
    infoset.num_cards_left_dict = {'landlord': 20, 'landlord_up': 17, 'landlord_down': 17}
    for pos, actions in infoset.played_cards.items():
        infoset.num_cards_left_dict[pos] -= len(actions)
    
    # 设置当前玩家剩余手牌
    infoset.num_cards_left_dict[player_position] = len(player_hand_cards)
    
    # 计算炸弹数量
    infoset.bomb_num = 0
    for _, action in infoset.card_play_action_seq:
        if action in bombs:
            infoset.bomb_num += 1
    
    # 设置所有玩家的手牌（简化处理）
    infoset.all_handcards = {pos: [] for pos in positions}
    infoset.all_handcards[player_position] = player_hand_cards
    
    # 计算其他玩家的手牌（简化处理）
    infoset.other_hand_cards = []
    
    # 获取合法动作
    legal_actions = get_legal_card_play_actions(player_hand_cards, infoset.last_move)
    infoset.legal_actions = legal_actions
    
    return infoset


def create_infoset_for_bid(player_position, player_hand_cards, bid_info):
    """
    为叫地主API创建InfoSet对象，使用game_eval中的InfoSet类
    """
    # 转换玩家位置到bid_infoset的格式
    if player_position == 'landlord':
        bid_position = 'first'
    elif player_position == 'landlord_down':
        bid_position = 'second'
    elif player_position == 'landlord_up':
        bid_position = 'third'
    else:
        bid_position = player_position
    
    infoset = InfoSet(bid_position)
    infoset.player_hand_cards = player_hand_cards
    infoset.bid_info = bid_info
    infoset.bid_over = False
    
    # 设置叫分的合法动作
    bid_count = max([b for b in bid_info if b >= 0] + [0])
    infoset.bid_count = bid_count
    
    legal_actions = get_legal_bid_actions(bid_count)
    infoset.legal_actions = legal_actions
    
    return infoset


def get_legal_card_play_actions(player_hand_cards, rival_move):
    """
    获取出牌的合法动作，使用game_eval中的逻辑
    """
    mg = MovesGener(player_hand_cards)
    
    if not rival_move:
        return mg.gen_moves()
    
    rival_type = md.get_move_type(rival_move)
    rival_move_type = rival_type['type']
    rival_move_len = rival_type.get('len', 1)
    moves = list()
    
    if rival_move_type == md.TYPE_0_PASS:
        moves = mg.gen_moves()
    
    elif rival_move_type == md.TYPE_1_SINGLE:
        all_moves = mg.gen_type_1_single()
        moves = ms.filter_type_1_single(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_2_PAIR:
        all_moves = mg.gen_type_2_pair()
        moves = ms.filter_type_2_pair(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_3_TRIPLE:
        all_moves = mg.gen_type_3_triple()
        moves = ms.filter_type_3_triple(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_4_BOMB:
        all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
        moves = ms.filter_type_4_bomb(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_5_KING_BOMB:
        moves = []
    
    elif rival_move_type == md.TYPE_6_3_1:
        all_moves = mg.gen_type_6_3_1()
        moves = ms.filter_type_6_3_1(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_7_3_2:
        all_moves = mg.gen_type_7_3_2()
        moves = ms.filter_type_7_3_2(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
        all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
        moves = ms.filter_type_8_serial_single(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
        all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
        moves = ms.filter_type_9_serial_pair(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
        all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
        moves = ms.filter_type_10_serial_triple(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_11_SERIAL_3_1:
        all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
        moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_12_SERIAL_3_2:
        all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
        moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_13_4_2:
        all_moves = mg.gen_type_13_4_2()
        moves = ms.filter_type_13_4_2(all_moves, rival_move)
    
    elif rival_move_type == md.TYPE_14_4_22:
        all_moves = mg.gen_type_14_4_22()
        moves = ms.filter_type_14_4_22(all_moves, rival_move)
    
    if rival_move_type not in [md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
        moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
    
    if len(rival_move) != 0:  # rival_move is not 'pass'
        moves = moves + [[]]
    
    for m in moves:
        m.sort()
    
    return moves


def get_legal_bid_actions(bid_count):
    """
    获取叫分的合法动作，使用game_eval中的逻辑
    """
    if bid_count == 0:
        return [[0], [1], [2], [3]]
    elif bid_count == 1:
        return [[0], [2], [3]]
    elif bid_count == 2:
        return [[0], [3]]
    else:
        return [[0]]


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