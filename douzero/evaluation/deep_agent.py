import torch
import numpy as np
import os
from douzero.env.env import get_obs
from douzero.env.env_douzero import get_obs_douzero
from douzero.env.env_res import _get_obs_resnet
from baseline.SLModel.BidModel import Net2 as Net
from collections import Counter


def _load_model(position, model_path, model_type):
    from douzero.dmc.models import model_dict, model_dict_douzero
    if model_type == "test":
        model = model_dict_douzero[position]()
    elif model_type == "best":
        from douzero.dmc.models_res import model_dict_resnet
        model = model_dict_resnet[position]()
    else:
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        if "test" in model_path:
            self.model_type = "test"
        elif "best" in model_path:
            self.model_type = "best"
        else:
            self.model_type = "new"
        self.model = _load_model(position, model_path, self.model_type)
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                            13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}
        
    def check_no_bombs(self, cards):
        card_counts = Counter(cards)
        for count in card_counts.values():
            if count == 4:
                return False

        if 20 in card_counts and 30 in card_counts:
            return False

        return True


    def act(self, infoset):
        try:
            # 确保infoset具有所需的属性
            self._ensure_infoset_attributes(infoset)
            
            if self.model_type == "test":
                obs = get_obs_douzero(infoset)
            elif self.model_type == "best":
                obs = _get_obs_resnet(infoset, infoset.player_position)
            else:
                obs = get_obs(infoset, bid_over=infoset.bid_over, new_model=True)

            z_batch = torch.from_numpy(obs['z_batch']).float()
            x_batch = torch.from_numpy(obs['x_batch']).float()
            if torch.cuda.is_available():
                z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
            if self.model_type != 'new':
                y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
            else:
                win_rate, win, lose = self.model.forward(z_batch, x_batch, return_value=True)['values']
                if infoset.player_position in ["landlord", "landlord_up", "landlord_down"]:
                    _win_rate = (win_rate + 1) / 2
                    y_pred = _win_rate * win + (1. - _win_rate) * lose
                    _win_rate = _win_rate.detach().cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()
                    if self.check_no_bombs(infoset.player_hand_cards) and infoset.spring is False and self.check_no_bombs(
                            infoset.other_hand_cards):
                        best_action_index = np.argmax(_win_rate, axis=0)[0]
                        best_action = infoset.legal_actions[best_action_index]
                    else:
                        y_pred = y_pred.flatten()
                        _win_rate = _win_rate.flatten()
                        max_adp = np.max(y_pred)
                        if max_adp >= 0:
                            min_threshold = max_adp * 0.95
                        else:
                            min_threshold = max_adp * 1.05
                        valid_indices = np.where(y_pred >= min_threshold)[0]
                        best_action_index = valid_indices[np.argmax(_win_rate[valid_indices])]
                        best_action = infoset.legal_actions[best_action_index]
                    return best_action
                else:
                    y_pred = win_rate[:, :1] * win + win_rate[:, 1:2] * lose
            y_pred = y_pred.detach().cpu().numpy()

            best_action_index = np.argmax(y_pred, axis=0)[0]
            best_action = infoset.legal_actions[best_action_index]
            return best_action
        except Exception as e:
            print(f"Error in DeepAgent.act: {e}")
            # 如果出错，返回第一个合法动作或者空动作
            if infoset.legal_actions:
                return infoset.legal_actions[0]
            return []
    
    def _ensure_infoset_attributes(self, infoset):
        """确保InfoSet对象具有所需的所有属性"""
        # 必需的属性列表
        required_attrs = [
            'player_position', 'player_hand_cards', 'num_cards_left_dict',
            'three_landlord_cards', 'bid_info', 'card_play_action_seq',
            'other_hand_cards', 'legal_actions', 'last_move', 'played_cards',
            'all_handcards', 'bomb_num', 'spring', 'bid_over'
        ]
        
        # 检查并设置缺失的属性
        for attr in required_attrs:
            if not hasattr(infoset, attr):
                if attr == 'num_cards_left_dict':
                    setattr(infoset, attr, {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0})
                elif attr == 'played_cards':
                    setattr(infoset, attr, {'landlord': [], 'landlord_up': [], 'landlord_down': []})
                elif attr == 'all_handcards':
                    setattr(infoset, attr, {'landlord': [], 'landlord_up': [], 'landlord_down': []})
                elif attr in ['player_hand_cards', 'three_landlord_cards', 'other_hand_cards', 'card_play_action_seq', 'legal_actions', 'last_move']:
                    setattr(infoset, attr, [])
                elif attr == 'bid_info':
                    setattr(infoset, attr, [-1, -1, -1])
                elif attr in ['bomb_num']:
                    setattr(infoset, attr, 0)
                elif attr in ['spring', 'bid_over']:
                    setattr(infoset, attr, False)
                else:
                    setattr(infoset, attr, None)


class SupervisedModel:

    def __init__(self):
        self.net = Net()
        self.net.eval()
        if torch.cuda.is_available():
            self.gpu = True
        else:
            self.gpu = False
        if self.gpu:
            self.net = self.net.cuda()
        if os.path.exists("baseline/SLModel/bid_weights_new.pkl"):
            if torch.cuda.is_available():
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl'))
            else:
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl', map_location=torch.device("cpu")))

    def RealToOnehot(self, cards):
        Onehot = torch.zeros((4, 15))
        m = 0
        for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
            Onehot[:cards.count(i), m] = 1
            m += 1
        return Onehot

    def predict_score(self, cards):
        input = RealToOnehot(cards)
        input = torch.flatten(input)
        input = input.unsqueeze(0)
        result = self.net(input)
        return result[0].item()

    def act(self, infoset):
        try:
            legal_action = infoset.legal_actions
            obs = torch.flatten(self.RealToOnehot(infoset.player_hand_cards))
            if self.gpu:
                obs = obs.cuda()
            predict = self.net.forward(obs.unsqueeze(0))
            one = -0.1
            two = 0
            three = 0.1
            if predict > three and ([3] in legal_action):
                return [3]
            elif predict > two and ([2] in legal_action):
                return [2]
            elif predict > one and ([1] in legal_action):
                return [1]
            else:
                return [0]
        except Exception as e:
            print(f"Error in SupervisedModel.act: {e}")
            # 如果出错，返回不叫
            return [0]

def RealToOnehot(cards):
    RealCard2EnvCard = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                        '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                        'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14}
    cards = [RealCard2EnvCard[c] for c in cards]
    Onehot = torch.zeros((4,15))
    for i in range(0, 15):
        Onehot[:cards.count(i),i] = 1
    return Onehot