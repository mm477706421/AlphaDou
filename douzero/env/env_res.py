from collections import Counter
import numpy as np

from douzero.env.game import GameEnv

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective):
        """
        Objective is wp/adp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action


def get_obs_res(infoset, model_type="old"):
    if model_type == "general":
        if infoset.player_position not in ["landlord", "landlord_up", "landlord_down"]:
            raise ValueError('')
        return _get_obs_general(infoset, infoset.player_position)
    elif model_type == "resnet":
        if infoset.player_position not in ["landlord", "landlord_up", "landlord_down"]:
            raise ValueError('')
        return _get_obs_resnet(infoset, infoset.player_position)
    else:
        if infoset.player_position == 'landlord':
            return _get_obs_landlord(infoset)
        elif infoset.player_position == 'landlord_up':
            return _get_obs_landlord_up(infoset)
        elif infoset.player_position == 'landlord_down':
            return _get_obs_landlord_down(infoset)
        else:
            raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list, model_type="old"):
    """
    将动作序列转换为数组格式
    """
    if model_type == "general":
        position_map = {"landlord": 0, "landlord_up": 1, "landlord_down": 2}
        action_seq_array = np.ones((len(action_seq_list), 57)) * -1  # Default Value -1 for not using area
        for row, list_cards in enumerate(action_seq_list):
            if list_cards:
                action_seq_array[row, :54] = _cards2array(list_cards[1])
                for pos in position_map:
                    if list_cards[0] == pos:
                        action_seq_array[row, 54 + position_map[pos]] = 1
                    else:
                        action_seq_array[row, 54 + position_map[pos]] = 0
    elif model_type == "resnet":
        # 为resnet模型创建适当的输入维度
        action_seq_array = np.zeros((20, 54))  # 使用20个通道来表示动作序列
        for row, list_cards in enumerate(action_seq_list[:20]):  # 只取前20个动作
            if list_cards:
                if isinstance(list_cards, list) and len(list_cards) > 1:
                    action_seq_array[row, :] = _cards2array(list_cards[1])
                else:
                    action_seq_array[row, :] = _cards2array(list_cards)
    else:
        action_seq_array = np.zeros((len(action_seq_list), 54))
        for row, list_cards in enumerate(action_seq_list):
            if list_cards:
                action_seq_array[row, :] = _cards2array(list_cards[1])
        action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15, new_model=True):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if new_model:
        sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
    infoset.card_play_action_seq, 15, False), "old")
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': 'landlord',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs

def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
    infoset.card_play_action_seq, 15, False), "old")
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': 'landlord_up',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
    infoset.card_play_action_seq, 15, False), "old")
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': 'landlord_down',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_resnet(infoset, position=None):
    """
    获取观察空间的函数，添加异常处理
    """
    try:
        if position is None:
            position = infoset.player_position
        
        # 确保infoset具有所需的属性
        _ensure_infoset_attributes(infoset)
        
        num_legal_actions = len(infoset.legal_actions)
        my_handcards = _cards2array(infoset.player_hand_cards)
        other_handcards = _cards2array(infoset.other_hand_cards)
        three_landlord_cards = _cards2array(infoset.three_landlord_cards)

        # 获取已出的牌，如果不存在则使用空列表
        landlord_played_cards = _cards2array(
            infoset.played_cards.get('landlord', []))
        landlord_up_played_cards = _cards2array(
            infoset.played_cards.get('landlord_up', []))
        landlord_down_played_cards = _cards2array(
            infoset.played_cards.get('landlord_down', []))

        # 获取剩余牌数
        landlord_num_cards_left = _get_one_hot_array(
            infoset.num_cards_left_dict.get('landlord', 0), 20)
        landlord_up_num_cards_left = _get_one_hot_array(
            infoset.num_cards_left_dict.get('landlord_up', 0), 17)
        landlord_down_num_cards_left = _get_one_hot_array(
            infoset.num_cards_left_dict.get('landlord_down', 0), 17)

        # 处理card_play_action_seq，如果不存在则使用空列表
        card_play_action_seq = getattr(infoset, 'card_play_action_seq', [])
        processed_seq = _process_action_seq(card_play_action_seq, 60)
        action_seq = _action_seq_list2array(processed_seq, "resnet")

        # 构建基础特征
        base_features = [
            my_handcards,          # 1x54
            other_handcards,       # 1x54
            three_landlord_cards,  # 1x54
            landlord_played_cards, # 1x54
            landlord_up_played_cards,   # 1x54
            landlord_down_played_cards  # 1x54
        ]

        # 处理叫分信息
        bid_info = np.array(infoset.bid_info)
        bid_info_channels = []
        for i in range(3):
            channel = np.zeros(54)
            channel.fill(bid_info[i])
            bid_info_channels.append(channel)

        # 处理春天标志
        spring_channel = np.zeros(54)
        spring_channel.fill(1 if infoset.spring else 0)

        # 构建所有特征通道
        all_features = []
        
        # 添加基础特征 (6通道)
        for f in base_features:
            all_features.append(np.array(f).reshape(1, -1))
            
        # 添加叫分信息 (3通道)
        for c in bid_info_channels:
            all_features.append(np.array(c).reshape(1, -1))
            
        # 添加春天标志 (1通道)
        all_features.append(spring_channel.reshape(1, -1))
        
        # 添加动作序列 (20通道)
        all_features.extend([row.reshape(1, -1) for row in action_seq])
        
        # 添加填充通道 (10通道)
        for _ in range(10):
            all_features.append(np.zeros((1, 54)))

        # 堆叠所有特征
        z = np.vstack(all_features)

        # 确保z的通道数为40
        if z.shape[0] < 40:
            padding = np.zeros((40 - z.shape[0], 54))
            z = np.vstack((z, padding))
        elif z.shape[0] > 40:
            z = z[:40, :]

        # 为每个legal action创建一个batch
        z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)

        # 准备x_batch (18维特征)
        x_batch = np.hstack((
            np.repeat(bid_info[np.newaxis, :], num_legal_actions, axis=0),  # 3维叫分信息
            np.repeat(_get_one_hot_bomb(infoset.bomb_num)[np.newaxis, :], num_legal_actions, axis=0)  # 15维炸弹信息
        ))

        x_no_action = np.hstack((bid_info, _get_one_hot_bomb(infoset.bomb_num)))

        obs = {
            'position': position,
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
        }
        return obs
    except Exception as e:
        print(f"Error in _get_obs_resnet: {e}")
        # 返回一个基本的观察空间
        return _get_default_obs(infoset, position)


def _ensure_infoset_attributes(infoset):
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


def _get_default_obs(infoset, position=None):
    """
    返回一个基本的观察空间
    """
    if position is None:
        position = getattr(infoset, 'player_position', 'landlord')
    
    num_legal_actions = len(getattr(infoset, 'legal_actions', [[]]))
    
    # 创建基本的观察空间
    x_batch = np.zeros((num_legal_actions, 18), dtype=np.float32)  # 3(bid_info) + 15(bomb_num)
    z_batch = np.zeros((num_legal_actions, 40, 54), dtype=np.float32)  # 40 channels
    
    obs = {
        'position': position,
        'x_batch': x_batch,
        'z_batch': z_batch,
        'legal_actions': getattr(infoset, 'legal_actions', [[]]),
        'x_no_action': np.zeros(18, dtype=np.int8),
        'z': np.zeros((40, 54), dtype=np.int8),
    }
    return obs

def _get_obs_general(infoset, position):
    """
    获取通用观察空间,与InfoSet格式保持一致
    """
    num_legal_actions = len(infoset.legal_actions)
    
    # 基础特征 (手牌信息)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    # 位置编码 (3维)
    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    # 叫分和倍数信息
    bid_info = np.array(infoset.bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    # 地主牌信息
    three_landlord_cards = _cards2array(infoset.three_landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    # 最后出牌
    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    # 当前动作
    my_action_batch = np.zeros((num_legal_actions, 54))
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    # 剩余牌数信息
    num_cards_left = np.zeros((num_legal_actions, 54))  # 使用54维向量表示剩余牌数
    for pos, num in infoset.num_cards_left_dict.items():
        if num > 0:
            num_cards_left[:, :] += (num / 20.0)  # 归一化

    # 已出牌信息
    played_cards = {pos: _cards2array(cards) for pos, cards in infoset.played_cards.items()}
    played_cards_batch = np.zeros((num_legal_actions, 162))  # 3个位置的出牌信息拼接
    for i, pos in enumerate(['landlord', 'landlord_up', 'landlord_down']):
        played_cards_batch[:, i*54:(i+1)*54] = np.repeat(played_cards[pos][np.newaxis, :],
                                                        num_legal_actions, axis=0)

    # 炸弹数
    bomb_num = np.zeros(15)  # 15维向量表示炸弹数
    bomb_num[infoset.bomb_num] = 1
    bomb_num_batch = np.repeat(bomb_num[np.newaxis, :],
                              num_legal_actions, axis=0)

    # 组合所有特征
    x_batch = np.hstack((
        position_info_batch,          # 3
        my_handcards_batch,          # 54
        other_handcards_batch,       # 54
        three_landlord_cards_batch,  # 54
        last_action_batch,           # 54
        played_cards_batch,          # 162
        num_cards_left,              # 54
        bomb_num_batch,              # 15
        bid_info_batch,              # 12
        my_action_batch              # 54
    ))                               # 总计: 516维

    # 无动作特征
    x_no_action = np.hstack((
        position_info,
        my_handcards,
        other_handcards,
        three_landlord_cards,
        last_action,
        played_cards['landlord'],
        played_cards['landlord_up'],
        played_cards['landlord_down'],
        num_cards_left[0],
        bomb_num,
        bid_info
    ))

    # 动作序列特征
    z = _action_seq_list2array(
        _process_action_seq(infoset.card_play_action_seq, 32),
        "general"
    )
    z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)

    # 确保特征维度正确
    expected_x_dim = 1340  # 模型期望的输入维度
    current_x_dim = x_batch.shape[1]
    
    if current_x_dim < expected_x_dim:
        # 如果维度不足,添加padding
        padding = np.zeros((x_batch.shape[0], expected_x_dim - current_x_dim))
        x_batch = np.hstack((x_batch, padding))
        padding_no_action = np.zeros(expected_x_dim - x_no_action.shape[0])
        x_no_action = np.hstack((x_no_action, padding_no_action))
    elif current_x_dim > expected_x_dim:
        # 如果维度过大,截断
        x_batch = x_batch[:, :expected_x_dim]
        x_no_action = x_no_action[:expected_x_dim]

    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def gen_bid_legal_actions(player_id, bid_info):
    self_bid_info = bid_info[:, [(player_id - 1) % 3, player_id, (player_id + 1) % 3]]
    curr_round = -1
    for r in range(4):
        if -1 in self_bid_info[r]:
            curr_round = r
            break
    bid_actions = []
    if curr_round != -1:
        self_bid_info[curr_round] = [0, 0, 0]
        bid_actions.append(np.array(self_bid_info).flatten())
        self_bid_info[curr_round] = [0, 1, 0]
        bid_actions.append(np.array(self_bid_info).flatten())
    return np.array(bid_actions)


def _get_obs_for_bid(player_id, bid_info, hand_cards):
    all_cards = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
                 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 17, 17, 17, 17, 20, 30]
    num_legal_actions = 2
    my_handcards = _cards2array(hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    other_cards = []
    other_cards.extend(all_cards)
    for card in hand_cards:
        other_cards.remove(card)
    other_handcards = _cards2array(other_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    position_info = np.array([0, 0, 0])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_legal_actions = gen_bid_legal_actions(player_id, bid_info)
    bid_info = bid_legal_actions[0]
    bid_info_batch = bid_legal_actions

    multiply_info = np.array([0, 0, 0])
    multiply_info_batch = np.repeat(multiply_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    three_landlord_cards = _cards2array([])
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array([])
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j in range(2):
        my_action_batch[j, :] = _cards2array([])

    landlord_num_cards_left = _get_one_hot_array(0, 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(0, 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(0, 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array([])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array([])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array([])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(0)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((position_info_batch,
                         my_handcards_batch,
                         other_handcards_batch,
                         three_landlord_cards_batch,
                         last_action_batch,
                         landlord_played_cards_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_num_cards_left_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         bid_info_batch,
                         multiply_info_batch,
                         my_action_batch))
    x_no_action = np.hstack((position_info,
                             my_handcards,
                             other_handcards,
                             three_landlord_cards,
                             last_action,
                             landlord_played_cards,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_num_cards_left,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq([], 32))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': "",
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': bid_legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        "bid_info_batch": bid_info_batch.astype(np.int8),
        "multiply_info": multiply_info.astype(np.int8)
    }
    return obs


def _get_obs_for_multiply(position, bid_info, hand_cards, landlord_cards):
    all_cards = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
                 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 17, 17, 17, 17, 20, 30]
    num_legal_actions = 3
    my_handcards = _cards2array(hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    other_cards = []
    other_cards.extend(all_cards)
    for card in hand_cards:
        other_cards.remove(card)
    other_handcards = _cards2array(other_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_info = np.array(bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    multiply_info = np.array([0, 0, 0])
    multiply_info_batch = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

    three_landlord_cards = _cards2array(landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array([])
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j in range(num_legal_actions):
        my_action_batch[j, :] = _cards2array([])

    landlord_num_cards_left = _get_one_hot_array(0, 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(0, 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(0, 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array([])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array([])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array([])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(0)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((position_info_batch,
                         my_handcards_batch,
                         other_handcards_batch,
                         three_landlord_cards_batch,
                         last_action_batch,
                         landlord_played_cards_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_num_cards_left_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         bid_info_batch,
                         multiply_info_batch,
                         my_action_batch))
    x_no_action = np.hstack((position_info,
                             my_handcards,
                             other_handcards,
                             three_landlord_cards,
                             last_action,
                             landlord_played_cards,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_num_cards_left,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq([], 32))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': "",
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': multiply_info_batch,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        "bid_info": bid_info.astype(np.int8),
        "multiply_info_batch": multiply_info.astype(np.int8)
    }
    return obs
