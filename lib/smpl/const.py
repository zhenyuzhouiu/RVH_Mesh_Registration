"""
some constants for smpl and smplh models
created by Xianghui, 12, January, 2022
"""

# related to smpl and smplh parameter count
SMPL_POSE_PRAMS_NUM = 72  # for the human body, it will use 24 key points to represent the pose
SMPLH_POSE_PRAMS_NUM = 156  # it includes the parameters of two hands
SMPLH_HANDPOSE_START = 66 # hand pose start index for smplh
NUM_BETAS = 10

# split smplh
GLOBAL_POSE_NUM = 3
BODY_POSE_NUM = 63  # TODO by Zhenyu: delete 3 joint keypoints (head+2 hands)?
HAND_POSE_NUM = 90  # 45 for each hand
TOP_BETA_NUM = 2

# split smpl
SMPL_HAND_POSE_NUM=6

