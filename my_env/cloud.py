# -*- coding: utf-8 -*-
"""
@Time    : 6/27/2024 6:25 PM
@Author  : Mingcheng & Bizhao
@FileName:
@Description:
@Package dependency:
"""

class cloud_agent:
    def __init__(self, cloud_idx):
        self.agent_name = 'cloud_%s' % cloud_idx
        self.ini_pos = None
        self.pos = None
        self.pre_pos = None
        self.cloud_actual_cur_shape = None
        self.cloud_actual_previous_shape = None
        self.vel = 50/3600  # assume 50 nautical mile per hour
        self.radius = 8  # nautical mile in radius
        self.goal = None
        self.reach_target = False
        self.x_fact = 3
        self.y_fact = 2



