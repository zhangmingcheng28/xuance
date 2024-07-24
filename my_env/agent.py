# -*- coding: utf-8 -*-
"""
@Time    : 6/23/2024 2:55 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""


class Agent:
    def __init__(self, agent_idx, start_pos, end_pos):
        self.agent_name = 'agent_%s' % agent_idx
        self.pos = start_pos
        self.ini_pos = start_pos
        self.pre_pos = None
        self.current_act = None
        self.pre_act = None
        self.vel = 1/9  # 1/9 nautical mile per sec, equals to 400 nautical mile per hour
        self.ref_line = None
        self.waypoints = None
        self.ini_heading = None
        self.heading = None
        self.pre_heading = None
        self.eta = None
        self.activation_flag = 0
        self.NMAC_radius = 5  # nautical mile in radius
        self.detectionRange = 150  # nautical mile
        self.probe_line = {}
        self.destination = end_pos
        self.destination_radius = 1
        self.reach_target = False
        self.bound_collision = False
        self.prd_collision = False
        self.drone_collision = False
        self.cloud_collision = False
        self.flight_data = []
