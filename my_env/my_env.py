import argparse
from utilities import *
import numpy as np
from copy import deepcopy
import random
import math
from agent import Agent
from gym.spaces import Box, Discrete
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawMultiAgentEnv, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IPPO_Agents
from xuance.torch.agents import MAPPO_Agents
from shapely.strtree import STRtree
from cloud import cloud_agent
from shapely.affinity import scale
from shapely.geometry import Polygon, Point, LineString


class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        self.env_id = env_config.env_id
        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(-np.inf, np.inf, shape=[25, ])
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[25, ]) for agent in self.agents}
        self.action_space = {agent: Discrete(n=5) for agent in self.agents}
        self.max_episode_steps = 200
        self._current_step = 0
        #  ------------ start of my own attributes -------------------
        self.norm_tool = None
        self.prd_polygons = None
        self.boundaries = None
        self.potential_ref_line = None
        self.cloud_config = None
        self.time_step = 15  # in seconds, represent simulation time step
        self.my_agent_self_data = {}
        #  ------------ end of my own attributes -------------------

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def reset(self):
        print("reset called")
        cloud_0 = [75, 50, 200, 50]
        cloud_1 = [140, 125, 200, 125]
        all_clouds = [cloud_0, cloud_1]
        cloud_config = []
        # ---------- cloud config -------------
        for cloud_idx, cloud_setting in enumerate(all_clouds):
            cloud_a = cloud_agent(cloud_idx)
            cloud_a.pos = Point(cloud_setting[0], cloud_setting[1])
            cloud_a.ini_pos = cloud_a.pos
            cloud_a.cloud_actual_cur_shape = scale(cloud_a.pos.buffer(cloud_a.radius), xfact=cloud_a.x_fact,
                                                   yfact=cloud_a.y_fact)  # width, along x-axis, length become 2, height, along y-axis, radius length become 1
            cloud_a.goal = Point(cloud_setting[2], cloud_setting[3])
            cloud_config.append(cloud_a)

        # --------- PRD config --------------
        PRD_1 = [(10, 10), (10, 60), (40, 60), (40, 10), (10, 10)]
        # PRD_2 = [(100, 100), (100, 150), (150, 150), (150, 100), (100, 100)]
        PRD_set = [PRD_1]
        prd_polygons = [Polygon(PRD) for PRD in PRD_set]
        # --------- end of PRD config --------------

        # --------- bound config ------------------
        x_left = LineString([(0, 0), (0, 200)])
        x_right = LineString([(200, 0), (200, 200)])
        y_top = LineString([(0, 200), (200, 200)])
        y_bottom = LineString([(0, 0), (200, 0)])
        bounds = [x_left, x_right, y_top, y_bottom]
        # add boundary lines to cloud
        # initialize normalizer
        self.norm_tool = NormalizeData([0, 200], [0, 200])
        # --------- end of bound config -------------

        # --------- potential reference line (only for display not involved in training)---------
        # line_w1 = LineString([(0, 150), (75, 75), (86.90, 0)])
        line_w1 = LineString([(0, 150), (60, 75), (86.90, 0)])
        line_w2 = LineString([(43.75, 200), (75, 75), (86.9, 0)])
        # line_0 = LineString([(110, 12), (185, 160)])
        line_w3 = LineString([(103.96, 0), (205.77, 200)])
        # line_1 = LineString([(140, 13), (160, 190)])
        line_w4 = LineString([(138.41, 0), (161.13, 200)])
        potential_RF = [line_w1, line_w2, line_w3, line_w4]
        # ---------- end of potential reference line ---------
        self.prd_polygons = prd_polygons
        self.boundaries = bounds
        self.potential_ref_line = potential_RF
        self.cloud_config = cloud_config
        self.cloud_movement = [[cloudAgent.pos for cloudAgent in cloud_config]]

        star_map_list = {
            # 'star1': [(7.071, 142.929), (75, 75), (85.333, 9.876)],
            'star1': [(7.071, 142.929), (60, 75), (79.73, 20)],
            'star2': [(46.176, 190.299), (75, 75), (85.334, 15)],
            'star3': [(108.497, 8.911), (185, 160)],
            'star4': [(139.538, 9.936), (160.002, 190.064)]
        }
        agent_ETA = [random.randint(0, self.num_agents) * 5 for a in range(self.num_agents)]
        for agent_idx in range(self.num_agents):
            keys = list(star_map_list.keys())  # Get a list of STAR-keys
            # random_star_key = random.choice(keys)  # Randomly choose a STAR-key
            if agent_idx == 0:
                random_star_key = keys[0]
            elif agent_idx == 1:
                random_star_key = keys[2]
            elif agent_idx == 2:
                random_star_key = keys[3]
            else:
                pass
            start_pos = np.array(star_map_list[random_star_key][0])
            end_pos = np.array(star_map_list[random_star_key][-1])
            agent_obj = Agent(agent_idx, start_pos, end_pos)
            # agent_obj.heading = np.arctan2([end_pos[1]-start_pos[1]], [end_pos[0]-start_pos[0]]) * 180 / np.pi  # in degree to destination
            agent_obj.ref_line = LineString(star_map_list[random_star_key])
            # agent_obj.ref_line = LineString([start_pos, np.array([75, 75]), end_pos])
            agent_obj.waypoints = [np.array(coord) for coord in list(agent_obj.ref_line.coords)[1:-1]]
            if len(agent_obj.waypoints) == 0:
                agent_obj.heading = np.arctan2([end_pos[1] - start_pos[1]],
                                               [end_pos[0] - start_pos[0]]) * 180 / np.pi  # in degree
                agent_obj.ini_heading = np.arctan2([end_pos[1] - start_pos[1]],
                                               [end_pos[0] - start_pos[0]]) * 180 / np.pi  # in degree
            else:
                agent_obj.heading = np.arctan2([agent_obj.waypoints[0][1] - start_pos[1]],
                                               [agent_obj.waypoints[0][0] - start_pos[0]]) * 180 / np.pi  # in degree
                agent_obj.ini_heading = np.arctan2([agent_obj.waypoints[0][1] - start_pos[1]],
                                               [agent_obj.waypoints[0][0] - start_pos[0]]) * 180 / np.pi  # in degree
            agent_obj.eta = agent_ETA[agent_idx]
            self.my_agent_self_data[agent_obj.agent_name] = agent_obj

        observation = self.get_cur_obs(self.agents)

        # observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        info = {}
        self._current_step = 0
        return observation, info

    def step(self, action_dict):
        self._current_step = self._current_step + 1  # this is the original time clock
        # ---------- start of my own step function ----------------
        action_pool = [-30, -15, 0, 15, 30]
        # move aircrafts
        for agent_idx, agent in enumerate(self.agents):
            agent_action_choice = action_pool[action_dict[agent]]
            my_env_agent = self.my_agent_self_data[agent]
            # --------- this is to ensure all attributes are filled --------- #
            my_env_agent.pre_act = deepcopy(my_env_agent.current_act)
            my_env_agent.pre_pos = deepcopy(my_env_agent.pos)
            my_env_agent.pre_heading = deepcopy(my_env_agent.heading)
            # --------- end of fill attributes are filled --------- #
            # aircraft will only activate if its eta equals to the current episode step.
            # if my_env_agent.eta == self._current_step-1:  # we need to -1, as the simulation step starts from 1. So after eta, agent always active
            if (my_env_agent.eta == self._current_step-1) and (my_env_agent.reach_target == False): # The activation flag will only change when agent does not reach their goal.
                my_env_agent.activation_flag = 1
            if my_env_agent.activation_flag == 1:
                # ensure we update the previous heading, position, action
                next_heading = my_env_agent.heading + agent_action_choice
                new_hx = my_env_agent.vel * math.cos(math.radians(next_heading))
                new_hy = my_env_agent.vel * math.sin(math.radians(next_heading))
                delta_x = new_hx * self.time_step
                delta_y = new_hy * self.time_step
                new_px = my_env_agent.pos[0] + delta_x
                new_py = my_env_agent.pos[1] + delta_y

                # update agent heading, position, and action
                my_env_agent.heading = next_heading
                my_env_agent.pos = np.array([new_px, new_py])
                my_env_agent.current_act = agent_action_choice
                my_env_agent.flight_data.append([my_env_agent.pos, my_env_agent.heading])  # Record the new position and heading

        # move all clouds based on their preset speed and vectors
        self.move_clouds()
        self.cloud_movement.append([cloudAgent.pos for cloudAgent in self.cloud_config])
        # get the updated actions
        observation = self.get_cur_obs(self.agents)

        # observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        rewards, terminated = self.obtain_reward()
        # rewards = {agent: np.random.random() for agent in self.agents}
        # terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return

    def move_clouds(self):
        for cloud_agent in self.cloud_config:
            # load cloud's previous position
            cloud_agent.pre_pos = deepcopy(cloud_agent.pos)
            cloud_agent.cloud_actual_previous_shape = deepcopy(cloud_agent.cloud_actual_cur_shape)

            # find cloud's next position
            cloud_start_pos = np.array([cloud_agent.pos.x, cloud_agent.pos.y])
            cloud_target = np.array([cloud_agent.goal.x, cloud_agent.goal.y])
            next_position = calculate_next_position(cloud_start_pos, cloud_target, cloud_agent.vel, self.time_step)

            # move cloud to new position
            cloud_agent.pos = Point(next_position[0], next_position[1])
            cloud_agent.cloud_actual_cur_shape = scale(cloud_agent.pos.buffer(cloud_agent.radius), xfact=cloud_agent.x_fact, yfact=cloud_agent.y_fact)

    def obtain_reward(self):
        # crash_penalty = 100
        crash_penalty = 200
        reaching_reward = 200
        wp_reach_reward = 0  # only appear once when agent first reach
        step_reward = {}
        done = {}
        reach_goal = [0 for _ in range(len(self.agents))]
        collision_indication = [0 for _ in range(len(self.agents))]
        for agent_idx, agent in enumerate(self.agents):  # loop through all agents, check if there is any crash case
            my_env_agent = self.my_agent_self_data[agent]
            step_reward[my_env_agent.agent_name] = 0  # initialize step reward for each agent
            host_pass_line = LineString([my_env_agent.pre_pos, my_env_agent.pos])
            host_passed_volume = host_pass_line.buffer(my_env_agent.NMAC_radius, cap_style=1)
            host_circle = Point(my_env_agent.pos[0], my_env_agent.pos[1]).buffer(my_env_agent.NMAC_radius, cap_style='round')
            target_circle = Point(my_env_agent.destination[0], my_env_agent.destination[1]).buffer(my_env_agent.destination_radius, cap_style='round')
            if len(my_env_agent.waypoints) > 0:
                wp_circle = Point(my_env_agent.waypoints[0][0], my_env_agent.waypoints[0][1]).buffer(my_env_agent.destination_radius, cap_style='round')

                # check if host drone has reached its waypoints
                if host_circle.intersects(wp_circle) or host_circle.within(wp_circle) or host_circle.overlaps(wp_circle):
                    print("{} reaches its waypoint".format(my_env_agent.agent_name))
                    step_reward[my_env_agent.agent_name] = step_reward[my_env_agent.agent_name] + wp_reach_reward
                    del my_env_agent.waypoints[0]

            # check of host drone reaches the goal
            if host_circle.intersects(target_circle) or host_circle.within(target_circle) or host_circle.overlaps(target_circle):
                my_env_agent.reach_target = True
                print("{} reaches its goal at step {}".format(my_env_agent.agent_name, self._current_step))


            # check whether crash into boundaries
            conflicting_lines = check_line_circle_conflict(host_circle, self.boundaries)
            if len(conflicting_lines) > 0:
                my_env_agent.bound_collision = True
                print("{} conflict with boundaries at step {}".format(my_env_agent.agent_name, self._current_step))


            # # check whether crash into polygons
            # conflicting_polygons = polygons_own_circle_conflict(host_circle, self.prd_polygons)
            # if len(conflicting_polygons) > 0:
            #     my_env_agent.prd_collision = True
            #     print("{} conflict with PRD".format(my_env_agent.agent_name))

            # check whether crash into clouds
            for clound in self.cloud_config:
                # cloud_area_moved = estimated_area_swap_by_arbitary_cloud(clound)  # area swapped by cloud there is error
                conflicting_cloud = polygons_single_cloud_conflict(host_circle, clound.cloud_actual_cur_shape)
                # conflicting_cloud = polygons_single_cloud_conflict(host_circle, cloud_area_moved)
                if len(conflicting_cloud) > 0:
                    my_env_agent.cloud_collision = True
                    print("{} conflict with cloud".format(my_env_agent.agent_name))
                    break

            # check whether current drone crash into other drones
            for other_agents in self.agents:
                my_env_other_agent = self.my_agent_self_data[other_agents]
                if my_env_agent.agent_name == my_env_other_agent.agent_name:
                    continue
                other_agent_pass_line = LineString([my_env_other_agent.pre_pos, my_env_other_agent.pos])
                other_agent_pass_volume = other_agent_pass_line.buffer(my_env_other_agent.NMAC_radius, cap_style=1)
                if between_polygon_conflict(host_passed_volume, other_agent_pass_volume):
                    my_env_agent.drone_collision = True
                    print(" {} hit {}".format(my_env_agent.agent_name, my_env_other_agent.agent_name))
                    break

            # assign reward and done for current agent
            if my_env_agent.drone_collision or my_env_agent.cloud_collision or my_env_agent.prd_collision or my_env_agent.bound_collision:
                step_reward[my_env_agent.agent_name] = - crash_penalty
                collision_indication[agent_idx] = 1
                # done[my_env_agent.agent_name] = 1
            else:
                # a normal step taken
                # distance from initial point to current aircraft position
                d_i_to_c = np.linalg.norm(my_env_agent.ini_pos - my_env_agent.pos)
                # distance from current aircraft position to final goal
                d_c_to_f = np.linalg.norm(my_env_agent.pos - my_env_agent.destination)
                # distance from initial point to final goal
                d_i_to_f = np.linalg.norm(my_env_agent.ini_pos - my_env_agent.destination)
                # dist_penaty = - (((d_i_to_c+d_c_to_f) / d_i_to_f)-1)*2
                dist_penaty = - (d_c_to_f / d_i_to_f) * 1.6
                # dist_penaty = - (((d_i_to_c+d_c_to_f) / d_i_to_f))**3

                # ----------- start of prob penalty -------------------
                prob_penalty = 0
                # prob_dist_list = [value[-1] for key, value in my_env_agent.probe_line.items()]
                # if min(prob_dist_list) >= 30:
                #     prob_penalty = 0
                # else:
                #     m = (0 - 1) / (30 - my_env_agent.NMAC_radius)
                #     prob_penalty = m*min(prob_dist_list)
                # ---------- end of prob penalty ------------------------

                step_reward[my_env_agent.agent_name] = step_reward[my_env_agent.agent_name] + dist_penaty + prob_penalty
                done[my_env_agent.agent_name] = 0

            # if my_env_agent.reach_target:  # load goal reaching score only once.
            if my_env_agent.reach_target and my_env_agent.activation_flag == 1:  # load goal reaching score only once.
                step_reward[my_env_agent.agent_name] = reaching_reward
                reach_goal[agent_idx] = 1
                # done[my_env_agent.agent_name] = 1
                my_env_agent.activation_flag = 0  # when drone reached no need to do any changes to the drone

        # use to fill the done indicator, possible to override the previous done (for normal step)
        if any(collision_indication):
            for agent_idx, agent in enumerate(self.agents):
                my_env_agent = self.my_agent_self_data[agent]
                done[my_env_agent.agent_name] = 1
        elif all(reach_goal):
            for agent_idx, agent in enumerate(self.agents):
                my_env_agent = self.my_agent_self_data[agent]
                done[my_env_agent.agent_name] = 1
        else:
            pass

        return step_reward, done

    def get_cur_obs(self, my_agents):
        observation = {}
        for agent_name in my_agents:
            my_agent_data = self.my_agent_self_data[agent_name]

            # look for nearest neighbour
            # region  ---- start of looking for nearest neighbor ----
            nearest_neigh_name = None
            shortest_neigh_dist = math.inf
            for other_agent_name in my_agents:
                if other_agent_name == agent_name:
                    continue # same agent we don't consider
                other_agent_data = self.my_agent_self_data[other_agent_name]
                diff_dist_vec = my_agent_data.pos - other_agent_data.pos  # host pos vector - intruder pos vector
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)
                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    nearest_neigh_name = other_agent_name
            # endregion ---- end of start of looking for nearest neighbor ----

            # region  ---- start of radar creation (only detect surrounding obstacles) ----
            drone_ctr = Point(my_agent_data.pos)
            # use centre point as start point
            st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            radar_dist = my_agent_data.detectionRange
            distances = {}
            radar_info = {}
            ed_points = {}
            line_collection = []  # a collection of all 20 radar's prob
            for point_deg, point_pos in st_points.items():
                # Create a line segment from the circle's center
                end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
                end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))
                end_point = Point(end_x, end_y)

                # current radar prob heading  # same as point-deg???
                cur_prob_heading = np.arctan2([end_y-my_agent_data.pos[1]], [end_x-my_agent_data.pos[0]])

                # Create the LineString from the start point to the end point
                cur_host_line = LineString([point_pos, end_point])
                line_collection.append(cur_host_line)

                # initialize minimum intersection point with end_point
                ed_points[point_deg] = end_point
                min_intersection_pt = end_point
                sensed_shortest_dist = cur_host_line.length
                distances[point_deg] = sensed_shortest_dist

                # check if line intersect with any boundaries
                for i, line in enumerate(self.boundaries):
                    if cur_host_line.intersects(line):
                        intersection_point = cur_host_line.intersection(line)
                        dist_to_intersection = LineString([point_pos, intersection_point]).length
                        if dist_to_intersection < sensed_shortest_dist:
                            # update global minimum end point and distance
                            ed_points[point_deg] = intersection_point
                            min_intersection_pt = intersection_point
                            sensed_shortest_dist = dist_to_intersection
                            distances[point_deg] = sensed_shortest_dist

                # check if line intersect with any clouds
                # initialize cloud context nearest nearest distance and point
                cloud_nearest_intersection_point = None
                cloud_nearest_distance = math.inf
                for cloud_obj in self.cloud_config:
                    clound_boundary = cloud_obj.cloud_actual_cur_shape.boundary
                    if cur_host_line.intersects(clound_boundary):
                        cloud_intersection_points = cur_host_line.intersection(clound_boundary)
                        if cloud_intersection_points.geom_type == 'MultiPoint':
                            for point in cloud_intersection_points.geoms:
                                distance = LineString([point_pos, point]).length
                                if distance < cloud_nearest_distance:
                                    cloud_nearest_distance = distance
                                    cloud_nearest_intersection_point = point
                        elif cloud_intersection_points.geom_type == 'Point':
                            distance = LineString([point_pos, cloud_intersection_points]).length
                            if distance < cloud_nearest_distance:
                                cloud_nearest_distance = distance
                                cloud_nearest_intersection_point = cloud_intersection_points

                # now compare the nearest distance in cloud context with the nearest distance in previous shortest
                if cloud_nearest_distance < sensed_shortest_dist:
                    # update global minimum end point and distance
                    ed_points[point_deg] = cloud_nearest_intersection_point
                    min_intersection_pt = cloud_nearest_intersection_point
                    sensed_shortest_dist = cloud_nearest_distance
                    distances[point_deg] = sensed_shortest_dist

                # all condition have check new we fill in the radar data
                radar_info[point_deg] = [min_intersection_pt, sensed_shortest_dist]
            # endregion ---- end of radar creation (only detect surrounding obstacles) ----

            # load radar_info to current agent
            my_agent_data.probe_line = radar_info

            norm_pos = self.norm_tool.nmlz_pos(my_agent_data.pos)
            norm_destination = self.norm_tool.nmlz_pos(my_agent_data.destination)
            norm_shortest_neigh_dist = shortest_neigh_dist / my_agent_data.detectionRange

            # radar_distance_list = [value for value in distances.values()]
            radar_distance_list = [value / my_agent_data.detectionRange for value in distances.values()]  # with normalization

            norm_obs_list = [norm_pos[0], norm_pos[1], my_agent_data.heading[0], my_agent_data.activation_flag, norm_shortest_neigh_dist, norm_destination[0], norm_destination[1]]
            combine_obs = norm_obs_list + radar_distance_list
            observation[agent_name] = np.array([combine_obs])

            # observation[agent_name] = np.array([my_agent_data.pos[0], my_agent_data.pos[1], my_agent_data.heading[0],
            #                                     my_agent_data.activation_flag,
            #                                     my_agent_data.destination[0], my_agent_data.destination[1]])
        return observation

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: IPPO for MPE.")
    parser.add_argument("--env-id", type=str, default="new_env_id")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="new_configs/my_env_own_config.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv
    set_seed(configs.seed)  # Set the random seed.
    envs = make_envs(configs)  # Make the environment.
    if configs.learner == 'IPPO_Learner':
        Agents = IPPO_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.
    elif configs.learner == 'MAPPO_Clip_Learner':
        Agents = MAPPO_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.
    else:
        Agents = None
        print('No valid learner parameters')

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():  # Print the training information.
        print(f"{k}: {v}")

    configs.benchmark = False

    if configs.benchmark:  # training while saving benchmark
        def env_fn():  # Define an environment function for test method.
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agents.test(env_fn, test_episode)
        Agents.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agents.train(eval_interval)
            test_scores = Agents.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agents.current_step}
                # save best model
                Agents.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:  # normal training
        # configs.test = 1
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)


            Agents.load_model(path=Agents.model_dir_load)
            scores = Agents.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")

            # ------------- show evaluation result gif -----------------

            print("Finish testing.")
        else:
            Agents.train(configs.running_steps // configs.parallels)
            Agents.save_model("final_train_model.pth")
            print("Finish training!")

    Agents.finish()
