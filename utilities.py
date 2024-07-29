# -*- coding: utf-8 -*-
"""
@Time    : 6/23/2024 1:09 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import random
from shapely.affinity import scale
import numpy as np
import matplotlib.image as mpimg
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from matplotlib.patches import Polygon as matPolygon
import cairosvg
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')


def load_svg_image(svg_path):
    png_image_data = cairosvg.svg2png(url=svg_path)
    image = Image.open(io.BytesIO(png_image_data))
    return image


cloud_img_path = r'F:\githubClone\HotspotResolver_24\pictures\heat_map_image.png'  # Replace with your image path
cloud_img = mpimg.imread(cloud_img_path)
aircraft_svg_path = r'F:\githubClone\HotspotResolver_24\pictures\Aircraft.svg'  # Replace with your SVG path
plane_img = load_svg_image(aircraft_svg_path)


def estimated_area_swap_by_arbitary_cloud(cloud_agent):
    initial_polygon = cloud_agent.cloud_actual_previous_shape
    final_polygon = cloud_agent.cloud_actual_cur_shape
    # Create a line from the previous position to the current position
    host_pass_line = LineString([cloud_agent.pre_pos, cloud_agent.pos])
    # Buffer the line to create the swept area
    max_dimension = max(initial_polygon.bounds[2] - initial_polygon.bounds[0],
                        initial_polygon.bounds[3] - initial_polygon.bounds[1])
    host_passed_volume = host_pass_line.buffer(max_dimension / 2, cap_style=1)

    # Calculate the union of the initial, final, and buffered areas
    union_area = initial_polygon.union(final_polygon).union(host_passed_volume)
    return union_area


def generate_elliptical_heatmap_data(size, radius, xfact, yfact):
    x = np.linspace(-radius * xfact, radius * xfact, size)
    y = np.linspace(-radius * yfact, radius * yfact, size)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((x**2 / (2.0 * (radius * xfact/3.0)**2)) + (y**2 / (2.0 * (radius * yfact/3.0)**2))))
    return x, y, z


def between_polygon_conflict(polygon1, polygon2):
    if polygon1.touches(polygon2):
        return 0  # "Touch"
    elif polygon1.intersects(polygon2) or polygon1.within(polygon2) or polygon1.overlaps(polygon2):
        return 1  # "Conflict"
    else:
        return 0  # "No Conflict"


def polygons_own_circle_conflict(circle, polygons):
    conflicts = []
    for polygon in polygons:
        if not polygon.touches(circle) and polygon.intersects(circle):
            conflicts.append(polygon)
        elif polygon.within(circle):
            conflicts.append(polygon)
    return conflicts


def polygons_single_cloud_conflict(circle, cloud_polygon):
    conflicts = []
    if not cloud_polygon.touches(circle) and cloud_polygon.intersects(circle):
        conflicts.append(cloud_polygon)
    elif cloud_polygon.within(circle):
        conflicts.append(cloud_polygon)
    return conflicts


def check_line_circle_conflict(circle, lines):
    conflicts = []
    for line in lines:
        if not line.touches(circle) and line.intersects(circle):
            conflicts.append(line)
        elif line.within(circle):
            conflicts.append(line)
    return conflicts


def generate_random_convex_polygon(num_points, x_max, y_max):
    points = [(random.randint(0, x_max), random.randint(0, y_max)) for _ in range(num_points)]
    # Sort the points
    points = sorted(points)

    # Build the convex hull using Andrew's monotone chain algorithm
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it's repeated at the beginning of the other half
    convex_hull = lower[:-1] + upper[:-1]
    # Ensure the polygon is closed
    if convex_hull[0] != convex_hull[-1]:
        convex_hull.append(convex_hull[0])
    return Polygon(convex_hull)

    # Generate random convex polygons
    # polygons = [generate_random_convex_polygon(random.randint(3, 10), 200, 200) for _ in range(5)]

def extract_agent_index(agent_string):
    match = re.search(r'agent_(\d+)', agent_string)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No agent index found in the string.")

# Function to calculate cross product of three points
def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# this is to remove the need for the package descrete
def shapelypoly_to_matpoly(ShapelyPolgon, inFill=False, Edgecolor=None, FcColor='blue'):
    xcoo, ycoo = ShapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=Edgecolor, facecolor=FcColor)
    return matPolyConverted


def animate(frame_num, ax, env, trajectory_eachPlay, cloud_movement):
    ax.clear()
    plt.axis('equal')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    # plot potential reference path for display
    for P_line_idx, P_line in enumerate(env.potential_ref_line):
        x, y = P_line.xy
        ax.plot(x, y, color="k")

    # shown boundaries
    # for bound_line in env.boundaries:
    #     x, y = bound_line.xy
    #     ax.plot(x, y, color="g")

    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    # ------------------- features that permanent exist ----------------------------------------------
    # show PRD
    for poly in env.prd_polygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, 'blue')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    # show clouds
    # for cloud_agent in env.cloud_config:
    #     cloud_actual_ini_shape = scale(cloud_agent.ini_pos.buffer(cloud_agent.radius), xfact=cloud_agent.x_fact, yfact=cloud_agent.y_fact)
    #     matp_poly = shapelypoly_to_matpoly(cloud_actual_ini_shape, False, 'red')
    #     ax.add_patch(matp_poly)

    for agent_name, agent in env.my_agent_self_data.items():
        agentIdx = extract_agent_index(agent_name)
        # plt.plot(agent.ini_pos[0], agent.ini_pos[1], 'o', color='y')
        plt.text(agent.ini_pos[0] + 3, agent.ini_pos[1] + 3, 'w_' + str(agentIdx + 1))
        # plot self_circle of the drone
        # self_circle = Point(agent.ini_pos[0],
        #                     agent.ini_pos[1]).buffer(agent.NMAC_radius, cap_style='round')
        # grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='k')
        # ax.add_patch(grid_mat_Scir)

        # plot current aircraft's final goal
        # plt.plot(agent.destination[0], agent.destination[1], marker='*', color='r', markersize=1)
        # plt.text(agent.destination[0]+2*agentIdx, agent.destination[1]+2*agentIdx, "E_"+str(agentIdx))
        dest_circle = Point(agent.destination[0], agent.destination[1]).buffer(agent.destination_radius, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(dest_circle, False, 'red')
        grid_mat_Scir.set_zorder(2)  # Set this to any desired integer
        ax.add_patch(grid_mat_Scir)
        # # link individual drone's starting position with its goal
        # ini = agent.ini_pos
        # # for wp in agent.goal:
        # for wp in agent.ref_line.coords:
        #     # plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
        #     plt.plot([wp[0], ini[0]], [wp[1], ini[1]], color='k')
        #     ini = wp
    # ------------------- end  features that permanent exist ----------------------------------------------

    # --------------------- start of plotting moving features ----------------------
    for agent_name, agent in env.my_agent_self_data.items():
        agent_idx = extract_agent_index(agent_name)
        start_frame = agent.eta
        agent_traj = trajectory_eachPlay[agent_idx]
        if frame_num >= start_frame and (frame_num - start_frame) < len(agent_traj):
            # Adjust the index by subtract the start frame
            adjusted_frame_num = frame_num - start_frame
            x, y = agent_traj[adjusted_frame_num][0][0], agent_traj[adjusted_frame_num][0][1]
            heading = agent_traj[adjusted_frame_num][1]
            img_extent = [
                x - agent.NMAC_radius,
                x + agent.NMAC_radius,
                y - agent.NMAC_radius,
                y + agent.NMAC_radius
            ]
            # Apply rotation
            transform = Affine2D().rotate_deg_around(x, y, heading-90) + ax.transData
            # Display the SVG image
            ax.imshow(plane_img, extent=img_extent, zorder=10, transform=transform)
            # plt.text(x-1, y-1, "A_"+str(a_idx))

            # plot self_circle of the drone
            self_circle = Point(x, y).buffer(agent.NMAC_radius, cap_style='round')
            grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=True, Edgecolor=None, FcColor='lightblue')  # None meaning no edge
            grid_mat_Scir.set_zorder(2)
            grid_mat_Scir.set_alpha(0.9)  # Set transparency to 0.5
            ax.add_patch(grid_mat_Scir)

    for c_idx, c_step in enumerate(cloud_movement[frame_num]):
        cloud_actual_cur_shape = scale(c_step.buffer(8), xfact=env.cloud_config[c_idx].x_fact, yfact=env.cloud_config[c_idx].y_fact)
        matp_poly = shapelypoly_to_matpoly(cloud_actual_cur_shape, False, 'red')
        ax.add_patch(matp_poly)

        # Get the cloud's centroid for positioning the image
        centroid = cloud_actual_cur_shape.centroid

        # Generate heatmap data
        heatmap_size = 100  # Resolution of the heatmap
        radius = 8
        xfact_ = env.cloud_config[c_idx].x_fact
        yfact_ = env.cloud_config[c_idx].y_fact

        # Create a custom colormap: green -> yellow -> red -> white
        colors = ["#00ff00", "#ffff00", "#ff0000"]
        n_bins = [10, 20, 30]  # Discretize color map, adjusted for smoother transition
        cmap_name = "weather_cmap"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=sum(n_bins))

        # Create a custom colormap: red -> yellow -> green
        # colors = [(1, 0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Adding a transparent start
        # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        # x, y, z = generate_elliptical_heatmap_data(heatmap_size, radius, xfact_, yfact_)

        # Generate heatmap data
        heatmap_size = 100  # Resolution of the heatmap
        x, y, z = generate_elliptical_heatmap_data(heatmap_size, radius, xfact_, yfact_)
        # Display the heatmap
        contour = ax.contourf(x + centroid.x, y + centroid.y, z, levels=50, cmap=cmap, alpha=1)
        for c in contour.collections:
            c.set_edgecolor("face")  # Remove contour lines

        # Create an elliptical patch for clipping
        ellipse_patch = Ellipse((centroid.x, centroid.y), width=2*radius*xfact_, height=2*radius*yfact_, angle=0)

        # Create a path from the patch and use it to clip the contour
        path = Path(ellipse_patch.get_verts())
        for collection in contour.collections:
            collection.set_clip_path(path, ax.transData)

        # # Set image size in data units
        # cloud_img_width = 40  # this image size value is the actual x&y value in matplotlib
        # cloud_img_height = 40
        # # Calculate the extent or the bounding box.
        # cloud_img_extent = [
        #     centroid.x - cloud_img_width/2,
        #     centroid.x + cloud_img_width/2,
        #     centroid.y - cloud_img_height/2,
        #     centroid.y + cloud_img_height/2
        # ]
        # # ax.imshow(cloud_img, extent=cloud_img_extent, zorder=2, alpha=0.5)  # Adjust alpha for transparency
        # ax.imshow(cloud_img, extent=cloud_img_extent, zorder=2)  # Adjust alpha for transparency

        # ---------------------end of plotting moving features ----------------------

    return ax.patches + [ax.texts]


def save_gif(env, trajectory_eachPlay, cloud_movement, eps_step):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(1, 1)

    plt.axis('equal')
    plt.xlim(0, 200)
    plt.ylim(0, 200)

    # plot potential reference path for display
    for P_line in env.potential_ref_line:
        x, y = P_line.xy
        ax.plot(x, y, color="k")

    # # shown boundaries
    # for bound_line in env.boundaries:
    #     x, y = bound_line.xy
    #     ax.plot(x, y, color="g")

    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    # show PRD
    for poly in env.prd_polygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, 'blue')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    # show clouds
    for cloud_agent in env.cloud_config:
        matp_poly = shapelypoly_to_matpoly(cloud_agent.cloud_actual_cur_shape, False, 'red')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    for agent_name, agent in env.my_agent_self_data.items():
        # Calculate the extent for the SVG image
        img_extent = [
            agent.ini_pos[0] - agent.NMAC_radius,
            agent.ini_pos[0] + agent.NMAC_radius,
            agent.ini_pos[1] - agent.NMAC_radius,
            agent.ini_pos[1] + agent.NMAC_radius
        ]
        # Apply rotation
        transform = Affine2D().rotate_deg_around(agent.ini_pos[0], agent.ini_pos[1], agent.ini_heading-90) + ax.transData
        # Display the SVG image
        ax.imshow(plane_img, extent=img_extent, zorder=2, transform=transform)

        plt.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
        # plot self_circle of the drone
        self_circle = Point(agent.ini_pos[0],
                            agent.ini_pos[1]).buffer(agent.NMAC_radius, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=True, Edgecolor=None, FcColor='lightblue')  # None meaning no edge
        grid_mat_Scir.set_zorder(2)
        grid_mat_Scir.set_alpha(0.5)  # Set transparency to 0.5
        ax.add_patch(grid_mat_Scir)

        # plot drone's detection range
        detec_circle = Point(agent.ini_pos[0],
                             agent.ini_pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
        detec_circle_mat = shapelypoly_to_matpoly(detec_circle, inFill=False, Edgecolor='g')
        # ax.add_patch(detec_circle_mat)

        # plot current agent's destination
        plt.plot(agent.destination[0], agent.destination[1], marker='*', color='red', markersize=15)
        plt.text(agent.destination[0], agent.destination[1], agent.agent_name)
        dest_circle = Point(agent.destination[0], agent.destination[1]).buffer(agent.destination_radius, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(dest_circle, False, 'red')
        grid_mat_Scir.set_zorder(2)
        ax.add_patch(grid_mat_Scir)

    # Create animation
    ani = animation.FuncAnimation(fig, animate, fargs=(ax, env, trajectory_eachPlay, cloud_movement), frames=eps_step,
                                  interval=300, blit=False)
    # Save as GIF
    gif_path = r'F:\githubClone\xuance\my_env' + str(1) + '.gif'
    ani.save(gif_path, writer='pillow')

    # Close figure
    plt.close(fig)


def display_cur_state(agents, prds, clouds, bounds, potential_ref_line):
    fig, ax = plt.subplots()
    cloud_img_path = r'F:\githubClone\HotspotResolver_24\pictures\heat_map_image.png'  # Replace with your image path
    cloud_img = mpimg.imread(cloud_img_path)
    svg_path = r'F:\githubClone\HotspotResolver_24\pictures\Aircraft.svg'  # Replace with your SVG path
    plane_img = load_svg_image(svg_path)
    for agent_idx, agent_obj in enumerate(agents):
        # draw reference line and aircraft's own marker and circle
        ref_x, ref_y = agent_obj.ref_line.xy
        ax.plot(ref_x, ref_y, color='cyan', label='Reference Line')

        # ax.plot(agent_obj.pos[0], agent_obj.pos[1], marker=MarkerStyle(">", fillstyle="right",
        #                                                                transform=Affine2D().rotate_deg(
        #                                                                    agent_obj.heading)), color='y')

        # Calculate the extent for the SVG image
        img_extent = [
            agent_obj.pos[0] - agent_obj.NMAC_radius,
            agent_obj.pos[0] + agent_obj.NMAC_radius,
            agent_obj.pos[1] - agent_obj.NMAC_radius,
            agent_obj.pos[1] + agent_obj.NMAC_radius
        ]
        # Apply rotation
        transform = Affine2D().rotate_deg_around(agent_obj.pos[0], agent_obj.pos[1], 0-90) + ax.transData
        # Display the SVG image
        ax.imshow(plane_img, extent=img_extent, zorder=2, transform=transform)

        plt.text(agent_obj.pos[0]+2, agent_obj.pos[1]+2, "A_"+str(agent_idx))
        # plot current host drone circle
        self_circle = Point(agent_obj.pos[0], agent_obj.pos[1]).buffer(agent_obj.NMAC_radius, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
        ax.add_patch(grid_mat_Scir)

        # plot current host drone's destination
        plt.plot(agent_obj.destination[0], agent_obj.destination[1], marker='*', color='y', markersize=15)

        plt.text(agent_obj.destination[0] + agent_idx, agent_obj.destination[1] + agent_idx, "E_"+str(agent_idx))

        dest_circle = Point(agent_obj.destination[0], agent_obj.destination[1]).buffer(agent_obj.destination_radius, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(dest_circle, False, 'red')
        ax.add_patch(grid_mat_Scir)

    # plot potential reference path for display
    for P_line in potential_ref_line:
        x, y = P_line.xy
        ax.plot(x, y, color="cyan")

    for i, polygon in enumerate(prds):  # show PRDs
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="orange", label=f'PRD {i + 1}')

    # show clouds
    for cloud in clouds:
        scaled_x, scaled_y = cloud.cloud_actual_cur_shape.exterior.xy
        ax.plot(scaled_x, scaled_y, label='Cloud', color='red')

        # Get the cloud's centroid for positioning the image
        centroid = cloud.cloud_actual_cur_shape.centroid
        # Set image size in data units
        cloud_img_width = 40  # this image size value is the actual x&y value in matplotlib
        cloud_img_height = 40
        # Calculate the extent or the bounding box.
        cloud_img_extent = [
            centroid.x - cloud_img_width/2,
            centroid.x + cloud_img_width/2,
            centroid.y - cloud_img_height/2,
            centroid.y + cloud_img_height/2
        ]

        # ax.imshow(cloud_img, extent=cloud_img_extent, zorder=2, alpha=0.5)  # Adjust alpha for transparency
        ax.imshow(cloud_img, extent=cloud_img_extent, zorder=2)  # Adjust alpha for transparency

        # Draw the bounding box
        rect = patches.Rectangle(
            (cloud_img_extent[0], cloud_img_extent[2]),  # (left, bottom)
            cloud_img_width,  # width
            cloud_img_width,  # height
            linewidth=1,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)


    # shown boundaries
    for line in bounds:
        x, y = line.xy
        ax.plot(x, y, color="g")

    ax.set_title('Random Convex Polygons within 200x200 Space')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.axis('equal')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.grid(True)

    plt.show()


def calculate_next_position(start_pos, target_pos, speed, time_step):
    # Calculate the direction vector from start to target
    direction_vector = target_pos - start_pos

    # Normalize the direction vector to get the unit direction vector
    distance_to_target = np.linalg.norm(direction_vector)
    unit_direction_vector = direction_vector / distance_to_target

    # Calculate the distance the agent will travel in one time step
    distance_travelled = speed * time_step

    # Calculate the new position
    new_position = start_pos + unit_direction_vector * distance_travelled

    return new_position


class NormalizeData:
    def __init__(self, x_min_max, y_min_max):
        self.normalize_max = 1
        self.normalize_min = -1
        self.dis_min_x = x_min_max[0]
        self.dis_max_x = x_min_max[1]
        self.dis_min_y = y_min_max[0]
        self.dis_max_y = y_min_max[1]

    def nmlz_pos(self, pos_c):
        x, y = pos_c[0], pos_c[1]
        x_normalized = 2 * ((x - self.dis_min_x) / (self.dis_max_x - self.dis_min_x)) - 1
        y_normalized = 2 * ((y - self.dis_min_y) / (self.dis_max_y - self.dis_min_y)) - 1
        return np.array([x_normalized, y_normalized])