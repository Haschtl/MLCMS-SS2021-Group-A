#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from collections import deque, Counter
from tkinter.filedialog import askopenfilename

# np.set_printoptions(threshold=sys.maxsize)

# Matplotlib configuration for pColormesh
plot_config = {"edgecolors": '#CCCCCC', "linewidths": 0.2}
cmap = colors.ListedColormap(
    ['purple', 'white', 'red', '#E0E000', "#E0E000"])
boundaries = [x-0 for x in [-1, 0, 1, 2, 3]]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)


def euclidean_norm(pos1: tuple[int, int], pos2: tuple[int, int]):
    return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)


def cost(distance: float, max_distance: float):
    if distance >= max_distance:
        return 0
    else:
        return np.exp(1/(distance**2-max_distance**2))


def nearest(current_position: tuple[int, int], positions: list[tuple[int, int]]):
    """
    Returns the position from "positions", which is nearest to "current_position"
    """
    distances = [euclidean_norm(current_position, pos)
                 for pos in positions]
    return positions[np.argmin(distances)]


def nearest_dist(current_position: tuple[int, int], positions: list[tuple[int, int]]):
    """

    """
    distances = [euclidean_norm(current_position, pos)
                 for pos in positions]
    return np.min(distances)

def minimum_value_index_not_in_list(costmap_dict, traversed):
    """
    Returns minimum valued index in the costmap that is not in the traversed list
    """
    sorted_costmap_indices = sorted(costmap_dict, key=costmap_dict.get)
    for i in sorted_costmap_indices:
        if i not in traversed:
            return i

def minimum_value_index_in_list(costmap, non_traversed):
    """
    Returns minimum valued index in the costmap that is in the non_traversed list
    """
    indexlisttranspose = np.array(non_traversed).T.tolist()
    filtered_costmap = costmap[tuple(indexlisttranspose)]
    # sorted_costmap_indices = np.argsort(filtered_costmap.ravel(), axis=None)
    # return non_traversed[sorted_costmap_indices[0]]
    min_indice = np.argmin(filtered_costmap.ravel(), axis=None)
    return non_traversed[min_indice]

def lowest_cost(costmap: np.array, cells: list[tuple[int, int]]):
    """
    Returns cell index with lowest cost in costmap from given list of cells
    """
    cell_costs = {cell : costmap[cell[0]][cell[1]] for cell in cells}
    lowest_cost_cell = min(cell_costs, key=cell_costs.get)
    return lowest_cost_cell


class Simulator:
    '''
    Simulator class containing all functions to perform simple crowd simulations
    Scenarios are saved in .npy-files (created with scenario_editor.py)
    Value-Mapping:
    - Obstacle: -1
    - Empty: 0
    - Pedestrian: 1
    - Target: 2
    '''

    def __init__(self, grid_scaling: float = 1, time_step: float = 1, duration: float = 1000, history_length:int = 5) -> None:
        '''
        Initialize empty scenario and plot configuration
        '''
        self.grid_scaling = grid_scaling  # Value represents scaling in meters: grid_scaling*index [meter]
        self.scenario: np.array = np.array([])
        self.all_targets = self._all_targets
        self.all_obstacles = self._all_obstacles
        self.costmap: np.array = np.array([])
        self.time_step: float = time_step
        self.duration: float = duration
        self.statistics = {
            "time": [],
            "pedestrians": [],
            "avg_distance": [],
            "avg_velocity": [],
        }
        # History has max-length of 10
        self.history: deque = deque([], history_length)
        self.last_history: list = []
        self.init_plot()

    def init_plot(self) -> None:
        '''
        Initialize plot
        '''
        plt.ion()  # Plots are shown non-blocking
        # plt.axis('square')

    def stay_open(self):
        '''
        Tell the simulator to stay open (sets plots to blocking-mode)
        '''
        plt.ioff()  # Plots are shown blocking
        plt.show()

    def draw(self, pause: float = 0.1) -> None:
        '''
        Draws the scenario in the matplotlib-figure
        '''
        if pause == 0:
            # Figure will freeze during simulation, if pause is zero
            pause = 0.05
        plt.cla()
        plt.pcolormesh(self.scenario, **plot_config, cmap=cmap, norm=norm)
        # plt.show()
        plt.draw()
        plt.pause(pause)

    def load(self, filepath: str) -> None:
        '''
        Load a .npy array file (shape: NxN)
        Value-Mapping:
        - Obstacle: -1
        - Empty: 0
        - Pedestrian: 1
        - Target: 2
        '''
        self.scenario = np.load(filepath)
        self.all_targets = self._all_targets
        self.all_obstacles = self._all_obstacles
        self.calculate_costs()
        # self.draw()

    @property
    def all_pedestrians(self):
        """
        Returns a list of tuples with indices of pedestrians
        """
        return np.argwhere(self.scenario == 1)

    @property
    def _all_targets(self):
        """
        Returns a list of tuples with indices of targets
        """
        return np.argwhere(self.scenario == 2)

    @property
    def _all_obstacles(self):
        """
        Returns a list of tuples with indices of obstacles
        """
        return np.argwhere(self.scenario == -1)

    def neighbours(self, position: tuple[int, int], diagonal: bool = True, pedestrians: bool = False, obstacles: bool = False, empty: bool = True, target: bool = True, nondiagonal: bool = True):
        """
        Returns the indices of empty (or target) neighbours of the position
        Args:
            position: The position to check
            diagonal: Set True to include the diagonal boxes
        """
        if nondiagonal:
            neighbours = [
                (position[0]+1, position[1]+0),
                (position[0]-1, position[1]+0),
                (position[0]+0, position[1]+1),
                (position[0]+0, position[1]-1)
            ]
        else:
            neighbours = []
        if diagonal:
            neighbours += [
                (position[0]+1, position[1]+1),
                (position[0]+1, position[1]-1),
                (position[0]-1, position[1]+1),
                (position[0]-1, position[1]-1)

            ]
        empty_neighbours = []
        for neighbour in neighbours:
            if self.is_position(neighbour, pedestrians, obstacles, empty, target):
                empty_neighbours.append(neighbour)
        return empty_neighbours

    def move2(self, position: tuple[int, int], new_position: tuple[int, int], velocity: float = 1.33, verbose: bool = False):
        """
        Move the element at position (a,b) in x and y direction
        New position: (position[0]+x,position[1]+y)
        If new position is an obstacle, moving is aborted
        If new position is another pedestrian, moving is aborted
        If new position is a target, the new position stays a target
        """
        if position[0] == new_position[0] and position[1] == new_position[1]:
            return
        value = self.scenario[position[0]][position[1]]
        if value != 1:
            if verbose:
                print("Can't move, this position is not a pedestrian")
            return
        if not self.is_position(new_position, verbose, empty=True, target=True):
            return
        if self.scenario[new_position[0]][new_position[1]] == 2:
            if verbose:
                print("Target reached")
            self.scenario[position[0]][position[1]] = 0
            return

        average_velocity = self.average_velocity(position)
        if average_velocity <= velocity:
            self.scenario[position[0]][position[1]] = 0
            self.scenario[new_position[0]][new_position[1]] = value
            self.last_history.append([np.copy(position), np.copy(new_position)])
        else:
            self.last_history.append([np.copy(position), np.copy(position)])


    def move(self, position: tuple[int, int], x: int, y: int, velocity:float = 1.33, verbose: bool = False):
        """
        Move the element at position (a,b) in x and y direction
        New position: (position[0]+x,position[1]+y)
        If new position is an obstacle, moving is aborted
        If new position is another pedestrian, moving is aborted
        If new position is a target, the new position stays a target
        """
        new_position = (position[0]+x, position[1]+y)
        self.move2(position, new_position, velocity, verbose)

    def is_valid_position(self, position: tuple[int, int], verbose: bool = False):
        """
        Returns true if position is inside scenario boarders
        """
        if position[0] >= self.scenario.shape[0] or position[1] >= self.scenario.shape[1] or position[0] < 0 or position[1] < 0:
            if verbose:
                print("Can't move, new position out of boundaries {}".format(
                    self.scenario.shape))
            return False
        return True

    def is_position(self, position: tuple[int, int], pedestrians: bool = False, obstacles: bool = False, empty: bool = True, target: bool = True, verbose: bool = False):
        """
        Returns true if position is (pedestrian, obstacle), empty or target
        """
        if not self.is_valid_position(position):
            return False
        if self.scenario[position[0]][position[1]] == -1 and not obstacles:
            if verbose:
                print("Can't move, new position is an obstacle")
            return False
        if self.scenario[position[0]][position[1]] == 1 and not pedestrians:
            if verbose:
                print("Can't move, new position is another pedestrian")
            return False
        if self.scenario[position[0]][position[1]] == 1 and not empty:
            if verbose:
                print("Position is empty")
            return False
        if self.scenario[position[0]][position[1]] == 2 and not target:
            if verbose:
                print("Position is target")
            return False
        return True

    def last_position(self, position: tuple[int, int]):
        """
        Returns the last position of a pedestrian, None if not found
        """
        for move in self.history[-1]:
            if position[0] == move[1][0] and position[1] == move[1][1]:
                return move[0]
        return position

    def last_positions(self, position: tuple[int, int], history_idx: int = -1):
        """
        Returns the last position of a pedestrian, None if not found
        """
        if len(self.history) == 0:
            return []
        # check if there was a move from this position
        for move in self.history[history_idx]:
            if position[0] == move[1][0] and position[1] == move[1][1]:
                pos = move[0]
                if -(history_idx-1) <= len(self.history):
                    last = self.last_positions(pos, history_idx-1)
                    last.append(pos)
                    return last
                else:
                    return [pos]
        # if no move was found, search back in history
        if -(history_idx-1) <= len(self.history):
            last = self.last_positions(position, history_idx-1)
            last.append(position)
            return last
        else:
            return [position]

    def last_move(self, position: tuple[int, int], history_idx: int = -1):
        """
        Returns the number of timesteps since last move, None if not found
        """
        if len(self.history) == 0:
            return None
        # check if there was a move from this position
        for move in self.history[history_idx]:
            if position[0] == move[1][0] and position[1] == move[1][1]:
                return -history_idx-1
        # if no move was found, search back in history
        if -(history_idx-1) <= len(self.history):
            return self.last_move(position, history_idx-1)
        else:
            return None #len(self.history)


    def velocity(self, position: tuple[int, int]):
        """
        Returns the last velocity of a pedestrian in m/s
        """
        last_position = self.last_position(position)
        return euclidean_norm(last_position, position)*self.grid_scaling/self.time_step



    def average_velocity(self, position: tuple[int, int]):
        """
        Returns the average velocity of a pedestrian in m/s
        """
        last_positions = self.last_positions(position, -1)

        distances = []
        for idx, pos in enumerate(last_positions):
            if idx >= len(last_positions)-1:
                distances.append(euclidean_norm(pos, position))
            else:
                distances.append(euclidean_norm(pos, last_positions[idx+1]))
        # print(distances)
        if len(distances) == 0:
            return 0
        return np.average([d*self.grid_scaling/self.time_step for d in distances])

    def last_distance(self, position: tuple[int, int]):
        last_position = self.last_position(position)
        if last_position is None:
            return None
        return euclidean_norm(position, last_position)

    def calculate_costs(self, diagonal: bool = True):
        # create costmap filled with inf on all cells
        print("Calculating costmap...")
        costmap = np.full(self.scenario.shape, np.inf)
        # set target costs to 0
        for t in self.all_targets:
            costmap[t[0]][t[1]] = 0
        # get indices of costmap in a costmap_indices list
        indices = np.indices(costmap.shape)
        costmap_indices = []
        for e in indices.transpose(1,2,0).tolist():
            costmap_indices += e
        # create a dict mapping indices to costs
        costmap_indices = [tuple(e) for e in costmap_indices if e not in self.all_obstacles.tolist()]

        # Dijkstra's algorithm to fill costmap with cost values
        non_traversed = [tuple(e) for e in costmap_indices]
        traversed = []
        
        last_progress = 0
        while Counter(traversed) != Counter(costmap_indices):
            progress = round(len(traversed)/len(costmap_indices)*100)
            if progress-last_progress>=5:
                print("{}%".format(progress))
                # plt.cla()
                # plt.pcolormesh(costmap)
                # plt.pause(0.1)
                last_progress = progress
            minimum_cost_index = minimum_value_index_in_list(costmap, non_traversed)
            # minimum_cost_index = minimum_value_index_not_in_list(costmap, traversed)
            traversed.append(minimum_cost_index)
            non_traversed.remove(minimum_cost_index)

            # create 2 lists for diagonal and non-diagonal neighbours so we can set higher cost for diagonal movement
            non_diag_neighbours = self.neighbours(minimum_cost_index, diagonal=False, pedestrians=True, obstacles=False)
            diag_neighbours = self.neighbours(minimum_cost_index, diagonal=True, pedestrians=True, obstacles=False, nondiagonal=False)
            for n in non_diag_neighbours:
                if costmap[minimum_cost_index]+1 < costmap[n]:
                    costmap[n] = costmap[minimum_cost_index]+1
            if diagonal:
                for n in diag_neighbours:
                    if costmap[minimum_cost_index]+1.414213 < costmap[n]:
                        costmap[n] = costmap[minimum_cost_index]+1.414213
        self.costmap = costmap
        print("done")
        plt.figure()
        plt.pcolormesh(self.costmap)
        input("Press ENTER to start the simulation")


###############################################################
#             Discrete time Simulation loop                   #
###############################################################

    def start(self, duration: float = 1000, timeout: float = 10, history_length: int = 5, grid_scaling: float = 0.4, visualize: bool = False, monitoring: bool = False, pause: float = 0.1, velocity:float = 1.33, velocity_control_cells=None, **kwargs):
        """
        Starts the simulation with a simple, discrete-time update scheme
        Args:
            time_step: constant time shift [s]
            duration: Maximum simulation time (-1 for infinite) [s]
            timeout: Maximum seconds of no moving pedestrian (-1 to disable) [s]
            history_len: Length of history (higher value: better average speed calculation, less performance)
            grid_scaling: Value represents scaling in meters: grid_scaling*index [meter]
            visualize: Visualize each time-step
            monitoring: Adds statistics on the visualization
            pause: Pause after each simulation step
        """
        time_step = grid_scaling/velocity
        self.grid_scaling = grid_scaling  
        simulation_time = 0
        self.time_step = time_step
        self.duration = duration
        self.history.clear()
        self.history = deque([],history_length)
        stuck_counter=0
        self.statistics = {
            "time":[],
            "pedestrians":[],
            "avg_distance":[],
            "avg_velocity":[],
        }
        self.velocity_control_cells=velocity_control_cells
        self.velocity_trackers = None
        if velocity_control_cells is not None:
            self.velocity_trackers = {k : [] for k, v in velocity_control_cells.items()}
        self.initial_pedestrian_count = self.all_pedestrians
        if monitoring:
            # fig, axs = plt.subplots(4)
            fig = plt.figure(constrained_layout=True)
            axs = []
            gs = fig.add_gridspec(3, 3)
            ax1 = fig.add_subplot(gs[0, 2])
            ax1.set_title("Pedestrians")
            axs.append(ax1)
            ax2 = fig.add_subplot(gs[1, 2])
            ax2.set_title("Average distance to target")
            axs.append(ax2)
            ax3 = fig.add_subplot(gs[2, 2])
            ax3.set_title("Average velocity")
            axs.append(ax3)
            axs.append(fig.add_subplot(gs[:, :-1]))
        '''
        Start the simulation
        '''
        while simulation_time <= duration or duration == -1:
            self.last_history = []
            self.simulate(simulation_time=simulation_time, velocity=velocity, **kwargs)
            if len(self.last_history) == 0:
                stuck_counter += 1
                if stuck_counter*time_step>=timeout and timeout!=-1:
                    print("No pedestrian moved. Stuck. Failed")
                    break
            else:
                stuck_counter = 0
            self.history.append(self.last_history)
            remaining_pedestrians = self.all_pedestrians
            if monitoring:
                self.monitor(remaining_pedestrians, simulation_time)
                self.draw_monitoring(axs)
            elif visualize:
                self.draw(pause)
            else:
                time.sleep(pause)
            print("Time: {}s / {}s\tPeds: {}".format(simulation_time,
                  duration, len(remaining_pedestrians)))


            if len(remaining_pedestrians) == 0:
                print("All pedestrians reached the target position")
                break
            simulation_time += time_step
        # calculate avg velocity and flow in measuring cells
        if self.velocity_control_cells is not None:
            ped_density = len(self.initial_pedestrian_count) / (self.scenario.shape[0] * self.scenario.shape[1])
            for k, vt in self.velocity_trackers.items():
                if len(vt) > 0:
                    vel = sum(vt) / len(vt)
                    flow = vel * ped_density
                    print("average velocity measured in {} : {}".format(k, vel))
                    print("flow measured in {} : {}".format(k, flow))
        if monitoring:
            self.draw_monitoring(axs)

    def simulate(self, simulation_time: float, velocity: float, **kwargs):
        """
        This is the main function executed for each time-step
        """
        # no need to use calculate_costs() if using anything other than avoid_obstacles()
        # self.random_walk()
        # self.random_walk2(diagonal=False)
        # self.direct_way(simulation_time=simulation_time, velocity=velocity, diagonal=True) # default diagonal=True
        # self.avoid_obstacles(diagonal=True, pedestrians_must_move=False) # default diagonal=True
        self.avoid_obstacles(simulation_time=simulation_time, velocity=velocity, **kwargs) # default diagonal=True

    def monitor(self, remaining_pedestrians, simulation_time: float):
        self.statistics["time"].append(simulation_time)
        self.statistics["pedestrians"].append(len(remaining_pedestrians))
        self.statistics["avg_distance"].append(np.average(
            [nearest_dist(ped, self.all_targets)*self.grid_scaling for ped in remaining_pedestrians]))
        self.statistics["avg_velocity"].append(np.average(
            [self.velocity(ped) for ped in remaining_pedestrians]))
    
    def draw_monitoring(self, axs):
        # plt.figure()
        # axs[0].clear()
        # axs[0].xlabel("Time [s]")
        # axs[0].ylabel("Pedestrians")
        
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()
        # plt.cla()
        plt.pcolormesh(self.scenario, **plot_config, cmap=cmap, norm=norm)
        # # plt.show()
        plt.draw()
        try:
            axs[0].plot(self.statistics["time"], self.statistics["pedestrians"])
        except Exception:
            pass
        # axs[0].show()
        # plt.xlabel("Time [s]")
        # plt.ylabel("Average distance [m]")
        try:
            axs[1].plot(self.statistics["time"], self.statistics["avg_distance"])
        except Exception:
            pass
        try:
            axs[2].plot(self.statistics["time"], self.statistics["avg_velocity"])
        except Exception as e:
            print(e)
        plt.pause(0.1)


###############################################################
#                 Timestep implementations                    #
###############################################################

    def random_walk(self, velocity:float=1):
        """
        performs a random step for all pedestrians
        If the position after the random step is not free, the step is ignored
        """
        for ped in self.all_pedestrians:
            dirs = np.random.randint(3, size=2)-1
            self.move(ped, *dirs, velocity)

    def random_walk2(self, velocity:float=1, diagonal: bool = True):
        """
        performs a random step for all pedestrians
        Each selects a random free neighbour to go to
        """
        for ped in self.all_pedestrians:
            neighbours = self.neighbours(ped, diagonal)
            if len(neighbours) != 0:
                idx = np.random.randint(len(neighbours))
                self.move2(ped, neighbours[idx], velocity)

    def direct_way(self, simulation_time: float, velocity: float = 1, pedestrians_must_move: bool =True, diagonal: bool = True):
        """
        Each pedestrian moves to the neighbour, which is nearest to a target
        Args:
            velocity: Target-velocity (m/s), should be 1, must not be >1
            pedestrians_must_move: If true, pedestrians can't stay on their position if they could move
        """
        all_peds = self.all_pedestrians
        # randomized order to prevent line-reading effects
        np.random.shuffle(all_peds)
        for ped in all_peds:  
            # get the empty (or target) neighbours of the pedestrian
            neighbours = self.neighbours(ped, diagonal)
            # get the average-velocity of the pedestrian
            average_velocity = self.average_velocity(ped)

            # record velocities of peds in measuring cells
            if self.velocity_control_cells is not None and 10 < simulation_time < 70:
                for k, vcc in self.velocity_control_cells.items():
                    if tuple(ped) in vcc:
                    #    print("{} in {}", tuple(ped), vcc)
                        self.velocity_trackers[k].append(average_velocity)
            if self.velocity_control_cells is not None and 70 < simulation_time < 75:
                ped_density = len(self.initial_pedestrian_count) / (self.scenario.shape[0] * self.scenario.shape[1])
                for k, vt in self.velocity_trackers.items():
                    if len(vt) > 0:
                        vel = sum(vt) / len(vt)
                        flow = vel * ped_density
                        print("average velocity measured in {} : {}".format(k, vel))
                        print("flow measured in {} : {}".format(k, flow))


            if len(neighbours) != 0 and average_velocity <= velocity:
                # only move the pedestrian, if his average speed is lower than the maximum velocity (default 1m/s ~ 1 idx/iteration)
                # this makes pedestrians have a equal velocity (bug: if the pedestrian stands still for some time, he will move faster than the max-speed)
                # get the target which is nearest to the pedestrian
                nearest_target = nearest(ped, self.all_targets)
                if pedestrians_must_move:
                    neighbour_nearest_to_target = nearest(
                        nearest_target, neighbours)  # get the neighbour which is nearest to that target
                else:
                    neighbour_nearest_to_target = nearest(
                        nearest_target, [ped, *neighbours])  # get the neighbour which is nearest to that target

                # move the pedestrian to the target
                self.move2(ped, neighbour_nearest_to_target, velocity)

    def avoid_obstacles(self, simulation_time: float, velocity: float = 1, pedestrians_must_move: bool =True, diagonal: bool = True, premovement_time: float = 1):
        """
        Each pedestrian moves to the neighbour, which is nearest to a target
        Args:
            velocity: Target-velocity (m/s), should be 1, must not be >1
            pedestrians_must_move: If true, pedestrians can't stay on their position if they could move
        """
        all_peds = self.all_pedestrians
        # randomized order to prevent line-reading effects
        np.random.shuffle(all_peds)
        for ped in all_peds:
            # get the empty (or target) neighbours of the pedestrian
            neighbours = self.neighbours(ped, diagonal)
            if not pedestrians_must_move:
                neighbours=[*neighbours,tuple(ped)]
            # get the average-velocity of the pedestrian
            average_velocity = self.average_velocity(ped)
            time_since_last_move=self.last_move(ped,-1)
            # record velocities of peds in measuring cells
            if self.velocity_control_cells is not None and 10 < simulation_time < 70:
                for k, vcc in self.velocity_control_cells.items():
                    if tuple(ped) in vcc:
                        self.velocity_trackers[k].append(average_velocity)
            if self.velocity_control_cells is not None and 70 < simulation_time < 71:
                ped_density = len(self.initial_pedestrian_count) / (self.scenario.shape[0] * self.scenario.shape[1])
                for k, vt in self.velocity_trackers.items():
                    if len(vt) > 0:
                        vel = sum(vt) / len(vt)
                        flow = vel * ped_density
                        print("average velocity measured in {} : {}".format(k, vel))
                        print("flow measured in {} : {}".format(k, flow))

            can_move = False
            if time_since_last_move is not None:
                time_since_last_move = time_since_last_move*self.time_step
                if time_since_last_move == 0 or time_since_last_move >= premovement_time:
                    # the pedestrian is already moving or
                    # the pedestrian stand still for the premovement time
                    can_move = True
            else:
                # the pedestrian did not make any move
                simulation_time = len(self.history)*self.time_step
                if simulation_time>=premovement_time:
                    can_move = True
                
            # if len(neighbours) != 0 and average_velocity <= velocity and can_move:
            if len(neighbours) != 0 and can_move:
                neighbour_with_lowest_cost = lowest_cost(self.costmap, neighbours)
                self.move2(ped, neighbour_with_lowest_cost, velocity)

if __name__ == "__main__":
    filename = askopenfilename()
    if filename == "":
        sys.exit(-1)
    sim = Simulator()
    sim.load(filename)
    # sim.calculate_costs(diagonal=True)

    try:
        sim.start(visualize=True, monitoring=True, grid_scaling=0.4,
                  velocity=1.33, diagonal=True, pedestrians_must_move=False, premovement_time=1,
                  velocity_control_cells=None)
                  #velocity_control_cells={"main":[(y,x) for x in range(248,252) for y in range(11,16)],
                  #                        "control1":[(y,x) for x in range(248,252) for y in range(16,21)],
                  #                        "control2":[(y,x) for x in range(198,202) for y in range(11,16)]})
                  # commented out velocity_control_cells are for the rimea4downscaled05.npy scenario
    except KeyboardInterrupt:
        print("Simulation aborted")

    sim.stay_open()
