# How to create a scenario
1. Run the editor `python scenario_editor.py`
2. Create a new file/edit file
3. Save .npy file to `/scenarios`

# How to run the simulation
1. Run simulator `python simulator.py`

# How to edit the simulation
The Simulator-class automatically loads a scenario and runs the simulation.

The function `simulate()` is executed in each time-step of the simulation. The Simulator.scenario is at time-step n always.


The simulator-class provides some functions and parameters, which are useful for the simulation:
```python

Simulator.scenario # The scenario (2D-Array, -1:Obstacle, 0:Empty, 1:Pedestrian, 2:Target) (n)
Simulator.all_pedestrians # Returns a list of positions (tuples) of all pedestrians
Simulator.all_targets # Returns a list of positions (tuples) of all targets
Simulator.all_obstacles # Returns a list of positions (tuples) of all obstacles
Simulator.history # A list of the moves of previous time-steps (n-1, n-2, ...) maximum: 10: [ [[[1,1],[1,2]], [[2,1],[2,2]], ...], [[[1,0],[1,1]], [[2,0],[2,1]], ...], ...] 

Simulator.neighbours(position: tuple[int, int], diagonal: bool = False) # returns a list of the positions (tuples) of empty (or target) neighbouring fields to "position"
Simulator.move2(position, new_position) # Move a üedestroam at "position" to "new_position" (new position must be empty or target)
Simulator.move(position, x, y) # Move a pedestrian at "position" with x and y offsets (new position must be empty or target)
Simulator.is_empty_position(position: tuple[int, int], verbose: bool = False) # Returns true if position is empty or a target
```


# Known bugs

Pedestrians can walk through thin diagonal walls:
```
     __              __         
   _|_|            _|_|         
 _|_|       ->   x|_|        
|_|x            |_|             


```