import csv
import json
import numpy as np


# An empty pedestrian dictionary
PEDESTRIAN = {
    "attributes": {
        "id": 1,
        "radius": 0.2,
        "densityDependentSpeed": False,
        "speedDistributionMean": 1.34,
        "speedDistributionStandardDeviation": 0.26,
        "minimumSpeed": 0.5,
        "maximumSpeed": 2.2,
        "acceleration": 2.0,
        "footstepHistorySize": 4,
        "searchRadius": 1.0,
        "walkingDirectionCalculation": "BY_TARGET_CENTER",
        "walkingDirectionSameIfAngleLessOrEqual": 45.0
    },
    "source": None,
    "targetIds": [],
    "nextTargetListIndex": 0,
    "isCurrentTargetAnAgent": False,
    "position": {
        "x": 3.298484123799754,
        "y": 0.627130303937212
    },
    "velocity": {
        "x": 0.0,
        "y": 0.0
    },
    "freeFlowSpeed": 0.8205853883147124,
    "followers": [],
    "idAsTarget": -1,
    "isChild": False,
    "isLikelyInjured": False,
    "psychologyStatus": {
        "mostImportantStimulus": None,
        "threatMemory": {
            "allThreats": [],
            "latestThreatUnhandled": False
        },
        "selfCategory": "TARGET_ORIENTED",
        "groupMembership": "IN_GROUP",
        "knowledgeBase": {
            "knowledge": []
        }
    },
    "groupIds": [],
    "groupSizes": [],
    "trajectory": {
        "footSteps": []
    },
    "modelPedestrianMap": None,
    "type": "PEDESTRIAN"
}


def load_postvis_file(filename):
    """
    Load a postvis file into a list of dicts (unused)
    """
    output = []
    with open( filename , 'r' ) as f:
        reader = csv.DictReader(f, delimiter=' ', quotechar='|')
        for line in reader:
            output.append(line)
    return output


class Scenario(dict):
    """
    A scenario class inheriting the dict-type with added functionality specific for scenario:
    - load and save the scenario
    - add pedestrians to the scenario
    """
    def __init__(self, filename:str=None):
        # super(dict).__init__()
        if filename is not None:
            self.load(filename)

    def load(self, filename:str):
        with open( filename, "r") as f:
            self.update(json.load(f))
        return self

    def save(self, filename, scenario_name:str=None):
        if scenario_name is not None:
            self["name"] = scenario_name
        with open( filename, "w") as f:
            json.dump(self, f, indent=2)

    def add_pedestrian(self, position:tuple[float,float], target:int=None):
        new_ped = PEDESTRIAN
        # add a target id
        _target = None
        for t in self.targets:
            if target is None or t["id"] == target:
                _target = t["id"]
                break
        if _target is None:
            if target is None:
                print("Warning: No targets in scenario! Pedestrian will not move!")
            else:
                print("Warning: Invalid target-Id! Pedestrian will not move!")
        new_ped["targetIds"].append(int(_target))
        # modify position
        new_ped["position"]["x"] = position[0]
        new_ped["position"]["y"] = position[1]
        # modify pedestrian id
        ped_id = 1
        while ped_id in self.pedetrian_ids:
            ped_id += 1
        new_ped["attributes"]["id"] = int(ped_id)

        self["scenario"]["topography"]["dynamicElements"].append(new_ped)
        print("Pedestrian {id} has been added at ({x},{y}) with target {t}".format(x=position[0],y=position[1], t=_target, id=ped_id))

    @property
    def targets(self):
        return self["scenario"]["topography"]["targets"]

    @property
    def pedetrian_ids(self):
        return [p["attributes"]["id"] for p in self["scenario"]["topography"]["dynamicElements"]]


# if __name__ == "__main__":
    # postvis = load_postvis_file("out/RiMEA 6 OSM_2021-05-10_13-13-24.738/postvis.traj")
    # scenario = Scenario("scenarios/RiMEA 6 OSM.scenario")
    # scenario.add_pedestrian((11.5,1.5))
    # scenario.save("scenarios/RiMEA 6 OSM_task3.scenario", "RiMEA 6 OSM Task3")
