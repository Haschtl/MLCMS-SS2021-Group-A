from scenario import Scenario

scenario = Scenario("scenarios/RiMEA 6 OSM.scenario")
scenario.add_pedestrian((11.5, 1.5))
scenario.save("scenarios/RiMEA 6 OSM_task3.scenario", "RiMEA 6 OSM Task3")
