# vrp_battery.py
import docplex
print(docplex.__version__)
from docplex.mp.model import Model
import numpy as np
import json
import math
from datetime import datetime

def haversine_distance(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Radius of Earth in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # in kilometers

    return distance

def load_data():
    import os
    import json

    # Paths to your data files
    data_dir = './filtered_data/'
    route_data_file = os.path.join(data_dir, 'filtered_route_data.json')
    package_data_file = os.path.join(data_dir, 'filtered_package_data.json')
    travel_times_file = os.path.join(data_dir, 'filtered_travel_times.json')

    # Load route data
    with open(route_data_file, 'r') as f:
        route_data = json.load(f)

    # Load package data
    with open(package_data_file, 'r') as f:
        package_data = json.load(f)

    # Load travel times
    with open(travel_times_file, 'r') as f:
        travel_times_data = json.load(f)

    # Process the first route in the data
    route_id = list(route_data.keys())[0]
    route_info = route_data[route_id]

    # Get vehicle capacity (convert from cm^3 to m^3)
    vehicle_capacity = route_info['executor_capacity_cm3'] / 1e6  # Convert cm^3 to m^3

    # Print vehicle capacity for verification
    print(f"Original vehicle capacity: {vehicle_capacity} m^3")

    # Adjust vehicle capacity if necessary
    vehicle_capacity *= 2  # Example: double the capacity
    print(f"Adjusted vehicle capacity: {vehicle_capacity} m^3")

    # Get the list of stops
    stops_info = route_info['stops']
    stop_ids = list(stops_info.keys())

    # Map stop IDs to indices starting from 0 (for depot)
    stop_indices = {stop_id: idx for idx, stop_id in enumerate(stop_ids)}

    # Assuming the first stop is the depot
    depot_id = stop_ids[0]
    depot_index = stop_indices[depot_id]

    # Coordinates of stops
    coords = {}
    for stop_id, idx in stop_indices.items():
        lat = stops_info[stop_id]['lat']
        lng = stops_info[stop_id]['lng']
        coords[idx] = (lat, lng)

    # Demands at each stop
    demand = {}
    for stop_id, idx in stop_indices.items():
        # Sum the volume of packages at this stop
        total_volume = 0
        if route_id in package_data and stop_id in package_data[route_id]:
            packages = package_data[route_id][stop_id]
            if packages:
                for pkg_id, pkg_info in packages.items():
                    dimensions = pkg_info['dimensions']
                    volume = (dimensions['depth_cm'] * dimensions['height_cm'] * dimensions['width_cm'])  # in cm^3
                    total_volume += volume
            else:
                print(f"Warning: No packages at stop_id {stop_id}")
        else:
            print(f"Warning: No package data for route_id {route_id} and stop_id {stop_id}")
        demand[idx] = total_volume / 1e6  # Convert cm^3 to m^3

    # Set demand at depot to zero
    demand[depot_index] = 0

    # Time windows at each stop
    time_windows = {}
    for stop_id, idx in stop_indices.items():
        # [Your existing code to set start_time and end_time]
        # For brevity, assuming default time windows
        start_time = 0
        end_time = 24 * 3600  # 24 hours in seconds
        time_windows[idx] = (start_time, end_time)
        # Optional: Print for verification
        print(f"Node {idx}: Start time = {start_time}, End time = {end_time}")

    # Service times at each stop
    service_time = {}
    for stop_id, idx in stop_indices.items():
        # [Your existing code to calculate service_time]
        # For brevity, setting default service times
        service_time[idx] = 60  # Default service time in seconds

    # Set service time at depot to zero
    service_time[depot_index] = 0

    # Assign num_nodes before building distance and time matrices
    num_nodes = len(coords)  # Total number of nodes including depot

    # Distance and travel time matrices
    distance = [[0] * num_nodes for _ in range(num_nodes)]
    time_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Build distance and time matrices
    for from_idx in coords:
        for to_idx in coords:
            if from_idx != to_idx:
                # Get travel time from travel_times_data
                from_stop_id = stop_ids[from_idx]
                to_stop_id = stop_ids[to_idx]
                travel_time = travel_times_data[route_id][from_stop_id][to_stop_id]  # in seconds
                time_matrix[from_idx][to_idx] = travel_time

                # Calculate distance based on coordinates (approximate)
                from_coord = coords[from_idx]
                to_coord = coords[to_idx]
                distance_km = haversine_distance(from_coord, to_coord)
                distance[from_idx][to_idx] = distance_km
            else:
                time_matrix[from_idx][to_idx] = 0
                distance[from_idx][to_idx] = 0

    # Collect user inputs
    num_vehicles = int(input("Enter number of vehicles: "))
    num_vehicles += 1  # Increase number of vehicles if necessary
    print(f"Adjusted number of vehicles: {num_vehicles}")
    vehicles = range(num_vehicles)

    # Energy consumption per unit distance (user input)
    energy_per_distance = float(input("Enter energy consumption per unit distance (e.g., kWh/km): "))

    # Battery capacity (user input)
    battery_capacity = float(input("Enter battery capacity (e.g., kWh): "))

    # Minimum battery level before recharging/swapping (user input)
    min_battery = float(input("Enter minimum battery level before recharging/swapping (e.g., kWh): "))

    # Charging time (user input, convert to seconds)
    charging_time_hours = float(input("Enter charging time at charging station (e.g., hours): "))
    charging_time = charging_time_hours * 3600  # Convert hours to seconds

    # Swapping time (user input, convert to seconds)
    swapping_time_hours = float(input("Enter swapping time at swapping station (e.g., hours): "))
    swapping_time = swapping_time_hours * 3600  # Convert hours to seconds

    # Scenario selection
    print("Select scenario:")
    print("1 - Battery Charging")
    print("2 - Battery Swapping")
    scenario = int(input("Enter scenario (1 or 2): "))

    # Build the data dictionary
    data = {
        'num_customers': num_nodes - 1,  # Exclude depot
        'customers': [i for i in coords.keys() if i != depot_index],
        'coords': coords,
        'demand': demand,
        'vehicle_capacity': vehicle_capacity,
        'num_vehicles': num_vehicles,
        'vehicles': vehicles,
        'distance': distance,
        'time_matrix': time_matrix,
        'energy_per_distance': energy_per_distance,
        'battery_capacity': battery_capacity,
        'min_battery': min_battery,
        'charging_time': charging_time,
        'swapping_time': swapping_time,
        'scenario': scenario,
        'time_windows': time_windows,
        'service_time': service_time,
        'depot_index': depot_index,
    }

    # Now, perform adjustments on 'data'

    # Adjust time windows
    for idx in data['time_windows']:
        data['time_windows'][idx] = (0, 86400)  # Set to full day
        print(f"Relaxed time window for node {idx}: Start time = 0, End time = 86400")

    # Simplify the problem by removing Node 3
    node_to_remove = 3
    if node_to_remove in data['customers']:
        data['customers'].remove(node_to_remove)
    if node_to_remove in data['coords']:
        del data['coords'][node_to_remove]
    if node_to_remove in data['demand']:
        del data['demand'][node_to_remove]
    if node_to_remove in data['time_windows']:
        del data['time_windows'][node_to_remove]
    if node_to_remove in data['service_time']:
        del data['service_time'][node_to_remove]

    # Update distance and time matrices
    node_indices = list(data['coords'].keys())
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(node_indices))}
    num_nodes = len(node_indices)  # Update num_nodes after removing a node

    distance = [[0] * num_nodes for _ in range(num_nodes)]
    time_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    for i_old, i_new in index_map.items():
        for j_old, j_new in index_map.items():
            distance[i_new][j_new] = data['distance'][i_old][j_old]
            time_matrix[i_new][j_new] = data['time_matrix'][i_old][j_old]

    data['coords'] = {index_map[i]: coord for i, coord in data['coords'].items()}
    data['demand'] = {index_map[i]: demand for i, demand in data['demand'].items()}
    data['time_windows'] = {index_map[i]: tw for i, tw in data['time_windows'].items()}
    data['service_time'] = {index_map[i]: st for i, st in data['service_time'].items()}
    data['distance'] = distance
    data['time_matrix'] = time_matrix
    data['depot_index'] = index_map[data['depot_index']]
    data['customers'] = [index_map[i] for i in data['customers']]

    # Update 'num_customers' in data dictionary
    data['num_customers'] = num_nodes - 1  # Exclude depot

    return data


def solve_vrp_battery(data):
    from docplex.mp.model import Model
    mdl = Model('VRP_with_Battery')

    # Decision variables
    x = mdl.binary_var_dict(
        ((i, j, k) for i in data['coords'] for j in data['coords'] for k in data['vehicles'] if i != j),
        name='x')

    t = mdl.continuous_var_dict(((i, k) for i in data['coords'] for k in data['vehicles']), lb=0, name='t')
    b = mdl.continuous_var_dict(((i, k) for i in data['coords'] for k in data['vehicles']), lb=0, name='b')

    # Objective: Minimize total distance
    mdl.minimize(mdl.sum(data['distance'][i][j] * x[i, j, k] for i in data['coords'] for j in data['coords']
                         for k in data['vehicles'] if i != j))

    # Constraints
    M = 1e6  # A large constant

    # Each customer is visited exactly once
    for i in data['customers']:
        mdl.add_constraint(mdl.sum(x[i, j, k] for j in data['coords'] for k in data['vehicles'] if i != j) == 1,
                           ctname=f"VisitOnce_{i}")

    # Flow conservation constraints
    for k in data['vehicles']:
        mdl.add_constraint(mdl.sum(x[data['depot_index'], j, k] for j in data['coords'] if j != data['depot_index']) == 1,
                           ctname=f"DepartDepot_{k}")
        mdl.add_constraint(mdl.sum(x[i, data['depot_index'], k] for i in data['coords'] if i != data['depot_index']) == 1,
                           ctname=f"ReturnDepot_{k}")

    for k in data['vehicles']:
        for h in data['coords']:
            if h != data['depot_index']:
                mdl.add_constraint(
                    mdl.sum(x[i, h, k] for i in data['coords'] if i != h) ==
                    mdl.sum(x[h, j, k] for j in data['coords'] if j != h),
                    ctname=f"FlowConservation_{h}_{k}"
                )

    # Vehicle capacity constraints
    for k in data['vehicles']:
        mdl.add_constraint(
            mdl.sum(data['demand'][i] * mdl.sum(x[i, j, k] for j in data['coords'] if i != j) for i in data['customers']) <= data['vehicle_capacity'],
            ctname=f"Capacity_{k}"
        )

    # Time window constraints
    for k in data['vehicles']:
        for i in data['coords']:
            mdl.add_constraint(t[i, k] >= data['time_windows'][i][0], ctname=f"TimeWindowStart_{i}_{k}")
            mdl.add_constraint(t[i, k] <= data['time_windows'][i][1], ctname=f"TimeWindowEnd_{i}_{k}")

    # Time constraints with travel time and service time
    for k in data['vehicles']:
        for i in data['coords']:
            for j in data['coords']:
                if i != j:
                    travel_time = data['time_matrix'][i][j]
                    mdl.add_constraint(
                        t[j, k] >= t[i, k] + data['service_time'][i] + travel_time - M * (1 - x[i, j, k]),
                        ctname=f"Time_{i}_{j}_{k}"
                    )

    # Battery level constraints
    for k in data['vehicles']:
        mdl.add_constraint(b[data['depot_index'], k] == data['battery_capacity'], ctname=f"BatteryStart_{k}")

    for k in data['vehicles']:
        for i in data['coords']:
            mdl.add_constraint(b[i, k] >= data['min_battery'], ctname=f"MinBattery_{i}_{k}")

    for k in data['vehicles']:
        for i in data['coords']:
            for j in data['coords']:
                if i != j:
                    energy_consumed = data['distance'][i][j] * data['energy_per_distance']
                    mdl.add_constraint(
                        b[j, k] >= b[i, k] - energy_consumed - M * (1 - x[i, j, k]),
                        ctname=f"Battery_{i}_{j}_{k}"
                    )

    # Charging or swapping constraints at depot
    if data['scenario'] == 1:
        # Battery Charging Scenario
        for k in data['vehicles']:
            for i in data['coords']:
                if i != data['depot_index']:
                    mdl.add_constraint(
                        t[data['depot_index'], k] >= t[i, k] + data['service_time'][i] + data['time_matrix'][i][data['depot_index']] + data['charging_time'] - M * (1 - x[i, data['depot_index'], k]),
                        ctname=f"ChargingTime_{i}_{k}"
                    )
                    mdl.add_constraint(
                        b[data['depot_index'], k] >= data['battery_capacity'] - M * (1 - x[i, data['depot_index'], k]),
                        ctname=f"BatteryRecharge_{i}_{k}"
                    )
    elif data['scenario'] == 2:
        # Battery Swapping Scenario
        for k in data['vehicles']:
            for i in data['coords']:
                if i != data['depot_index']:
                    mdl.add_constraint(
                        t[data['depot_index'], k] >= t[i, k] + data['service_time'][i] + data['time_matrix'][i][data['depot_index']] + data['swapping_time'] - M * (1 - x[i, data['depot_index'], k]),
                        ctname=f"SwappingTime_{i}_{k}"
                    )
                    mdl.add_constraint(
                        b[data['depot_index'], k] >= data['battery_capacity'] - M * (1 - x[i, data['depot_index'], k]),
                        ctname=f"BatterySwap_{i}_{k}"
                    )

    # Solver parameters
    #mdl.parameters.workmem = 1024  # Limit working memory to 1GB
    #mdl.parameters.mip.limits.treememory = 2048  # Limit tree memory to 2GB
    #mdl.parameters.mip.strategy.file = 3  # Node files are compressed and stored on disk
    #mdl.parameters.mip.strategy.nodeselect = 1  # Best-bound node selection
    #mdl.parameters.emphasis.memory = 1  # Emphasize memory conservation
    #mdl.parameters.timelimit = 3600  # Set a time limit of 1 hour
    #mdl.parameters.preprocessing.presolve = 0  # Disable presolve
    #mdl.parameters.simplex.tolerances.feasibility = 1e-9  # Tighten feasibility tolerance

    # Solve the model
    solution = mdl.solve(log_output=True)

    if solution:
        print("Total distance traveled: ", solution.objective_value)
        # Extract the routes
        for k in data['vehicles']:
            print(f"\nRoute for vehicle {k}:")
            route = [data['depot_index']]
            next_node = None
            current_node = data['depot_index']
            while True:
                for j in data['coords']:
                    if j != current_node and x[current_node, j, k].solution_value > 0.5:
                        route.append(j)
                        next_node = j
                        break
                if next_node == data['depot_index'] or next_node is None:
                    route.append(data['depot_index'])
                    break
                current_node = next_node
                next_node = None
            print(f"  Nodes visited: {route}")
            print(f"  Arrival times and battery levels:")
            for node in route:
                arrival_time = t[node, k].solution_value
                battery_level = b[node, k].solution_value
                print(f"  Node {node}: Arrival time = {arrival_time:.2f}, Battery level = {battery_level:.2f}")
    else:
        print("No solution found. Attempting to refine conflict...")

        # Refine the conflict
    from docplex.mp.conflict_refiner import ConflictRefiner

    conflict_refiner = ConflictRefiner()
    conflicts = conflict_refiner.refine_conflict(mdl)
    conflict_refiner.display_conflicts(conflicts)

if __name__ == '__main__':
    data = load_data()
    total_demand = sum(data['demand'][i] for i in data['customers'])
    print(f"Total demand: {total_demand}")
    solve_vrp_battery(data)