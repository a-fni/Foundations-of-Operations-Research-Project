"""
=== Foundations of Operations Research small project ===

The following script solves a mixed-integer linear programming optimization problem whose objective
is to minimize the maxspan of drone travel times in a TSP problem (mTSP). The number of agents (or
drones here) is equal to 4 and two open test cases have been provided with respectively 64 and 200+
nodes that must (collaboratively) be visited while minimizing the longest travel time of the drones.

This project was developed in substitution to the lab assesment of the Foundations of Operations
Research MEng course at Politecnico di Milano (namely, if passed its score amounts to the 4 points
assigned by the lab test).
"""


# Importing required libraries and modules
import argparse, csv, math, sys, time
import matplotlib.pyplot as plt
import mip

from mpl_toolkits.mplot3d   import Axes3D
from collections            import deque


###################################################################################################
######################################### CUSTOM CLASSES ##########################################
###################################################################################################

class SubTourCutGenerator(mip.ConstrsGenerator):
    """
    Class in charge of creating a lazy-subtour detector and cut generator.
    The generate_constr method will add new cuts as soon as a new incumbent is found.
    """

    def __init__(self, V: list, E: list, K: list, start_node: int):
        """
        Main class constructor

        @arg V: list of vertices
        @arg E: list of edges between vertices as tuples (i, j)
        @arg K: list of drones (enumeration of incremental values)
        @arg start_node: node in V where drone routes should start
        """

        self.V = V
        self.E = E
        self.K = K
        self.start_node = start_node


    def generate_constrs(self, model, depth=0, npass=0) -> None:
        """
        Function in charge of generating cuts dynamically / lazily as
        new incumbents are found.

        @arg model: model which we are optimizing
        @arg depth: parameter required from super-class, not used
        @arg npass: parameter required from super-class, not used
        """

        # presolved variables
        vars_by_name = {v.name: v for v in model.vars}

        active_edges = {k: [] for k in self.K}

        # 1. read solution
        for k in self.K:
            for (i, j) in self.E:
                name = f"x_{k}_{i}_{j}"
                if name not in vars_by_name:
                    continue

                val = vars_by_name[name].x
                if val is not None and val > 0.9:
                    active_edges[k].append((i, j))

        if all(len(active_edges[k]) == 0 for k in self.K):
            return

        # 2. detect subtours
        all_tours = get_subtours_fast(
            self.V, active_edges, self.K, self.start_node
        )

        # 3. add lazy constraints
        for k in self.K:
            for tour in all_tours[k]:
                if is_valid_tour(tour, self.start_node):
                    continue

                nodes = set()
                for i, j in tour:
                    nodes.add(i)
                    nodes.add(j)

                expr = mip.xsum(
                    vars_by_name[f"x_{k}_{i}_{j}"]
                    for i in nodes
                    for j in nodes
                    if (i, j) in self.E
                    and f"x_{k}_{i}_{j}" in vars_by_name
                )

                model.add_lazy_constr(expr <= len(nodes) - 1)


###################################################################################################
######################################## UTILITY FUNCTIONS ########################################
###################################################################################################

def get_subtours(V: list, E: list, x: dict, K: list, depot: int=0, tol: float=0.5) -> dict:
    """
    Function in charge of finding all subtours given a set of paths for each drone

    @arg V: list of nodes of our graph
    @arg E: list of edges of our graph as pair (i, j) where i and j are in V
    @arg x: dict mapping triplets (k, i, j) to edge objects
    @arg K: list of drone incremental indices
    @arg depot: index of starting node in our network
    @arg tol: edge activation threshold

    @returns dict mapping drone index to its list of subtours
    """

    sub_tours = { k: [] for k in K }
    V_visited = { (k, v): False for k in K for v in V }

    # Inspect all continuous paths of the drone
    for k in K:
        for v in V:
            if V_visited[(k, v)]:
                continue
            V_visited[(k, v)] = True

            # Recursively build the subtour and check that it is of positive length
            sub_tour = sub_tour_dfs(V, E, x, k, v, V_visited, tol=tol)
            if len(sub_tour) > 0:
                sub_tours[k].append(sub_tour)

    return sub_tours

def sub_tour_dfs(
    V: list,
    E: list,
    x: dict,
    k: int,
    v: int,
    V_visited: list,
    tour: list=[],
    tol: float=0.5
) -> list:
    """
    Function in charge of recursively building a subtour from a starting node

    @arg V: list of nodes of our graph
    @arg E: list of edges of our graph as pair (i, j) where i and j are in V
    @arg x: dict mapping triplets (k, i, j) to edge objects
    @arg k: index of the drone for which we are currently building the subtour
    @arg v: index of node from which we are building the current subtour
    @arg V_visited: nodes in V we currently visited for this subtour
    @arg tour: list of nodes which currently make up our tour. Initially we have []
    @arg tol: edge activation threshold

    @returns list with the nodes making up the subtour
    """

    for u in V:
        if (v, u) in E and x[(k, v, u)].x > tol:
            
            # Break cycles while performing DFS visit
            if V_visited[(k, u)]:
                return tour + [(v, u)]
            V_visited[(k, u)] = True

            # Recursive call
            return sub_tour_dfs(V, E, x, k, u, V_visited, tour + [(v, u)], tol)
    
    # Return the tour we have constructed su far
    return tour

def process_coords(fname: str) -> list:
    """
    Function in charge of opening the dataframe file to read its coordinate values

    @arg fname: name of csv file to read

    @returns list of coordinates as a triplet of float values
    """

    coords: list = None
    with open(fname, "r") as f:
        reader = csv.reader(f)
        coords = list(reader)

    # Popping header row
    coords.pop(0)

    # Converting to float
    coords = [(float(x), float(y), float(z)) for (x, y, z) in coords]
    return coords

def dist(c1: tuple, c2: tuple, ndims: int=3) -> float:
    """
    Function in charge of computing the euclidean distance between two coordinates
    in a given number of dimensions

    @arg c1: triplet containing coordinates of first node
    @arg c2: triplet containing coordinates of second node
    @arg ndims: number of dimensions to consider

    @returns float representing the euclidean distance between the two nodes
    """

    d2: float = 0
    for i in range(ndims):
        d2 += (c1[i] - c2[i])**2
    return math.sqrt(d2)

def isEdge_type2(distance: float, c1: tuple, c2: tuple, md2: float, md3: float) -> bool:
    """
    Utility function to determine whether an pair of nodes might form a type-2 edge. The
    logic is the following: in at least two dimensions the difference between the respective
    coordinates must not exceed md3, while also not being at an overall euclidean distance
    greater than md2
    
    @arg distance: actual euclidean distance
    @arg c1: triplet containing coordinates of first node
    @arg c2: triplet containing coordinates of second node
    @arg md2: overall maximum euclidean distance between c1 and c2
    @arg md3: maximum coordinate difference that we must have in at least two dimensions

    @returns bool indicating whether c1->c2 is a type-2 edge
    """
    
    return distance <= md2 and (
        (abs(c1[0] - c2[0]) <= md3 and abs(c1[1] - c2[1]) <= md3) or
        (abs(c1[1] - c2[1]) <= md3 and abs(c1[2] - c2[2]) <= md3) or
        (abs(c1[0] - c2[0]) <= md3 and abs(c1[2] - c2[2]) <= md3)
    )

def determine_edges(
    V: list,
    coords: list,
    base_point: list,
    entry_height: list,
    start: int,
    end: int,
    md1: float=4.0,
    md2: float=11.0,
    md3: float=0.5,
) -> list:
    """
    Function in charge of determing which node pairs (i, j) should actually constitute
    an edge in our graph

    @arg V: list of nodes as indices
    @arg coords: list of nodes as coordinates. Note that coords[i] refers to node V[i]
    @arg base_point: coordinate of the starting point to our network
    @arg entry_height: y value (height) that nodes must have to be considered entry points
    @arg start: index of starting node in V (entry to our network)
    @arg end: index of ending node in V (destination in our network)
    @arg md1: maximum euclidean distance between two nodes for type-1 edge
    @arg md2: maximum euclidean distance between two nodes for type-2 edge
    @arg md3: maximum difference between at least two dimensions for type-2 edge

    WARNING: argument entry_point is unused in the current version of this function however
        it was required by a previous implementation and was hence kept for retro-compatibility
        reasons.

    @returns list of edges as pairs (i, j) where i and j are in V
    """

    distances   : dict = {}
    edges       : list = []

    # Computing euclidean distances
    for i in V:
        for j in V:
            c1, c2 = coords[i], coords[j]
            ndims  = len(c1)
            distances[(i, j)] = dist(c1, c2, ndims=ndims)

    # Finding actual edges
    for edge, distance in distances.items():
        i,  j  = edge
        c1, c2 = coords[edge[0]], coords[edge[1]]

        if edge in edges:
            # If the edge is already present, do not consider it again
            continue
        elif i == j:
            # Do not consider reflexive edges
            continue
        elif i == start and j == end:
            # Can't go from start to end
            continue
        elif i == end or j == start:
            # No edges reach the start and no edges leave the end (potholes)
            continue
        elif i == start:
            # If i is start node, use entry-point logic to determine edges
            if c2[1] <= entry_height:
                edges.append(edge)
        elif j == end:
            # If j is end node, use entry-point logic to determine edge
            if c1[1] <= entry_height:
                edges.append(edge)
        elif distance <= md1 or isEdge_type2(distance, c1, c2, md2, md3):
            # In all other cases, distance must be either less than md1, or
            # the edge must be such that deviation on two dimensions is less
            # than md3 while also having at most an euclidean distance of md2
            edges.append(edge)

    return distances, edges

def compute_weights(coords: list, V: list, E: list, d: list) -> dict:
    """
    Function in charge of computing the weights of each edge of the graph

    @arg coords: list of actual spacial coordinates as triplets
    @arg V: list of nodes, expressed as integers. NB: len(V) == len(coords)
    @arg E: list of edges of the graph as pairs (i, j), where i and j are in V
    @arg d: list of euclidean distances for each edge. NB: d[i] refers to edge E[i].

    WARNING: argument d is unused in the current version (was required by previous
        implementation and was kept for retro-compatibility reasons)

    @returns dict: mapping between edge (i, j) to its weight
    """

    # Weight factors for horizontal, vertical-up and vertical-down movements
    vh: float = 1.5
    vu: float = 1.0
    vd: float = 2.0

    weights: dict = {}
    for (i, j) in E:
        # Obtain for each edge in the graph the vertical and horizontal deviation
        c1, c2 = coords[i], coords[j]
        h_dev = math.sqrt(
            (c1[0] - c2[0])**2 +
            (c1[2] - c2[2])**2
        )
        v_dev = c2[1] - c1[1]

        # Determining whether the motion is upward or downward
        vv: float = vu if v_dev >= 0 else vd

        # Determine the actual weight of the edge
        weights[(i, j)] = max(
            h_dev / vh,
            abs(v_dev) / vv,
        )

    return weights

def print_drone_paths(x: dict, E: list, K: list, start: int, tol: float=0.5):
    """
    Function in charge of printing the paths of each drone

    @arg x: dict mapping triplets (k, i, j) to edge i->j value for drone k
    @arg E: list of edges available in the graph
    @arg K: list of drones as incremental indices
    @arg start: starting node in the graph
    @arg tol: threshold to determine whether an edge is to be considered active or not
    """

    # Obtain all subtours and then iterate on each one per drone
    tours = get_subtours(V, E, x, K, tol=tol)
    
    for k in K:
        # construct the path
        for tour in tours[k]:
            nodes = []
            if tour[0][0] == start:
                for (i, j) in tour:
                    nodes.append(str(i))

                # Print the tour
                nodes.append("0")
                print(f"Drone {k}: {'-'.join(nodes)}")

def visualize_graph_3d(coords: list, E: list, title="3D Graph"):
    """
    Function in charge of producing a 3D visualization of the graph at hand

    @arg coords: list of coordinates of nodes of the graph. Each item is a triplet
    @arg E: list of edges between nodes expressed as indices (i, j)
    @arg title: title of the graph to display
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    ax.scatter(xs, ys, zs, s=40)

    # Label each node with its index
    for idx, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(idx), fontsize=10)

    for (i, j) in E:
        x = [coords[i][0], coords[j][0]]
        y = [coords[i][1], coords[j][1]]
        z = [coords[i][2], coords[j][2]]
        ax.plot(x, y, z)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

def get_subtours_fast(V: list, active_edges: dict, K: list, start_node: int) -> list:
    """
    Function in charge of obtaining all subtours given a set of active edges

    @arg V: set of nodes
    @arg active_edges: dictionary that assignes to each drone index the set of its active edges
    @arg K: list of drone indices
    @arg start_node: starting node in the set of edges

    @returns list of subtours for each drone
    """

    subtours = {k: [] for k in K}
    adj = {k: {} for k in K}
    for k in K:
        for (i, j) in active_edges[k]:
            adj[k][i] = j

    for k in K:
        visited = set()

        if start_node in adj[k]:
            curr = start_node
            path = []
            while curr in adj[k]:
                visited.add(curr)
                nxt = adj[k][curr]
                path.append((curr, nxt))
                curr = nxt
                if curr in visited:
                    break

        nodes_in_route = list(adj[k].keys())
        for node in nodes_in_route:
            if node in visited:
                continue

            # If we find an unvisited node, it must be part of a disjoint cycle
            cycle = []
            curr = node
            while curr not in visited:
                visited.add(curr)
                if curr in adj[k]:
                    nxt = adj[k][curr]
                    cycle.append((curr, nxt))
                    curr = nxt
                else:
                    break # Should not happen in a cycle

                # If we wrap around to the start of this specific search
                if curr == node:
                    break

            if cycle:
                subtours[k].append(cycle)

    return subtours

def is_valid_tour(tour: list, start_node: int) -> bool:
    """
    Function in charge of checking whether a tour is valid or not. A tour is valid if it
    contains the starting node

    @arg tour: list of edges making up the tour
    @arg start_node: the starting node that most be present in the tour for it to be valid

    @returns bool indicating whether the tour is valid or not
    """

    for (i, j) in tour:
        if i == start_node:
            return True
    return False


###################################################################################################
######################################## MAIN SCRIPT LOGIC ########################################
###################################################################################################

if __name__ == "__main__":
    # Local variables and constants
    verbose     : bool = False
    sanity_check: bool = False
    graph_show  : bool = False

    n_drones: int = 4
    max_dist_1: float = 4.0
    max_dist_2: float = 11.0
    max_dist_3: float = 0.5

    fname       : str   = sys.argv[1]
    entry_height: float = -12.5         if "1" in fname else -20
    base_coords : tuple = (0, -16, 0)   if "1" in fname else (0, -40, 0)

    # Adding depot node at beginning and end (as fictitious end node)
    coords: list = [base_coords] + process_coords(fname) + [base_coords]
    start, end = 0, len(coords)-1

    # Set of drones
    K: list = [i for i in range(n_drones)]

    # Defining vertices and edges
    V: list = [i for i in range(len(coords))]
    d, E = determine_edges(
        V, coords,
        base_coords,
        entry_height,
        start,
        end
    )

    # Weights of edges
    w: dict = compute_weights(coords, V, E, d)
    if verbose:
        print(len(V), len(E), len(w), len(d), len(V)**2)
        
        # If required, display graphically the 3D graph loaded
        if graph_show:
            visualize_graph_3d(coords, E)

        # If required, check that dataset loading and preprocessing went well
        if sanity_check:

            # Check that each edge has a reflexive match unless it is a
            # generator or a pothole
            generators, potholes = 0, 0
            for edge in E:
                if edge[1] == start or edge[0] == end:
                    print(f"=== Looping found: {edge} ===")
                elif edge[0] == start:
                    print(f"Generator found: {edge}")
                    generators += 1
                elif edge[1] == end:
                    print(f"Pothole found: {edge}")
                    potholes += 1
                elif edge[::-1] not in E:
                    print(f"=== Missing edge reflexive: {edge} ===")

            print(f"\nFound {generators} generators and {potholes} potholes\n")
            if generators != potholes:
                print("=== Generators and potholes are NOT matching ===")
                exit(1)

            for v in V:
                count_in, count_out = 0, 0
                for u in V:
                    if (v, u) in E:
                        count_out += 1
                    if (u, v) in E:
                        count_in += 1
                print(f"Node {v} has balance = {count_in - count_out}, #IN = {count_in}, #OUT = {count_out}")
                if count_in != count_out and v != start and v != end:
                    print(f"=== Balance is NOT matching for node {v} ===")
                    exit(1)

    # Creating the model
    m: mip.Model = mip.Model()
    m.verbose = verbose

    # === Variables' definition ===

    # Edge activation
    x: dict = {
        (k, i, j): m.add_var(var_type=mip.BINARY, name=f"x_{k}_{i}_{j}") for k in K for (i, j) in E
    }

    # Variables indicating time visiting a node
    times = { k: m.add_var(var_type=mip.CONTINUOUS, lb=0.0, ub=len(V)*max(w.values())) for k in K }

    # === Constraints' definition ===

    # Ending time constraints
    for k in K:
        m.add_constr(
            times[k] >= mip.xsum(w[(i, j)] * x[(k, i, j)] for (i, j) in E)
        )

    # Leaving depot and returning to depot constraints
    for k in K:
        m.add_constr(mip.xsum(x[(k, start, j)] for j in V if (start, j) in E) == 1)
        m.add_constr(mip.xsum(x[(k, j, end)]   for j in V if (j, end)   in E) == 1)

    # Flow constraints
    for v in V[1:-1]:
        # Every node, except start and end, has #in == 1
        m.add_constr(mip.xsum(x[(k, v, j)] for k in K for j in V if (v, j) in E) == 1)

        # Every drone that enters a node, except start and end, must also leave it
        for k in K:
            m.add_constr(
                mip.xsum(x[(k, v, j)] for j in V if (v, j) in E) -
                mip.xsum(x[(k, j, v)] for j in V if (j, v) in E) == 0
            )

    # Symmetry breaking
    for k in range(n_drones-1):
        m.add_constr(
            times[k] >= times[k+1]
        )

    if verbose:
        print(f"Variables: {len(m.vars)}, constraints: {len(m.constrs)}\n")

    # === Objective function definition ===
    m.objective = mip.minimize(times[0])

    # Checking for subtours lazily to improve efficiency
    m.lazy_constrs_generator = SubTourCutGenerator(V, E, K, start_node=0)

    # Optimizing while benchmarking execution times
    start   = time.perf_counter()
    outcome = m.optimize()
    end     = time.perf_counter()

    # Printing final balanced paths
    if verbose:
        print(f"=== Solution obtained in {(end-start)/60:.2f} minutes ===")
        print(outcome)
    print_drone_paths(x, E, K, 0, 0.5)

    # If required printing all subtours (should be one for each drone) and times
    if verbose:
        sub_tours = get_subtours(V, E, x, K, depot=0, tol=0.5)
        for k in K:
            print(f"{k} : {sub_tours[k]}")
        
        for k in K:
            print(f"Drone {k} time: {times[k].x}")
