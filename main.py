# Importing required libraries and modules
import argparse, csv, math, sys, time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mip
from collections import deque


class SubTourCutGenerator(mip.ConstrsGenerator):
    def __init__(self, V, E, K, start_node):
        self.V = V
        self.E = E
        self.K = K
        self.start_node = start_node

    def generate_constrs(self, model, depth=0, npass=0):
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


def get_subtours(V, E, x, K, depot=0, tol=0.5):
    sub_tours = { k: [] for k in K }
    V_visited = { (k, v): False for k in K for v in V }

    # Inspect all continuous paths of the drone
    for k in K:
        for v in V:
            if V_visited[(k, v)]:
                continue
            V_visited[(k, v)] = True
            sub_tour = sub_tour_dfs(V, E, x, k, v, V_visited, tol=tol)
            if len(sub_tour) > 0:
                sub_tours[k].append(sub_tour)
    return sub_tours

def sub_tour_dfs(V, E, x, k, v, V_visited, tour=[], tol=0.5):
    for u in V:
        if (v, u) in E and x[(k, v, u)].x > tol:
            # Break cycles
            if V_visited[(k, u)]:
                return tour + [(v, u)]
            V_visited[(k, u)] = True
            return sub_tour_dfs(V, E, x, k, u, V_visited, tour + [(v, u)], tol)
    return tour

def process_coords(fname: str) -> list:
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
    d2: float = 0
    for i in range(ndims):
        d2 += (c1[i] - c2[i])**2
    return math.sqrt(d2)

def isEdge_type2(distance, c1: tuple, c2: tuple, md2: float, md3: float) -> bool:
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
    distances   : dict = {}
    edges       : list = []

    # Computing distances
    for i in V:
        for j in V:
            c1, c2 = coords[i], coords[j]
            ndims  = len(c1)
            distances[(i, j)] = dist(c1, c2, ndims=ndims)

    # Finding edges
    for edge, distance in distances.items():
        i, j = edge
        c1, c2 = coords[edge[0]], coords[edge[1]]

        if edge in edges:
            continue
        elif i == j:
            continue  # Exclude staying stationary
        elif i == start and j == end:
            continue  # Can't go from start to end
        elif i == end or j == start:
            continue  # No edges to start and no edges from end (potholes)
        elif i == start:
            if c2[1] <= entry_height:
                edges.append(edge)
        elif j == end:
            if c1[1] <= entry_height:
                edges.append(edge)
        elif distance <= md1 or isEdge_type2(distance, c1, c2, md2, md3):
            edges.append(edge)

    return distances, edges

def compute_weights(
    coords: list,
    V: list,
    E: dict,
    d: list
) -> dict:
    vh: float = 1.5
    vu: float = 1.0
    vd: float = 2.0

    weights: dict = {}
    for (i, j) in E:
        c1, c2 = coords[i], coords[j]
        h_dev = math.sqrt(
            (c1[0] - c2[0])**2 +
            (c1[2] - c2[2])**2
        )
        v_dev = c2[1] - c1[1]

        # Determining whether the motion is upward or downward
        vv: float = vu if v_dev >= 0 else vd
        weights[(i, j)] = max(
            h_dev / vh,
            abs(v_dev) / vv,
        )

    return weights

def print_drone_paths(x, E, K, start, tol=0.5):
    tours = get_subtours(V, E, x, K, tol=tol)
    for k in K:
        for tour in tours[k]:
            nodes = []
            if tour[0][0] == start:
                for (i, j) in tour:
                    nodes.append(str(i))
                nodes.append("0")
                print(f"Drone {k}: {'-'.join(nodes)}")

def visualize_graph_3d(coords, E, title="3D Graph"):
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

def get_subtours_fast(V, active_edges, K, start_node):
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

def is_valid_tour(tour, start_node):
    for (i, j) in tour:
        if i == start_node:
            return True
    return False


if __name__ == "__main__":
    # Global variables and constants
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
        if graph_show:
            visualize_graph_3d(coords, E)
        if sanity_check:
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

    # Edge activation
    x: dict = {
        (k, i, j): m.add_var(var_type=mip.BINARY, name=f"x_{k}_{i}_{j}") for k in K for (i, j) in E
    }

    # Variables indicating time visiting a node
    times = { k: m.add_var(var_type=mip.CONTINUOUS, lb=0.0, ub=len(V)*max(w.values())) for k in K }

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

    m.objective = mip.minimize(times[0])
    m.lazy_constrs_generator = SubTourCutGenerator(V, E, K, start_node=0)

    start   = time.perf_counter()
    outcome = m.optimize()
    end     = time.perf_counter()

    if verbose:
        print(f"=== Solution obtained in {(end-start)/60:.2f} minutes ===")
        print(outcome)

    print_drone_paths(x, E, K, 0, 0.5)

    if verbose:
        sub_tours = get_subtours(V, E, x, K, depot=0, tol=0.5)
        for k in K:
            print(f"{k} : {sub_tours[k]}")
        
        for k in K:
            print(f"Drone {k} time: {times[k].x}")
