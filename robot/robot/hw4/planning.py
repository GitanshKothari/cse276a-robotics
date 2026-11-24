import heapq
import math
from pathlib import Path

import numpy as np

import arena


def clearance(start, end, cylinder):
    a = np.asarray(start)[:2]
    b = np.asarray(end)[:2]
    center = np.asarray(cylinder[:2])
    delta = b - a
    length_sq = delta @ delta
    fraction = np.clip((center - a) @ delta / length_sq, 0, 1) if length_sq else 0
    box_min = np.asarray(arena.BOUNDS[:2])
    box_max = np.asarray(arena.BOUNDS[2:])
    wall = min(np.min(np.minimum(a, b) - box_min), np.min(box_max - np.maximum(a, b)))
    return min(float(wall), float(np.linalg.norm(a + fraction * delta - center) - cylinder[2]))


def search(edges, threshold=None):
    widest = threshold is None
    costs = {0: -math.inf if widest else 0.0}
    parents = {}
    queue = [(costs[0], 0)]
    
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != costs[u]:
            continue
        
        if u == 1:
            if widest:
                return -cost
            route = [1]
            while route[-1] != 0:
                route.append(parents[route[-1]])
            return route[::-1]
        
        for v, distance, gap in edges[u]:
            if not widest and gap < threshold - 1e-9:
                continue
            candidate = max(cost, -gap) if widest else cost + distance
            if candidate < costs.get(v, math.inf):
                costs[v], parents[v] = candidate, u
                heapq.heappush(queue, (candidate, v))
    
    raise RuntimeError("No usable route; check the arena, start, goal, and margin.")


def plan(start, goal, cylinder, method):
    margin = arena.ROBOT_RADIUS + arena.PLANNING_MARGIN
    nodes = [tuple(start[:2]), tuple(goal[:2])]
    edges = [[], []]

    def add(point):
        if clearance(point, point, cylinder) < margin - 1e-9:
            return None
        nodes.append(tuple(point))
        edges.append([])
        return len(nodes) - 1

    def connect(u, v):
        gap = clearance(nodes[u], nodes[v], cylinder)
        if gap >= margin - 1e-9:
            distance = math.dist(nodes[u], nodes[v])
            edges[u].append((v, distance, gap))
            edges[v].append((u, distance, gap))

    for point in nodes:
        if clearance(point, point, cylinder) < margin:
            raise RuntimeError("Start or goal is too close to the cylinder or wall.")

    xmin, ymin, xmax, ymax = arena.BOUNDS
    
    if method == "distance":
        center = np.asarray(cylinder[:2])
        ring_radius = (cylinder[2] + margin + 1e-5) / math.cos(math.pi / 32)
        
        for angle in np.linspace(0, 2 * math.pi, 32, endpoint=False):
            add(center + ring_radius * np.array([math.cos(angle), math.sin(angle)]))
        for u in range(len(nodes)):
            for v in range(u):
                connect(u, v)
    
    elif method == "safety":
        spacing = 0.08
        grid = {}
        
        for i, x in enumerate(np.arange(xmin, xmax, spacing)):
            for j, y in enumerate(np.arange(ymin, ymax, spacing)):
                index = add((x, y))
                if index is not None:
                    grid[i, j] = index
        
        for (i, j), u in grid.items():
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                v = grid.get((i + dx, j + dy))
                if v is not None:
                    connect(u, v)
        
        for u in (0, 1):
            for v in grid.values():
                if math.dist(nodes[u], nodes[v]) <= 2 * spacing:
                    connect(u, v)
        
        connect(0, 1)
    
    elif method == "prm":
        rng = np.random.default_rng(276)
        for point in rng.uniform((xmin, ymin), (xmax, ymax), size=(450, 2)):
            add(point)
        points = np.asarray(nodes)
        
        for u, point in enumerate(points):
            for v in np.argsort(np.linalg.norm(points - point, axis=1))[:17]:
                if v != u and not any(neighbor == v for neighbor, _, _ in edges[u]):
                    connect(u, v)
        if not any(v == 1 for v, _, _ in edges[0]):
            connect(0, 1)
    
    else:
        raise ValueError(f"Unknown planner {method}")

    threshold = search(edges) if method == "safety" else margin
    route = [nodes[i] for i in search(edges, threshold)]
    
    if method == "safety":
        index = 0
        while index < len(route) - 2:
            for end in range(len(route) - 1, index + 1, -1):
                if clearance(route[index], route[end], cylinder) >= threshold - 1e-9:
                    del route[index + 1:end]
                    break
            index += 1
    return route, nodes, edges


def plot(file, method, route, nodes, edges, cylinder, trace=(), mapped_tags=None):
    xmin, ymin, xmax, ymax = arena.BOUNDS
    scale = 520 / max(xmax - xmin, ymax - ymin)

    def screen(p):
        return 50 + scale * (p[0] - xmin), 55 + scale * (ymax - p[1])

    def points(path):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in map(screen, path))

    def line(path, color, width, opacity=1):
        return (f'<polyline points="{points(path)}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" opacity="{opacity}"/>')

    def dot(position, radius, color):
        x, y = screen(position)
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}"/>'

    def label(position, name):
        x, y = screen(position)
        return f'<text x="{x + 6:.1f}" y="{y - 5:.1f}" font-size="12">{name}</text>'

    width, height = (xmax - xmin) * scale, (ymax - ymin) * scale
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="645" font-family="sans-serif">',
           '<rect width="650" height="645" fill="white"/>',
           f'<text x="50" y="32" font-size="20">HW4 {method}</text>',
           f'<rect x="50" y="55" width="{width}" height="{height}" fill="#f8fafc" stroke="#334155"/>']
    margin = arena.ROBOT_RADIUS + arena.PLANNING_MARGIN
    inset = margin * scale
    svg.append(f'<rect x="{50 + inset}" y="{55 + inset}" width="{width - 2*inset}" '
               f'height="{height - 2*inset}" fill="none" stroke="#94a3b8" stroke-dasharray="4 3"/>')
    svg += [dot(cylinder, (cylinder[2] + margin) * scale, "#fed7aa"),
            dot(cylinder, cylinder[2] * scale, "#64748b")]
    if method in ("distance", "prm"):
        for u, neighbors in enumerate(edges):
            svg.extend(line([nodes[u], nodes[v]], "#64748b", 0.6, 0.18)
                       for v, _, _ in neighbors if u < v)
    svg.extend(dot(p, 1.5, "#87a9ba") for p in nodes[2:])
    for tag, p in arena.BOUNDARY_TAGS.items():
        svg.extend([dot(p, 3, "#334155"), label(p, f"Tag {tag}")])
    for tag, p in (mapped_tags or {}).items():
        svg.extend([dot(p, 4, "#a21caf"), label(p, f"Mapped {tag}")])
    if route:
        svg.append(line(route, "#1565c0", 3))
        svg.extend([dot(route[0], 4, "#1565c0"), label(route[0], "Start"),
                    dot(route[-1], 4, "#1565c0"), label(route[-1], "Goal")])
    if len(trace) > 1:
        svg.append(line(trace, "#b91c1c", 2))
    svg.append('<text x="50" y="610" font-size="13">Blue: planned route. Red: estimated robot trace. Orange: inflated obstacle.</text>')
    svg.append("</svg>")
    Path(file).write_text("\n".join(svg))
