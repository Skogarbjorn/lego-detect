from shapely.geometry import Polygon
import networkx as nx

def to_polygon(points):
    return Polygon([tuple(p) for p in points])

def buffer_percent(poly, percent):
    minx, miny, maxx, maxy = poly.bounds
    size = max(maxx - minx, maxy - miny)
    return poly.buffer(size * percent)

def calculate_connections(data):
    if not data:
        return None
    houses = data["houses"] 
    paths = data["paths"]

    name_counts = {}
    G = nx.Graph()

    margin = 0.5

    for house in houses:
        class_name = house["class"]
        count = name_counts.get(class_name, 0) + 1
        name_counts[class_name] = count

        if count == 1:
            node_name = class_name
        else:
            node_name = f"{class_name}{count}"

        G.add_node(node_name, kind="house", polygon=to_polygon(house["points"]))

    for i, path in enumerate(paths):
        G.add_node(f"path_{i}", kind="path", polygon=to_polygon(path["points"]))

    for hi, hdata in [(n, G.nodes[n]) for n in G if G.nodes[n]["kind"]=="house"]:
        for pj, pdata in [(n, G.nodes[n]) for n in G if G.nodes[n]["kind"]=="path"]:
            if buffer_percent(hdata["polygon"], margin).intersects(pdata["polygon"]):
                G.add_edge(hi, pj)

    for p1, pdata1 in [(n, G.nodes[n]) for n in G if G.nodes[n]["kind"]=="path"]:
        for p2, pdata2 in [(n, G.nodes[n]) for n in G if G.nodes[n]["kind"]=="path"]:
            if p1 < p2:
                if buffer_percent(pdata1["polygon"], margin).intersects(pdata2["polygon"]):
                    G.add_edge(p1, p2)

    return G
