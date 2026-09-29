"""Small directed road graph shared by model, routing, and traffic tests."""

import networkx as nx

from stgat_lstm.graph_data import NODE_CONTEXT_COLUMNS, build_real_graph_data_from_graph


def build_test_graph_data():
    graph = nx.MultiDiGraph(crs="EPSG:32651")
    graph.add_node("1", x=0.0, y=0.0, highway="traffic_signals")
    graph.add_node("2", x=100.0, y=0.0)
    graph.add_node("3", x=100.0, y=100.0)
    graph.add_node("4", x=200.0, y=0.0)
    graph.add_node("far", x=5_000.0, y=5_000.0)
    graph.add_edge("1", "2", key="0", name="Taft Avenue", highway="primary", length=100.0,
                   oneway="True", geometry="LINESTRING (0 0, 100 0)")
    graph.add_edge("2", "4", key="0", name="Main Road", highway="secondary", length=100.0,
                   geometry="LINESTRING (100 0, 200 0)")
    graph.add_edge("2", "3", key="0", name="Side Road", highway="residential", length=100.0,
                   geometry="LINESTRING (100 0, 100 100)")
    graph.add_edge("3", "4", key="0", name="Return Road", highway="residential", length=141.4,
                   geometry="LINESTRING (100 100, 200 0)")
    graph.add_edge("far", "far", key="0", name="Far Road", highway="service", length=1.0,
                   geometry="LINESTRING (5000 5000, 5001 5000)")
    rows = {
        node_id: {column: float(index + 1) for index, column in enumerate(NODE_CONTEXT_COLUMNS)}
        for node_id in ("1", "2", "3", "4", "far")
    }
    return build_real_graph_data_from_graph(graph, rows, corridor_buffer_m=500.0)
