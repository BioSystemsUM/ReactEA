import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


class PlotResults:

    def __init__(self, configs):
        self.configs = configs
        self.load_results()

    def load_results(self):
        results = pd.read_csv(self.configs[''])

    def plot_results(self):
        pass

    # function to draw a graph with images as nodes
    def draw_graph(self, graph, node_image_map):
        # create a new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        # draw the graph
        nx.draw(graph, pos=nx.spring_layout(graph), node_image=node_image_map, node_size=10, node_color='#A0CBE2',
                with_labels=False, font_size=8, ax=ax)
        # show the graph
        plt.show()
