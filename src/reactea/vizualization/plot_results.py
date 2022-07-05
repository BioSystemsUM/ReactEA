import ast

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from reactea.utilities.chem_utils import ChemUtils
from reactea.utilities.io import Loaders


class PlotResults:
    """
    Plots EA results by index.
    """

    def __init__(self, output_configs: dict, solution_index: int = 0):
        """
        Initializes the PlotResults class.

        Parameters
        ----------
        output_configs: dict
            Output configuration dictionary.
        solution_index: int
            Index of the solution to plot.
        """
        self.solution_index = solution_index
        self.configs = output_configs
        self.case = self.load_case()

    def load_case(self):
        """
        Loads the case to plot.

        Returns
        -------
        Case to plot.
        """
        return Loaders.load_results_case(self.solution_index, self.configs)

    def plot_results(self,
                     fig_size: tuple = (20, 20),
                     mol_size: float = 0.2,
                     label_pos: float = 0.5,
                     font_size: int = 20,
                     save_fig: bool = True):
        """
        Plots the case as a graph. With mol images as nodes and reaction rules ids as edges.

        Parameters
        ----------
        fig_size: tuple
            Figure size.
        mol_size: float
            Molecule size.
        label_pos: float
            Label position relative to the edge position (0.5 is in the middle).
        font_size: int
            Font size.
        save_fig: bool
            Whether to save the figure.
        """
        graph = nx.Graph()
        final = self.case.FINAL_SMILES
        intermediates = ast.literal_eval(self.case.INTERMEDIATE_SMILES)
        if len(intermediates) == 0:
            print('This molecule did not improve, thus no graph will be plotted!')
            return
        rules = ast.literal_eval(self.case.RULE_IDS)
        node_images = {}
        edge_labels = {}
        graph.add_node(intermediates[0])
        node_images[intermediates[0]] = ChemUtils.smiles_to_img(intermediates[0])
        for i in range(len(intermediates)-1):
            graph.add_node(intermediates[i+1])
            node_images[intermediates[i+1]] = ChemUtils.smiles_to_img(intermediates[i+1])
            graph.add_edge(intermediates[i], intermediates[i+1])
            edge_labels[(intermediates[i], intermediates[i+1])] = rules[i]
        graph.add_node(final)
        node_images[final] = ChemUtils.smiles_to_img(final, highlightMol=True)
        graph.add_edge(intermediates[-1], final)
        edge_labels[(intermediates[-1], final)] = rules[-1]
        self.draw_graph(graph=graph,
                        node_images=node_images,
                        edge_labels=edge_labels,
                        fig_size=fig_size,
                        mol_size=mol_size,
                        label_pos=label_pos,
                        font_size=font_size,
                        save_fig=save_fig)

    def draw_graph(self,
                   graph,
                   node_images,
                   edge_labels,
                   fig_size=(20, 20),
                   mol_size=0.2,
                   label_pos=0.5,
                   font_size=20,
                   save_fig=True):
        """
        Draws the result graph.

        Parameters
        ----------
        graph: nx.Graph
            Graph to draw.
        node_images: dict
            dict with node:images.
        edge_labels: dict
            dict with edge:label.
        fig_size: tuple
            Figure size.
        mol_size: float
            Molecule size.
        label_pos: float
            Label position relative to the edge position (0.5 is in the middle).
        font_size: int
            Font size.
        save_fig: bool
            Whether to save the figure.
        """
        pos = nx.circular_layout(graph)
        fig = plt.figure(figsize=fig_size)
        ax = plt.subplot()
        ax.set_aspect('equal')
        nx.draw_networkx_edges(graph, pos, ax=ax)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        p2 = mol_size / 2.0
        for n in graph:
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, mol_size, mol_size])
            a.set_aspect('equal')
            a.imshow(self.crop_image_with_transparency(node_images[n]))
            a.axis('off')
        ax.axis('off')
        if len(graph.nodes()) > 3:
            font_size = font_size / (len(graph.nodes())-2)
        nx.draw_networkx_edge_labels(graph,
                                     pos,
                                     edge_labels=edge_labels,
                                     label_pos=label_pos,
                                     font_size=font_size,
                                     ax=ax)
        plt.show()
        if save_fig:
            plt.savefig(f'outputs/{self.configs["exp_name"]}/result_{self.case}_graph.png')

    @staticmethod
    def crop_image_with_transparency(img):
        """
        Crops the image with transparency.

        Parameters
        ----------
        img: PIL.Image
            Image to crop.

        Returns
        -------
        Cropped image.
        """
        # Insuring the image has an alpha channel
        img.putalpha(255)

        # Image to numpy array
        image_data = np.array(img)

        # Computing the mask of white pixels
        r, g, b, a = np.rollaxis(image_data, axis=-1)
        white_pixels_mask = np.logical_and(np.logical_and(r == 255, g == 255), b == 255)

        # Replacing all white pixels by transparent pixels
        a[white_pixels_mask] = 0

        # Computing bounding box of non zero pixels
        l, u, r, b = Image.fromarray(image_data).getbbox()
        w, h = img.size

        mask = Image.new('L', img.size, color=255)
        epsilon = 10

        # Applying transparency
        # (https://stackoverflow.com/questions/4379978/python-pil-how-to-make-area-transparent-in-png)
        for transparent_zone in [(0, 0, l - epsilon, h), (0, 0, w, u - epsilon), (r + epsilon, 0, w, h),
                                 (0, b + epsilon, w, h)]:
            draw = ImageDraw.Draw(mask)
            draw.rectangle(transparent_zone, fill=0)
            img.putalpha(mask)

        return img
