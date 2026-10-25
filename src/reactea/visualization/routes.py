import ast
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol, rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Geometry import Point2D

from reactea.io_streams import Loaders

HIGHLIGHT_COLOUR = (1.0, 0.72, 0.42)
TEXT_COLOUR = (0.15, 0.15, 0.15)
MUTED_COLOUR = (0.45, 0.45, 0.45)
BACKGROUND_COLOUR = (1.0, 1.0, 1.0)


@dataclass
class Route:
    """
    Transformation route of a solution, from the seed compound to the final compound.

    Attributes
    ----------
    smiles: List[str]
        SMILES of the seed, the intermediate compounds and the final compound (in this order).
    rule_ids: List[str]
        ids of the reaction rules applied at each step (len(smiles) - 1).
    scores: Dict[str, float]
        objective values of the final compound.
    index: Optional[int]
        index of the solution in the final population.
    """
    smiles: List[str]
    rule_ids: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    index: Optional[int] = None

    @property
    def n_steps(self) -> int:
        """
        Number of reaction steps of the route.

        Returns
        -------
        int:
            number of reaction steps.
        """
        return len(self.rule_ids)


def load_routes(path: Union[str, Path]) -> List[Route]:
    """
    Loads the transformation routes of the final population of a ReactEA run.

    Parameters
    ----------
    path: Union[str, Path]
        a FINAL_TRANSFORMATIONS_*.csv file, or the run's output folder (the most recent file is used).

    Returns
    -------
    List[Route]:
        one route per solution of the final population (same order as the results files).
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob('FINAL_TRANSFORMATIONS_*.csv'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No FINAL_TRANSFORMATIONS_*.csv file found in {path}")
        path = candidates[-1]
    transformations = pd.read_csv(path, sep=';', header=0)

    # FINAL_<time>.csv has the objective values, in the same order as the transformations file
    scores_path = path.with_name(path.name.replace('FINAL_TRANSFORMATIONS_', 'FINAL_'))
    scores = pd.read_csv(scores_path, sep=';', header=0) if scores_path.exists() else None
    if scores is not None and len(scores) != len(transformations):
        scores = None

    routes = []
    for i, row in enumerate(transformations.itertuples(index=False)):
        route_scores = {}
        if scores is not None:
            route_scores = {k: float(v) for k, v in scores.iloc[i].items() if k != 'SMILES'}
        routes.append(Route(smiles=ast.literal_eval(row.INTERMEDIATE_SMILES) + [row.FINAL_SMILES],
                            rule_ids=[str(rule_id) for rule_id in ast.literal_eval(row.RULE_IDS)],
                            scores=route_scores,
                            index=i))
    return routes


@lru_cache(maxsize=1)
def _rules_metadata() -> Dict[str, Tuple[Optional[str], List[str]]]:
    """
    Maps each reaction rule id (InternalID and RuleID) to its EC numbers and co-reactants.

    Returns
    -------
    Dict[str, Tuple[Optional[str], List[str]]]:
        rule id -> (EC numbers, co-reactants SMILES)
    """
    metadata = {}
    for row in Loaders.load_rules_metadata().itertuples(index=False):
        ec_numbers = row.EC_Numbers if isinstance(row.EC_Numbers, str) else None
        reactants = row.Reactants.split(';') if isinstance(row.Reactants, str) else []
        coreactants = [r for r in reactants if r != 'Any']
        metadata[row.InternalID] = metadata[row.RuleID] = (ec_numbers, coreactants)
    return metadata


def _rule_labels(rule_id: str, max_ec_numbers: int = 2) -> Tuple[List[str], List[str]]:
    """
    Text to draw above (EC numbers) and below (co-reactants and rule id) a reaction arrow.

    Parameters
    ----------
    rule_id: str
        reaction rule id.
    max_ec_numbers: int
        maximum number of EC numbers to show.

    Returns
    -------
    Tuple[List[str], List[str]]:
        lines above the arrow and lines below the arrow.
    """
    ec_numbers, coreactants = _rules_metadata().get(rule_id, (None, []))
    above = []
    if ec_numbers == 'spontaneous_reaction':
        above.append('spontaneous')
    elif ec_numbers:
        ecs = ec_numbers.split(';')
        above = [f'EC {ec}' for ec in ecs[:max_ec_numbers]]
        if len(ecs) > max_ec_numbers:
            above[-1] += f' (+{len(ecs) - max_ec_numbers})'
    below = []
    for smiles in coreactants[:2]:
        mol = Chem.MolFromSmiles(smiles)
        below.append(f'+ {CalcMolFormula(mol)}' if mol is not None else '+ co-reactant')
    if len(coreactants) > 2:
        below.append(f'+ {len(coreactants) - 2} more')
    below.append(rule_id)
    return above, below


def _changed_atoms(previous: Mol, mol: Mol, pattern: Mol) -> List[int]:
    """
    Atoms of `mol` changed by the reaction that transformed `previous` into `mol`.

    Parameters
    ----------
    previous: Mol
        molecule before the reaction.
    mol: Mol
        molecule after the reaction.
    pattern: Mol
        maximum common substructure of both molecules.

    Returns
    -------
    List[int]:
        indices of the changed atoms of `mol`.
    """
    match = mol.GetSubstructMatch(pattern)
    previous_match = previous.GetSubstructMatch(pattern)
    if not match or not previous_match:
        return []
    changed = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in set(match)}
    for previous_idx, idx in zip(previous_match, match):
        before, after = previous.GetAtomWithIdx(previous_idx), mol.GetAtomWithIdx(idx)
        if before.GetDegree() != after.GetDegree() or before.GetTotalNumHs() != after.GetTotalNumHs():
            changed.add(idx)
    return sorted(changed)


def _shorten(text: str, max_chars: int) -> str:
    """
    Shortens a text to at most `max_chars` characters, keeping its start and end.

    Parameters
    ----------
    text: str
        text to shorten.
    max_chars: int
        maximum number of characters.

    Returns
    -------
    str:
        shortened text.
    """
    if len(text) <= max_chars:
        return text
    keep = max(1, (max_chars - 3) // 2)
    return f'{text[:keep]}...{text[-keep:]}'


def _prepare_molecules(smiles: List[str],
                       align: bool,
                       highlight_changes: bool,
                       mcs_timeout: int) -> Tuple[List[Optional[Mol]], List[List[int]]]:
    """
    Computes the 2D depictions of the route's molecules and the atoms changed at each step.

    Each molecule is aligned to the previous one using their maximum common substructure (MCS). The changed atoms
    are the ones outside the MCS (added or modified) and the ones in the MCS whose neighbours or hydrogens changed
    (e.g. where a group was removed).

    Parameters
    ----------
    smiles: List[str]
        SMILES of the route's molecules.
    align: bool
        whether to align each molecule to the previous one.
    highlight_changes: bool
        whether to compute the changed atoms.
    mcs_timeout: int
        MCS timeout (seconds) for each pair of molecules.

    Returns
    -------
    Tuple[List[Optional[Mol]], List[List[int]]]:
        molecules (None if the SMILES is invalid) and the changed atoms of each molecule.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    changed_atoms = [[] for _ in mols]
    previous = None
    for i, mol in enumerate(mols):
        if mol is None:
            previous = None
            continue
        rdDepictor.Compute2DCoords(mol)
        if previous is not None and (align or highlight_changes):
            mcs = rdFMCS.FindMCS([previous, mol], timeout=mcs_timeout, ringMatchesRingOnly=True)
            if mcs.numAtoms > 0:
                pattern = Chem.MolFromSmarts(mcs.smartsString)
                if align:
                    try:
                        rdDepictor.GenerateDepictionMatching2DStructure(mol, previous, refPatt=pattern)
                    except ValueError:
                        # keep the default depiction
                        pass
                if highlight_changes:
                    changed_atoms[i] = _changed_atoms(previous, mol, pattern)
        previous = mol
    return mols, changed_atoms


class RouteDrawing:
    """
    Drawing of a transformation route as a reaction scheme: seed -> intermediates -> final compound.

    Displayed inline in Jupyter notebooks; use `save` to export it as SVG (vector) or PNG.
    """

    def __init__(self,
                 route: Route,
                 ncols: int = 4,
                 panel_size: Tuple[int, int] = (300, 260),
                 arrow_width: int = 170,
                 font_size: int = 13,
                 align: bool = True,
                 highlight_changes: bool = True,
                 show_rule_info: bool = True,
                 show_scores: bool = True,
                 mcs_timeout: int = 2):
        """
        Initializes a RouteDrawing.

        Parameters
        ----------
        route: Route
            route to draw.
        ncols: int
            maximum number of molecules per row (longer routes wrap into several rows).
        panel_size: Tuple[int, int]
            size (width, height) of each molecule's panel.
        arrow_width: int
            width of the space between molecules where the reaction arrows are drawn.
        font_size: int
            font size of the labels.
        align: bool
            whether to align each molecule to the previous one.
        highlight_changes: bool
            whether to highlight the atoms changed at each step.
        show_rule_info: bool
            whether to label the arrows with the rules' EC numbers, co-reactants and ids (only the ids otherwise).
        show_scores: bool
            whether to show the objective values below the final compound.
        mcs_timeout: int
            MCS timeout (seconds) for each pair of molecules (used to align and highlight).
        """
        self.route = route
        self.ncols = ncols
        self.panel_size = panel_size
        self.arrow_width = arrow_width
        self.font_size = font_size
        self.show_rule_info = show_rule_info
        self.show_scores = show_scores
        self._mols, self._changed_atoms = _prepare_molecules(route.smiles, align, highlight_changes, mcs_timeout)
        self._svg = None

    def _legend(self, i: int) -> List[str]:
        """
        Legend lines of the i-th molecule of the route.

        Parameters
        ----------
        i: int
            molecule index.

        Returns
        -------
        List[str]:
            legend lines.
        """
        if self.route.n_steps == 0:
            legend = ['Seed (unchanged)']
        elif i == 0:
            legend = ['Seed']
        elif i == len(self._mols) - 1:
            legend = ['Final']
        else:
            legend = [f'Step {i}']
        if i == len(self._mols) - 1 and self.show_scores:
            legend += [f'{name}: {value:.3g}' for name, value in self.route.scores.items()]
        return legend

    @property
    def _line_height(self) -> float:
        return self.font_size * 1.4

    def _layout(self) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        Computes the top-left corner of each molecule's panel and the size of the drawing.
        Each row has the molecule panels and, below them, a band for the legends.
        Rows after the first one start with the arrow coming from the previous row.

        Returns
        -------
        Tuple[List[Tuple[float, float]], float, float]:
            panel positions, width and height.
        """
        panel_width, panel_height = self.panel_size
        legend_lines = 1 + (len(self.route.scores) if self.show_scores else 0)
        row_height = panel_height + legend_lines * self._line_height + self.font_size
        positions = []
        width = 0
        for i in range(len(self._mols)):
            row, col = divmod(i, self.ncols)
            x = (self.arrow_width if row > 0 else 0) + col * (panel_width + self.arrow_width)
            positions.append((x, row * row_height))
            width = max(width, x + panel_width)
        height = ((len(self._mols) - 1) // self.ncols + 1) * row_height
        return positions, width, height

    def _draw(self, drawer_class, scale: float) -> rdMolDraw2D.MolDraw2D:
        """
        Draws the route.

        Parameters
        ----------
        drawer_class:
            rdMolDraw2D.MolDraw2DSVG or rdMolDraw2D.MolDraw2DCairo.
        scale: float
            scale factor of the drawing.

        Returns
        -------
        rdMolDraw2D.MolDraw2D:
            the finished drawer.
        """
        positions, width, height = self._layout()
        panel_width, panel_height = self.panel_size
        drawer = drawer_class(int(width * scale), int(height * scale),
                              int(panel_width * scale), int(panel_height * scale))
        options = drawer.drawOptions()
        options.fixedBondLength = 28 * scale
        options.setBackgroundColour(BACKGROUND_COLOUR)
        line_height = self._line_height * scale

        # background of the whole canvas (the molecule panels only clear their own area)
        drawer.SetOffset(0, 0)
        drawer.SetFillPolys(True)
        drawer.SetColour(BACKGROUND_COLOUR)
        drawer.DrawRect(Point2D(0, 0), Point2D(width * scale, height * scale), True)

        for i, (mol, changed) in enumerate(zip(self._mols, self._changed_atoms)):
            x, y = positions[i]
            drawer.SetOffset(int(x * scale), int(y * scale))
            if mol is None:
                drawer.SetFontSize(self.font_size * scale)
                drawer.SetColour(MUTED_COLOUR)
                drawer.DrawString('invalid SMILES', Point2D(panel_width * scale / 2, panel_height * scale / 2), 0,
                                  True)
                continue
            changed_bonds = [bond.GetIdx() for bond in mol.GetBonds()
                             if bond.GetBeginAtomIdx() in changed or bond.GetEndAtomIdx() in changed]
            drawer.DrawMolecule(mol,
                                highlightAtoms=changed,
                                highlightBonds=changed_bonds,
                                highlightAtomColors={a: HIGHLIGHT_COLOUR for a in changed},
                                highlightBondColors={b: HIGHLIGHT_COLOUR for b in changed_bonds})

        drawer.SetOffset(0, 0)
        drawer.SetFontSize(self.font_size * scale)
        for i, (x, y) in enumerate(positions):
            for k, line in enumerate(self._legend(i)):
                # the first line (Seed, Step i, Final) is darker than the scores
                drawer.SetColour(TEXT_COLOUR if k == 0 else MUTED_COLOUR)
                drawer.DrawString(line, Point2D((x + panel_width / 2) * scale,
                                                (y + panel_height) * scale + line_height * (k + 0.5)), 0, True)

        drawer.SetLineWidth(max(1, int(1.5 * scale)))
        max_chars = int(self.arrow_width / (self.font_size * 0.6))
        for step, rule_id in enumerate(self.route.rule_ids, start=1):
            if step >= len(positions):
                break
            x, y = positions[step]
            x_start, x_end = (x - self.arrow_width + 12) * scale, (x - 12) * scale
            y_arrow = (y + panel_height / 2) * scale
            drawer.SetColour(TEXT_COLOUR)
            drawer.DrawArrow(Point2D(x_start, y_arrow), Point2D(x_end, y_arrow), False, 0.08, 0.5, TEXT_COLOUR,
                             True)
            above, below = _rule_labels(rule_id) if self.show_rule_info else ([], [rule_id])
            above = [_shorten(line, max_chars) for line in above]
            below = [_shorten(line, max_chars) for line in below]
            x_middle = (x_start + x_end) / 2
            for k, line in enumerate(reversed(above)):
                drawer.DrawString(line, Point2D(x_middle, y_arrow - line_height * (k + 1)), 0, True)
            for k, line in enumerate(below):
                # the rule id (last line) is muted
                drawer.SetColour(MUTED_COLOUR if k == len(below) - 1 else TEXT_COLOUR)
                drawer.DrawString(line, Point2D(x_middle, y_arrow + line_height * (k + 1)), 0, True)
        drawer.FinishDrawing()
        return drawer

    @property
    def svg(self) -> str:
        """
        SVG of the drawing.

        Returns
        -------
        str:
            SVG text.
        """
        if self._svg is None:
            self._svg = self._draw(rdMolDraw2D.MolDraw2DSVG, scale=1).GetDrawingText()
        return self._svg

    def to_png(self, scale: float = 2) -> bytes:
        """
        PNG of the drawing.

        Parameters
        ----------
        scale: float
            resolution multiplier (2 doubles the width and height in pixels).

        Returns
        -------
        bytes:
            PNG data.
        """
        return self._draw(rdMolDraw2D.MolDraw2DCairo, scale=scale).GetDrawingText()

    def save(self, path: Union[str, Path], scale: float = 2):
        """
        Saves the drawing as SVG or PNG (chosen by the file extension).

        Parameters
        ----------
        path: Union[str, Path]
            output file (.svg or .png).
        scale: float
            resolution multiplier for PNG files.
        """
        path = Path(path)
        if path.suffix == '.svg':
            path.write_text(self.svg)
        elif path.suffix == '.png':
            path.write_bytes(self.to_png(scale))
        else:
            raise ValueError(f"Unsupported format '{path.suffix}' (use .svg or .png; SVG can be converted to PDF)")

    def _repr_svg_(self) -> str:
        return self.svg


def draw_route(route: Route, **kwargs) -> RouteDrawing:
    """
    Draws a transformation route as a reaction scheme (see RouteDrawing for the options).

    Parameters
    ----------
    route: Route
        route to draw.
    kwargs:
        drawing options (see RouteDrawing).

    Returns
    -------
    RouteDrawing:
        the drawing (displayed inline in notebooks; use `save` to export it).
    """
    return RouteDrawing(route, **kwargs)


def save_routes(routes: List[Route],
                output_dir: Union[str, Path],
                fmt: str = 'svg',
                skip_unchanged: bool = True,
                unique: bool = True,
                **kwargs) -> List[Path]:
    """
    Draws and saves several routes (one file per route, named after the solution's index).

    Parameters
    ----------
    routes: List[Route]
        routes to save.
    output_dir: Union[str, Path]
        folder where the files are saved (created if needed).
    fmt: str
        file format ('svg' or 'png').
    skip_unchanged: bool
        whether to skip the solutions that were not transformed (seed == final compound).
    unique: bool
        whether to skip repeated routes.
    kwargs:
        drawing options (see RouteDrawing).

    Returns
    -------
    List[Path]:
        saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    paths = []
    for route in routes:
        key = (tuple(route.smiles), tuple(route.rule_ids))
        if (skip_unchanged and route.n_steps == 0) or (unique and key in seen):
            continue
        seen.add(key)
        path = output_dir / f'route_{route.index}.{fmt}'
        draw_route(route, **kwargs).save(path)
        paths.append(path)
    return paths