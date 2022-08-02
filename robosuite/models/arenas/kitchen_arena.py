import numpy as np

from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion


class KitchenArena(TableArena):
    """
    Kitchen Environment

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/kitchen_arena.xml",
    ):
        super().__init__(table_full_size=table_full_size,
                         table_friction=table_friction,
                         table_offset=table_offset,
                         has_legs=has_legs,
                         xml=xml_path_completion(xml))