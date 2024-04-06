from openqdc.datasets.interaction.L7 import L7
from openqdc.methods import InteractionMethod, InterEnergyType


class X40(L7):
    """
    X40 interaction dataset of 40 dimer pairs as
    introduced in the following paper:

    Benchmark Calculations of Noncovalent Interactions of Halogenated Molecules
    Jan Řezáč, Kevin E. Riley, and Pavel Hobza
    Journal of Chemical Theory and Computation 2012 8 (11), 4285-4292
    DOI: 10.1021/ct300647k

    Dataset retrieved and processed from:
    http://cuby4.molecular.cz/dataset_x40.html
    """

    __name__ = "x40"
    __energy_methods__ = [
        InteractionMethod.CCSD_T_CBS,  # "CCSD(T)/CBS",
        InteractionMethod.MP2_CBS,  # "MP2/CBS",
        InteractionMethod.DCCSDT_HA_DZ,  # "dCCSD(T)/haDZ",
        InteractionMethod.DCCSDT_HA_TZ,  # "dCCSD(T)/haTZ",
        InteractionMethod.MP2_5_CBS_ADZ,  # "MP2.5/CBS(aDZ)",
    ]
    __energy_type__ = [
        InterEnergyType.TOTAL,
    ] * 5

    energy_target_names = []

    def _process_name(self, item):
        return item.shortname
