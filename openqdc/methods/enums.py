from enum import Enum, unique

from loguru import logger
from numpy import array, float32

from openqdc.methods.atom_energies import atom_energy_collection, to_e_matrix
from openqdc.utils.constants import ATOM_SYMBOLS


class StrEnum(str, Enum):
    def __str__(self):
        return self.value.lower()


@unique
class QmType(StrEnum):
    FF = "Force Field"
    SE = "Semi Empirical"
    DFT = "Density Functional Theory"
    HF = "Hartree Fork"
    CC = "Couple Cluster"
    MP2 = "Moller Plesset"


@unique
class InterEnergyType(StrEnum):  # InteractionEnergyType
    ES = "electrostatic"
    EX = "exchange"
    EX_S2 = "exchange S^2"
    IND = "induction"
    TOTAL = "total"
    EX_IND = "exchange-induction"
    DISP = "dispersion"
    EX_DISP_OS = "exchange dispersion opposite-spin"
    EX_DISP_SS = "exchange dispersion same-spin"
    DELTA_HF = "Delta HF vs SAPT0"


class BasisSet(StrEnum):
    NN = "nn"
    SZ = "sz"
    DZP = "dzp"
    TZP = "tzp"
    CBS = "cbs"
    HA_DZ = "haDZ"
    HA_TZ = "haTZ"
    CBS_ADZ = "cbs(adz)"
    STO3G = "sto-3g"
    GSTAR = "6-31g*"
    CC_PVDZ = "cc-pvdz"
    CC_PVTZ = "cc-pvtz"
    CC_PVQZ = "cc-pvqz"
    DEF2_SVP = "def2-svp"
    DEF2_DZVP = "def2-dzvp"
    DEF2_TZVP = "def2-tzvp"
    DEF2_TZVPPD = "def2-tzvppd"
    JUN_CC_PVDZ = "jun-cc-pvdz"
    AUG_CC_PWCVXZ = "aug-cc-pwcvxz"
    JUN_CC_PVDDZ = "jun-cc-pV(D+d)Z"
    AUG_CC_PVDDZ = "aug-cc-pV(D+d)Z"
    NONE = ""


class CORRECTION(StrEnum):
    D = "d"  # Grimme’s -D2 Dispersion Correction
    D1 = "d1"  # Grimme’s -D1 Dispersion Correction
    D3 = "d3"  # Grimme’s -D3 (zero-damping) Dispersion Correction
    D3BJ = "d3(bj)"  # Grimme’s -D3 (BJ-damping) Dispersion Correction
    D3M = "d3m"  # Grimme’s -D3 (zero-damping, short-range refitted) Dispersion Correction
    D3MBJ = "d3m(bj)"  # Grimme’s -D3 (BJ-damping, short-range refitted) Dispersion Correction
    D4 = "d4"  # Grimmes -D4 correction (we don t have any so feel free to not add this one)
    GCP = "gcp"  # Geometrical Counter-Poise Correction
    CP = "cp"  # Counter-Poise Correction
    MBD = "mbd"  # Many-Body Dispersion Correction/vdw-TS correction
    VWN = "vwn"  #
    VWN5 = "vwn5"  #
    NONE = ""


class Functional(Enum):
    B1LYP_VWN5 = "b1lyp", CORRECTION.VWN5
    B1PW91_VWN5 = "b1pw91", CORRECTION.VWN5
    B3LYP = "b3lyp"
    B3LYP_VWN5 = "b3lyp", CORRECTION.VWN5
    B3LYP_S_VWN5 = "b3lyp*", CORRECTION.VWN5
    B3LYPD = "b3lyp", CORRECTION.D
    B3LYP_D3_BJ = "b3lyp", CORRECTION.D3BJ
    B97 = "b97"
    B97_1 = "b97-1"
    B97_2 = "b97-2"
    B97_D = "b97", CORRECTION.D
    BECKE00 = "becke00"
    BECKE00_X_ONLY = "becke00-x-only"
    BECKE00X_XC = "becke00x(xc)"
    BECKE88X_BR89C = "becke88x+br89c"
    BHANDH = "bhandh"
    BHANDHLYP = "bhandhlyp"
    BLAP3 = "blap3"
    BLYP = "blyp"
    BLYPD = "blyp", CORRECTION.D
    BMTAU1 = "bmtau1"
    BOP = "bop"
    BP = "bp"
    BP86_D = "bp86", CORRECTION.D
    CCSD = "ccsd"
    CCSDT = "ccsd(t)"
    DCCSDT = "dccsd(t)"
    DFT3B = "dft3b"
    DLPNO_CCSDT = "dlpno-ccsd(t)"
    DLPNO_CCSDT0 = "dlpno-ccsd(t0)"
    DSD_BLYP_D3_BJ = "dsd-blyp", CORRECTION.D3BJ
    FIXED = "fixed"  # TODO: remove after cleaning the L7 dataset
    FN_DMC = "fn-dmc"
    FT97 = "ft97"
    GFN1_XTB = "gfn1_xtb"
    GFN2_XTB = "gfn2_xtb"
    HCTH = "hcth"
    HCTH_120 = "hcth-120"
    HCTH_147 = "hcth-147"
    HCTH_407 = "hcth-407"
    HCTH_93 = "hcth-93"
    HF = "hf"
    HF_R2SCAN_DC4 = "hf-r2scan-dc4"
    KCIS_MODIFIED = "kcis-modified"
    KCIS_ORIGINAL = "kcis-original"
    KMLYP_VWN5 = "kmlyp", CORRECTION.VWN5
    KT1 = "kt1"
    KT2 = "kt2"
    LDA_VWN = "lda", CORRECTION.VWN
    LNO_CCSDT = "lno-ccsd(t)"
    M05 = "m05"
    M05_2X = "m05-2x"
    M06 = "m06"
    M06_2X = "m06-2x"
    M06_L = "m06-l"
    MP2 = "MP2"
    MP2_5 = "MP2.5"
    MP2C = "MP2C"
    MPBE = "mpbe"
    MPBE0KCIS = "mpbe0kcis"
    MPBE1KCIS = "mpbe1kcis"
    MPBEKCIS = "mpbekcis"
    MPW = "mpw"
    MPW1K = "mpw1k"
    MPW1PW = "mpw1pw"
    MVS = "mvs"
    MVSX = "mvsx"
    O3LYP_VWN5 = "o3lyp", CORRECTION.VWN5
    OLAP3 = "olap3"
    OLYP = "olyp"
    OPBE = "opbe"
    OPBE0 = "opbe0"
    OPERDEW = "operdew"
    PBE = "pbe"
    PBE_D = "pbe", CORRECTION.D
    PBE_D3_BJ = "pbe", CORRECTION.D3BJ
    PBE0 = "pbe0"
    PBE0_MBD = "pbe0+mbd"
    PBESOL = "pbesol"
    PKZB = "pkzb"
    PKZBX_KCISCOR = "pkzbx-kciscor"
    PM6 = "pm6"
    PW91 = "pw91"
    QCISDT = "qcisd(t)"
    R2_SCAN = "r2Scan"
    R2_SCAN_HF = "r2Scan@hf"
    R2_SCAN_R2_SCAN50 = "r2Scan@r2Scan50"
    R2_SCAN50 = "r2Scan50"
    R2_SCAN100 = "r2Scan100"
    R2_SCAN10 = "r2Scan10"
    R2_SCAN20 = "r2Scan20"
    R2_SCAN25 = "r2Scan25"
    R2_SCAN30 = "r2Scan30"
    R2_SCAN40 = "r2Scan40"
    R2_SCAN60 = "r2Scan60"
    R2_SCAN70 = "r2Scan70"
    R2_SCAN80 = "r2Scan80"
    R2_SCAN90 = "r2Scan90"
    REVPBE = "revpbe"
    REVPBE_D3_BJ = "revpbe", CORRECTION.D3BJ
    REVTPSS = "revtpss"
    RGE2 = "rge2"
    RPBE = "rpbe"
    SAPT0 = "sapt0"
    SCAN = "scan"
    SCAN_HF = "scan@hf"
    SCAN_R2SCAN50 = "scan@r2scan50"
    SSB_D = "ssb", CORRECTION.D
    SVWN = "svwn"
    TMGGA = "t-mgga"
    TAU_HCTH = "tau-hcth"
    TAU_HCTH_HYBRID = "tau-hcth-hybrid"
    TCSSD_T = "tccsd(t)"
    TPSS = "tpss"
    TPSSD = "tpss", CORRECTION.D
    TPSSH = "tpssh"
    TTM2_1_F = "ttm2.1-f"
    VS98 = "vs98"
    VS98_X_XC = "vs98-x(xc)"
    VS98_X_ONLY = "vs98-x-only"
    WB97M_D3BJ = "wb97m", CORRECTION.D3BJ
    WB97X = "wb97x"
    WB97X_D = "wb97x", CORRECTION.D
    WB97X_D3 = "wb97x", CORRECTION.D3
    X3LYP_VWN5 = "x3lyp", CORRECTION.VWN5
    XLYP = "xlyp"
    NONE = ""

    def __init__(self, functional: str, correction: BasisSet = CORRECTION.NONE):
        self.functional = functional
        self.correction = correction

    def __str__(self):
        if self.correction != "":
            s = "-".join([str(self.functional), str(self.correction)])
        else:
            s = str(self.functional)
        return s


class QmMethod(Enum):
    def __init__(self, functional: Functional, basis_set: BasisSet, cost: float = 0):
        self.functional = functional
        self.basis_set = basis_set
        self.cost = cost

    def __str__(self):
        if self.basis_set != "":
            s = "/".join([str(self.functional), str(self.basis_set)])
        else:
            s = str(self.functional)
        return s

    @property
    def atom_energies_matrix(self):
        """Get the atomization energy matrix"""
        energies = self.atom_energies_dict
        mat = to_e_matrix(energies)

        return mat

    @property
    def atom_energies_dict(self):
        """Get the atomization energy dictionary"""
        raise NotImplementedError()


class PotentialMethod(QmMethod):  # SPLIT FOR INTERACTIO ENERGIES AND FIX MD1
    B1LYP_VWN5_DZP = Functional.B1LYP_VWN5, BasisSet.DZP
    B1LYP_VWN5_SZ = Functional.B1LYP_VWN5, BasisSet.SZ
    B1LYP_VWN5_TZP = Functional.B1LYP_VWN5, BasisSet.TZP
    B1PW91_VWN5_DZP = Functional.B1PW91_VWN5, BasisSet.DZP
    B1PW91_VWN5_SZ = Functional.B1PW91_VWN5, BasisSet.SZ
    B1PW91_VWN5_TZP = Functional.B1PW91_VWN5, BasisSet.TZP
    B3LYP_STO3G = Functional.B3LYP, BasisSet.STO3G  # TODO: calculate e0s
    B3LYP_VWN5_DZP = Functional.B3LYP_VWN5, BasisSet.DZP
    B3LYP_VWN5_SZ = Functional.B3LYP_VWN5, BasisSet.SZ
    B3LYP_VWN5_TZP = Functional.B3LYP_VWN5, BasisSet.TZP
    B3LYP_S_VWN5_DZP = Functional.B3LYP_S_VWN5, BasisSet.DZP
    B3LYP_S_VWN5_SZ = Functional.B3LYP_S_VWN5, BasisSet.SZ
    B3LYP_S_VWN5_TZP = Functional.B3LYP_S_VWN5, BasisSet.TZP
    B3LYP_D_DZP = Functional.B3LYPD, BasisSet.DZP
    B3LYP_D_SZ = Functional.B3LYPD, BasisSet.SZ
    B3LYP_D_TZP = Functional.B3LYPD, BasisSet.TZP
    B3LYP_D3_BJ_DEF2_TZVP = Functional.B3LYP_D3_BJ, BasisSet.DEF2_TZVP
    B3LYP_6_31G_D = Functional.B3LYP, BasisSet.GSTAR
    B3LYP_DEF2_TZVP = Functional.B3LYP, BasisSet.DEF2_TZVP
    B97_1_DZP = Functional.B97_1, BasisSet.DZP
    B97_1_SZ = Functional.B97_1, BasisSet.SZ
    B97_1_TZP = Functional.B97_1, BasisSet.TZP
    B97_2_DZP = Functional.B97_2, BasisSet.DZP
    B97_2_SZ = Functional.B97_2, BasisSet.SZ
    B97_2_TZP = Functional.B97_2, BasisSet.TZP
    B97_D_DZP = Functional.B97_D, BasisSet.DZP
    B97_D_SZ = Functional.B97_D, BasisSet.SZ
    B97_D_TZP = Functional.B97_D, BasisSet.TZP
    B97_DZP = Functional.B97, BasisSet.DZP
    B97_SZ = Functional.B97, BasisSet.SZ
    B97_TZP = Functional.B97, BasisSet.TZP
    BECKE00_X_ONLY_DZP = Functional.BECKE00_X_ONLY, BasisSet.DZP
    BECKE00_X_ONLY_SZ = Functional.BECKE00_X_ONLY, BasisSet.SZ
    BECKE00_X_ONLY_TZP = Functional.BECKE00_X_ONLY, BasisSet.TZP
    BECKE00_DZP = Functional.BECKE00, BasisSet.DZP
    BECKE00_SZ = Functional.BECKE00, BasisSet.SZ
    BECKE00_TZP = Functional.BECKE00, BasisSet.TZP
    BECKE00X_XC_DZP = Functional.BECKE00X_XC, BasisSet.DZP
    BECKE00X_XC_SZ = Functional.BECKE00X_XC, BasisSet.SZ
    BECKE00X_XC_TZP = Functional.BECKE00X_XC, BasisSet.TZP
    BECKE88X_BR89C_DZP = Functional.BECKE88X_BR89C, BasisSet.DZP
    BECKE88X_BR89C_SZ = Functional.BECKE88X_BR89C, BasisSet.SZ
    BECKE88X_BR89C_TZP = Functional.BECKE88X_BR89C, BasisSet.TZP
    BHANDH_DZP = Functional.BHANDH, BasisSet.DZP
    BHANDH_SZ = Functional.BHANDH, BasisSet.SZ
    BHANDH_TZP = Functional.BHANDH, BasisSet.TZP
    BHANDHLYP_DZP = Functional.BHANDHLYP, BasisSet.DZP
    BHANDHLYP_SZ = Functional.BHANDHLYP, BasisSet.SZ
    BHANDHLYP_TZP = Functional.BHANDHLYP, BasisSet.TZP
    BLAP3_DZP = Functional.BLAP3, BasisSet.DZP
    BLAP3_SZ = Functional.BLAP3, BasisSet.SZ
    BLAP3_TZP = Functional.BLAP3, BasisSet.TZP
    BLYP_D_DZP = Functional.BLYPD, BasisSet.DZP
    BLYP_D_SZ = Functional.BLYPD, BasisSet.SZ
    BLYP_D_TZP = Functional.BLYPD, BasisSet.TZP
    BLYP_DZP = Functional.BLYP, BasisSet.DZP
    BLYP_SZ = Functional.BLYP, BasisSet.SZ
    BLYP_TZP = Functional.BLYP, BasisSet.TZP
    BMTAU1_DZP = Functional.BMTAU1, BasisSet.DZP
    BMTAU1_SZ = Functional.BMTAU1, BasisSet.SZ
    BMTAU1_TZP = Functional.BMTAU1, BasisSet.TZP
    BOP_DZP = Functional.BOP, BasisSet.DZP
    BOP_SZ = Functional.BOP, BasisSet.SZ
    BOP_TZP = Functional.BOP, BasisSet.TZP
    BP_DZP = Functional.BP, BasisSet.DZP
    BP_SZ = Functional.BP, BasisSet.SZ
    BP_TZP = Functional.BP, BasisSet.TZP
    BP86_D_DZP = Functional.BP86_D, BasisSet.DZP
    BP86_D_SZ = Functional.BP86_D, BasisSet.SZ
    BP86_D_TZP = Functional.BP86_D, BasisSet.TZP
    CCSD_T_CBS = Functional.CCSDT, BasisSet.CBS
    CCSD_T_CC_PVTZ = Functional.CCSDT, BasisSet.CC_PVDZ
    CCSD_T_CC_PVDZ = Functional.CCSDT, BasisSet.CC_PVDZ
    CCSD_CC_PVDZ = Functional.CCSD, BasisSet.CC_PVDZ

    DFT3B = Functional.DFT3B, BasisSet.NONE
    DSD_BLYP_D3_BJ_DEF2_TZVP = Functional.DSD_BLYP_D3_BJ, BasisSet.DEF2_TZVP
    FT97_DZP = Functional.FT97, BasisSet.DZP
    FT97_SZ = Functional.FT97, BasisSet.SZ
    FT97_TZP = Functional.FT97, BasisSet.TZP
    GFN1_XTB = Functional.GFN1_XTB, BasisSet.NONE
    GFN2_XTB = Functional.GFN2_XTB, BasisSet.NONE
    HCTH_120_DZP = Functional.HCTH_120, BasisSet.DZP
    HCTH_120_SZ = Functional.HCTH_120, BasisSet.SZ
    HCTH_120_TZP = Functional.HCTH_120, BasisSet.TZP
    HCTH_147_DZP = Functional.HCTH_147, BasisSet.DZP
    HCTH_147_SZ = Functional.HCTH_147, BasisSet.SZ
    HCTH_147_TZP = Functional.HCTH_147, BasisSet.TZP
    HCTH_407_DZP = Functional.HCTH_407, BasisSet.DZP
    HCTH_407_SZ = Functional.HCTH_407, BasisSet.SZ
    HCTH_407_TZP = Functional.HCTH_407, BasisSet.TZP
    HCTH_93_DZP = Functional.HCTH_93, BasisSet.DZP
    HCTH_93_SZ = Functional.HCTH_93, BasisSet.SZ
    HCTH_93_TZP = Functional.HCTH_93, BasisSet.TZP
    HF_DEF2_TZVP = Functional.HF, BasisSet.DEF2_TZVP
    HF_CC_PVDZ = (
        Functional.HF,
        BasisSet.CC_PVDZ,
    )
    HF_CC_PVQZ = (
        Functional.HF,
        BasisSet.CC_PVQZ,
    )
    HF_CC_PVTZ = (
        Functional.HF,
        BasisSet.CC_PVTZ,
    )
    KCIS_MODIFIED_DZP = Functional.KCIS_MODIFIED, BasisSet.DZP
    KCIS_MODIFIED_SZ = Functional.KCIS_MODIFIED, BasisSet.SZ
    KCIS_MODIFIED_TZP = Functional.KCIS_MODIFIED, BasisSet.TZP
    KCIS_ORIGINAL_DZP = Functional.KCIS_ORIGINAL, BasisSet.DZP
    KCIS_ORIGINAL_SZ = Functional.KCIS_ORIGINAL, BasisSet.SZ
    KCIS_ORIGINAL_TZP = Functional.KCIS_ORIGINAL, BasisSet.TZP
    KMLYP_VWN5_DZP = Functional.KMLYP_VWN5, BasisSet.DZP
    KMLYP_VWN5_SZ = Functional.KMLYP_VWN5, BasisSet.SZ
    KMLYP_VWN5_TZP = Functional.KMLYP_VWN5, BasisSet.TZP
    KT1_DZP = Functional.KT1, BasisSet.DZP
    KT1_SZ = Functional.KT1, BasisSet.SZ
    KT1_TZP = Functional.KT1, BasisSet.TZP
    KT2_DZP = Functional.KT2, BasisSet.DZP
    KT2_SZ = Functional.KT2, BasisSet.SZ
    KT2_TZP = Functional.KT2, BasisSet.TZP
    LDA_VWN_DZP = Functional.LDA_VWN, BasisSet.DZP
    LDA_VWN_SZ = Functional.LDA_VWN, BasisSet.SZ
    LDA_VWN_TZP = Functional.LDA_VWN, BasisSet.TZP
    M05_2X_DZP = Functional.M05_2X, BasisSet.DZP
    M05_2X_SZ = Functional.M05_2X, BasisSet.SZ
    M05_2X_TZP = Functional.M05_2X, BasisSet.TZP
    M05_DZP = Functional.M05, BasisSet.DZP
    M05_SZ = Functional.M05, BasisSet.SZ
    M05_TZP = Functional.M05, BasisSet.TZP
    M06_2X_DZP = Functional.M06_2X, BasisSet.DZP
    M06_2X_SZ = Functional.M06_2X, BasisSet.SZ
    M06_2X_TZP = Functional.M06_2X, BasisSet.TZP
    M06_L_DZP = Functional.M06_L, BasisSet.DZP
    M06_L_SZ = Functional.M06_L, BasisSet.SZ
    M06_L_TZP = Functional.M06_L, BasisSet.TZP
    M06_DZP = Functional.M06, BasisSet.DZP
    M06_SZ = Functional.M06, BasisSet.SZ
    M06_TZP = Functional.M06, BasisSet.TZP
    MP2_CC_PVDZ = Functional.MP2, BasisSet.CC_PVDZ
    MP2_CC_PVQZ = Functional.MP2, BasisSet.CC_PVQZ
    MP2_CC_PVTZ = Functional.MP2, BasisSet.CC_PVTZ
    MPBE_DZP = Functional.MPBE, BasisSet.DZP
    MPBE_SZ = Functional.MPBE, BasisSet.SZ
    MPBE_TZP = Functional.MPBE, BasisSet.TZP
    MPBE0KCIS_DZP = Functional.MPBE0KCIS, BasisSet.DZP
    MPBE0KCIS_SZ = Functional.MPBE0KCIS, BasisSet.SZ
    MPBE0KCIS_TZP = Functional.MPBE0KCIS, BasisSet.TZP
    MPBE1KCIS_DZP = Functional.MPBE1KCIS, BasisSet.DZP
    MPBE1KCIS_SZ = Functional.MPBE1KCIS, BasisSet.SZ
    MPBE1KCIS_TZP = Functional.MPBE1KCIS, BasisSet.TZP
    MPBEKCIS_DZP = Functional.MPBEKCIS, BasisSet.DZP
    MPBEKCIS_SZ = Functional.MPBEKCIS, BasisSet.SZ
    MPBEKCIS_TZP = Functional.MPBEKCIS, BasisSet.TZP
    MPW_DZP = Functional.MPW, BasisSet.DZP
    MPW_SZ = Functional.MPW, BasisSet.SZ
    MPW_TZP = Functional.MPW, BasisSet.TZP
    MPW1K_DZP = Functional.MPW1K, BasisSet.DZP
    MPW1K_SZ = Functional.MPW1K, BasisSet.SZ
    MPW1K_TZP = Functional.MPW1K, BasisSet.TZP
    MPW1PW_DZP = Functional.MPW1PW, BasisSet.DZP
    MPW1PW_SZ = Functional.MPW1PW, BasisSet.SZ
    MPW1PW_TZP = Functional.MPW1PW, BasisSet.TZP
    MVS_DZP = Functional.MVS, BasisSet.DZP
    MVS_SZ = Functional.MVS, BasisSet.SZ
    MVS_TZP = Functional.MVS, BasisSet.TZP
    MVSX_DZP = Functional.MVSX, BasisSet.DZP
    MVSX_SZ = Functional.MVSX, BasisSet.SZ
    MVSX_TZP = Functional.MVSX, BasisSet.TZP
    O3LYP_VWN5_DZP = Functional.O3LYP_VWN5, BasisSet.DZP
    O3LYP_VWN5_SZ = Functional.O3LYP_VWN5, BasisSet.SZ
    O3LYP_VWN5_TZP = Functional.O3LYP_VWN5, BasisSet.TZP
    OLAP3_DZP = Functional.OLAP3, BasisSet.DZP
    OLAP3_SZ = Functional.OLAP3, BasisSet.SZ
    OLAP3_TZP = Functional.OLAP3, BasisSet.TZP
    OLYP_DZP = Functional.OLYP, BasisSet.DZP
    OLYP_SZ = Functional.OLYP, BasisSet.SZ
    OLYP_TZP = Functional.OLYP, BasisSet.TZP
    OPBE_DZP = Functional.OPBE, BasisSet.DZP
    OPBE_SZ = Functional.OPBE, BasisSet.SZ
    OPBE_TZP = Functional.OPBE, BasisSet.TZP
    OPBE0_DZP = Functional.OPBE0, BasisSet.DZP
    OPBE0_SZ = Functional.OPBE0, BasisSet.SZ
    OPBE0_TZP = Functional.OPBE0, BasisSet.TZP
    OPERDEW_DZP = Functional.OPERDEW, BasisSet.DZP
    OPERDEW_SZ = Functional.OPERDEW, BasisSet.SZ
    OPERDEW_TZP = Functional.OPERDEW, BasisSet.TZP
    PBE_D_DZP = Functional.PBE_D, BasisSet.DZP
    PBE_D_SZ = Functional.PBE_D, BasisSet.SZ
    PBE_D_TZP = Functional.PBE_D, BasisSet.TZP
    PBE_D3_BJ_DEF2_TZVP = Functional.PBE_D3_BJ, BasisSet.DEF2_TZVP
    PBE_DEF2_TZVP = Functional.PBE, BasisSet.DEF2_TZVP
    PBE_DZP = Functional.PBE, BasisSet.DZP
    PBE_SZ = Functional.PBE, BasisSet.SZ
    PBE_TZP = Functional.PBE, BasisSet.TZP
    PBE0_DZP = Functional.PBE0, BasisSet.DZP
    PBE0_DEF2_TZVP = Functional.PBE0, BasisSet.DEF2_TZVP
    PBE0_SZ = Functional.PBE0, BasisSet.SZ
    PBE0_TZP = Functional.PBE0, BasisSet.TZP
    PBE0_MBD_DEF2_TZVPP = Functional.PBE0_MBD, BasisSet.DEF2_TZVPPD
    PBESOL_DZP = Functional.PBESOL, BasisSet.DZP
    PBESOL_SZ = Functional.PBESOL, BasisSet.SZ
    PBESOL_TZP = Functional.PBESOL, BasisSet.TZP
    PKZB_DZP = Functional.PKZB, BasisSet.DZP
    PKZB_SZ = Functional.PKZB, BasisSet.SZ
    PKZB_TZP = Functional.PKZB, BasisSet.TZP
    PKZBX_KCISCOR_DZP = Functional.PKZBX_KCISCOR, BasisSet.DZP
    PKZBX_KCISCOR_SZ = Functional.PKZBX_KCISCOR, BasisSet.SZ
    PKZBX_KCISCOR_TZP = Functional.PKZBX_KCISCOR, BasisSet.TZP
    PM6 = Functional.PM6, BasisSet.NONE
    PW91_DZP = Functional.PW91, BasisSet.DZP
    PW91_SZ = Functional.PW91, BasisSet.SZ
    PW91_TZP = Functional.PW91, BasisSet.TZP
    REVPBE_D3_BJ_DEF2_TZVP = Functional.REVPBE_D3_BJ, BasisSet.DEF2_TZVP
    REVPBE_DZP = Functional.REVPBE, BasisSet.DZP
    REVPBE_SZ = Functional.REVPBE, BasisSet.SZ
    REVPBE_TZP = Functional.REVPBE, BasisSet.TZP
    REVTPSS_DZP = Functional.REVTPSS, BasisSet.DZP
    REVTPSS_SZ = Functional.REVTPSS, BasisSet.SZ
    REVTPSS_TZP = Functional.REVTPSS, BasisSet.TZP
    RGE2_DZP = Functional.RGE2, BasisSet.DZP
    RGE2_SZ = Functional.RGE2, BasisSet.SZ
    RGE2_TZP = Functional.RGE2, BasisSet.TZP
    RPBE_DZP = Functional.RPBE, BasisSet.DZP
    RPBE_SZ = Functional.RPBE, BasisSet.SZ
    RPBE_TZP = Functional.RPBE, BasisSet.TZP
    SSB_D_DZP = Functional.SSB_D, BasisSet.DZP
    SSB_D_SZ = Functional.SSB_D, BasisSet.SZ
    SSB_D_TZP = Functional.SSB_D, BasisSet.TZP
    SVWN_DEF2_TZVP = Functional.SVWN, BasisSet.DEF2_TZVP
    TMGGA_DZP = Functional.TMGGA, BasisSet.DZP
    TMGGA_SZ = Functional.TMGGA, BasisSet.SZ
    TMGGA_TZP = Functional.TMGGA, BasisSet.TZP
    TAU_HCTH_HYBRID_DZP = Functional.TAU_HCTH_HYBRID, BasisSet.DZP
    TAU_HCTH_HYBRID_SZ = Functional.TAU_HCTH_HYBRID, BasisSet.SZ
    TAU_HCTH_HYBRID_TZP = Functional.TAU_HCTH_HYBRID, BasisSet.TZP
    TAU_HCTH_DZP = Functional.TAU_HCTH, BasisSet.DZP
    TAU_HCTH_SZ = Functional.TAU_HCTH, BasisSet.SZ
    TAU_HCTH_TZP = Functional.TAU_HCTH, BasisSet.TZP
    TCSSD_T_CC_PVDZ = Functional.TCSSD_T, BasisSet.CC_PVDZ
    TPSSD_DZP = Functional.TPSSD, BasisSet.DZP
    TPSSD_SZ = Functional.TPSSD, BasisSet.SZ
    TPSSD_TZP = Functional.TPSSD, BasisSet.TZP
    TPSS_DZP = Functional.TPSS, BasisSet.DZP
    TPSS_SZ = Functional.TPSS, BasisSet.SZ
    TPSS_TZP = Functional.TPSS, BasisSet.TZP
    TPSSH_DEF2_TZVP = Functional.TPSSH, BasisSet.DEF2_TZVP
    TPSSH_DZP = Functional.TPSSH, BasisSet.DZP
    TPSSH_SZ = Functional.TPSSH, BasisSet.SZ
    TPSSH_TZP = Functional.TPSSH, BasisSet.TZP
    TTM2_1_F = Functional.TTM2_1_F, BasisSet.NONE
    VS98_X_XC_DZP = Functional.VS98_X_XC, BasisSet.DZP
    VS98_X_XC_SZ = Functional.VS98_X_XC, BasisSet.SZ
    VS98_X_XC_TZP = Functional.VS98_X_XC, BasisSet.TZP
    VS98_X_ONLY_DZP = Functional.VS98_X_ONLY, BasisSet.DZP
    VS98_X_ONLY_SZ = Functional.VS98_X_ONLY, BasisSet.SZ
    VS98_X_ONLY_TZP = Functional.VS98_X_ONLY, BasisSet.TZP
    VS98_DZP = Functional.VS98, BasisSet.DZP
    VS98_SZ = Functional.VS98, BasisSet.SZ
    VS98_TZP = Functional.VS98, BasisSet.TZP
    WB97M_D3BJ_DEF2_TZVPPD = Functional.WB97M_D3BJ, BasisSet.DEF2_TZVPPD
    WB97X_D_DEF2_SVP = Functional.WB97X_D, BasisSet.DEF2_SVP
    WB97X_D3_DEF2_TZVP = Functional.WB97X_D3, BasisSet.DEF2_TZVP
    WB97X_D3_CC_PVDZ = Functional.WB97X_D3, BasisSet.CC_PVDZ
    WB97X_6_31G_D = Functional.WB97X, BasisSet.GSTAR
    WB97X_CC_PVTZ = Functional.WB97X, BasisSet.CC_PVTZ
    X3LYP_VWN5_DZP = Functional.X3LYP_VWN5, BasisSet.DZP
    X3LYP_VWN5_SZ = Functional.X3LYP_VWN5, BasisSet.SZ
    X3LYP_VWN5_TZP = Functional.X3LYP_VWN5, BasisSet.TZP
    XLYP_DZP = Functional.XLYP, BasisSet.DZP
    XLYP_SZ = Functional.XLYP, BasisSet.SZ
    XLYP_TZP = Functional.XLYP, BasisSet.TZP
    NONE = Functional.NONE, BasisSet.NONE

    def _build_default_dict(self):
        e0_dict = {}
        for SYMBOL in ATOM_SYMBOLS:
            for CHARGE in range(-10, 11):
                e0_dict[(SYMBOL, CHARGE)] = array([0], dtype=float32)
        return e0_dict

    @property
    def atom_energies_dict(self):
        """Get the atomization energy dictionary"""
        key = str(self)
        try:
            # print(key)
            energies = atom_energy_collection.get(key, {})
            if len(energies) == 0:
                raise
        except:  # noqa
            logger.info(f"No available atomization energy for the QM method {key}. All values are set to 0.")
            energies = self._build_default_dict()
        return energies


class InteractionMethod(QmMethod):
    CCSD_T_NN = Functional.CCSDT, BasisSet.NN
    CCSD_T_CBS = Functional.CCSDT, BasisSet.CBS
    CCSD_T_CC_PVDZ = Functional.CCSDT, BasisSet.CC_PVDZ
    DCCSDT_HA_DZ = Functional.DCCSDT, BasisSet.HA_DZ
    DCCSDT_HA_TZ = Functional.DCCSDT, BasisSet.HA_TZ
    DLPNO_CCSDT = Functional.DLPNO_CCSDT, BasisSet.NONE
    DLPNO_CCSDT0 = (
        Functional.DLPNO_CCSDT0,
        BasisSet.NONE,
    )
    FN_DMC = Functional.FN_DMC, BasisSet.NONE
    FIXED = Functional.FIXED, BasisSet.NONE
    LNO_CCSDT = Functional.LNO_CCSDT, BasisSet.NONE
    MP2_CBS = Functional.MP2, BasisSet.CBS
    MP2_CC_PVDZ = Functional.MP2, BasisSet.CC_PVDZ
    MP2_CC_PVQZ = Functional.MP2, BasisSet.CC_PVQZ
    MP2_CC_PVTZ = Functional.MP2, BasisSet.CC_PVTZ
    MP2_5_CBS_ADZ = Functional.MP2_5, BasisSet.CBS_ADZ
    MP2C_CBS = Functional.MP2C, BasisSet.CBS
    QCISDT_CBS = Functional.QCISDT, BasisSet.CBS
    SAPT0_AUG_CC_PWCVXZ = Functional.SAPT0, BasisSet.AUG_CC_PWCVXZ
    SAPT0_JUN_CC_PVDZ = Functional.SAPT0, BasisSet.JUN_CC_PVDZ
    SAPT0_JUN_CC_PVDDZ = Functional.SAPT0, BasisSet.JUN_CC_PVDDZ
    SAPT0_AUG_CC_PVDDZ = Functional.SAPT0, BasisSet.AUG_CC_PVDDZ

    @property
    def atom_energies_dict(self):
        """Get an empty atomization energy dictionary because Interaction methods don't require this"""
        return {}


if __name__ == "__main__":
    for method in PotentialMethod:
        (str(method), len(method.atom_energies_dict))
