from loguru import logger
from enum import Enum, StrEnum
from openqdc.methods.atom_energies import atom_energy_collection, to_e_matrix


class QmType(StrEnum):
    FF  = "Empirical Force Field"
    SE  = "Semi Empirical"
    DFT = "Density Functional Theory"
    HF  = "Hartree Fork"
    CC  = "Couple Cluster"
    MP2 = "Moller Plesset"


class InterEnergyType(StrEnum): # InteractionEnergyType
    ES              = "electrostatic",
    EX              = "exchange",
    EX_S2           = "exchange S^2",
    IND             = "induction",
    TOTAL           = "total",
    EX_IND          = "exchange-induction",
    DISP            = "dispersion",
    EX_DISP_OS      = "exchange dispersion opposite-spin",
    EX_DISP_SS      = "exchange dispersion same-spin",
    DELTA_HF        = "Delta HF vs SAPT0"


class BasisSet(StrEnum):  
    NN              = 'nn'
    SZ              = 'sz'
    DZP             = 'dzp'
    TZP             = 'tzp'
    CBS             = 'cbs'
    HA_DZ           = 'haDZ'
    HA_TZ           = 'haTZ'
    CBS_ADZ         = 'cbs(adz)'
    GSTAR           = '6-31g*'
    CC_PVDZ         = 'cc-pvdz'
    CC_PVTZ         = 'cc-pvtz'
    CC_PVQZ         = 'cc-pvqz'
    DEF2_SVP        = 'def2-svp'
    DEF2_DZVP       = 'def2-dzvp'
    DEF2_TZVP       = 'def2-tzvp'
    DEF2_TZVPPD     = 'def2-tzvppd'  
    JUN_CC_PVDZ     = 'jun-cc-pvdz'     
    AUG_CC_PWCVXZ   = 'aug-cc-pwcvxz'
    JUN_CC_PVDDZ    = 'jun-cc-pV(D+d)Z'
    AUG_CC_PVDDZ    = 'aug-cc-pV(D+d)Z'
    NONE            = ''


class Functional(StrEnum):
    B1LYP_VWN5      = "b1lyp(vwn5)"
    B1PW91_VWN5     = "b1pw91(vwn5)"
    B3LYP           = "b3lyp"
    B3LYP_VWN5      = "b3lyp(vwn5)"
    B3LYP_S_VWN5    = "b3lyp*(vwn5)"
    B3LYPD          = "b3lyp-d"
    B3LYP_D3_BJ      = "b3lyp-d3(bj)"
    B97             = "b97"
    B97_1           = "b97-1"
    B97_2           = "b97-2"
    B97_D           = "b97-d"
    BECKE00         = "becke00"
    BECKE00_X_ONLY  = "becke00-x-only"
    BECKE00X_XC     = "becke00x(xc)"
    BECKE88X_BR89C  = "becke88x+br89c"
    BHANDH          = "bhandh"
    BHANDHLYP       = "bhandhlyp"
    BLAP3           = "blap3"
    BLYP            = "blyp"
    BLYPD           = "blyp-d"
    BMTAU1          = "bmtau1"
    BOP             = "bop"
    BP              = "bp"
    BP86_D          = "bp86-d"
    CCSD            = "ccsd"
    CCSDT           = "ccsd(t)"
    DCCSDT          = "dccsd(t)"
    DFT3B           = "dft3b"
    DLPNO_CCSDT     = "dlpno-ccsd(t)"
    DLPNO_CCSDT0    = "dlpno-ccsd(t0)"
    DSD_BLYP_D3_BJ  = "dsd-blyp-d3(bj)"
    FIXED           = "fixed" # TODO: remove after cleaning the L7 dataset 
    FN_DMC          = "fn-dmc"
    FT97            = "ft97"
    GFN1_XTB        = "gfn1_xtb"
    GFN2_XTB        = "gfn2_xtb"
    HCTH            = "hcth"
    HCTH_120        = "hcth-120"
    HCTH_147        = "hcth-147"
    HCTH_407        = "hcth-407"
    HCTH_93         = "hcth-93"
    HF              = "hf"
    KCIS_MODIFIED   = "kcis-modified"
    KCIS_ORIGINAL   = "kcis-original"
    KMLYP_VWN5      = "kmlyp(vwn5)"
    KT1             = "kt1"
    KT2             = "kt2"
    LDA_VWN         = "lda(vwn)"
    LNO_CCSDT       = "lno-ccsd(t)"
    M05             = "m05"
    M05_2X          = "m05-2x"
    M06             = "m06"
    M06_2X          = "m06-2x"
    M06_L           = "m06-l"
    MP2             = "MP2"
    MP2_5           = "MP2_5"
    MP2C            = "MP2C"
    MPBE            = "mpbe"
    MPBE0KCIS       = "mpbe0kcis"
    MPBE1KCIS       = "mpbe1kcis"
    MPBEKCIS        = "mpbekcis"
    MPW             = "mpw"
    MPW1K           = "mpw1k"
    MPW1PW          = "mpw1pw"
    MVS             = "mvs"
    MVSX            = "mvsx"
    O3LYP_VWN5      = "o3lyp(vwn5)"
    OLAP3           = "olap3"
    OLYP            = "olyp"
    OPBE            = "opbe"
    OPBE0           = "opbe0"
    OPERDEW         = "operdew"
    PBE             = "pbe"
    PBE_D           = "pbe-d"
    PBE_D3_BJ       = "pbe-d3(bj)"
    PBE0            = "pbe0"
    PBESOL          = "pbesol"
    PKZB            = "pkzb"
    PKZBX_KCISCOR   = "pkzbx-kciscor"
    PM6             = "pm6"
    PW91            = "pw91"
    QCISDT          = "qcisd(t)"
    REVPBE          = "revpbe"
    REVPBE_D3_BJ    = "revpbe-d3(bj)"
    REVTPSS         = "revtpss"
    RGE2            = "rge2"
    RPBE            = "rpbe"
    SAPT0           = "sapt0"
    SSB_D           = "ssb-d"
    SVWN            = "svwn"
    TMGGA           = "t-mgga"
    TAU_HCTH        = "tau-hcth"
    TAU_HCTH_HYBRID = "tau-hcth-hybrid"
    TPSS            = "tpss"
    TPSSD           = "tpss-d"
    TPSSH           = "tpssh"
    TTM2_1_F        = "ttm2.1-f"
    VS98            = "vs98"
    VS98_X_XC       = "vs98-x(xc)"
    VS98_X_ONLY     = "vs98-x-only"
    WB97M_D3BJ      = "wb97m-d3bj"
    WB97X           = "wb97x"
    WB97X_D         = "wb97x-d"
    WB97X_D3        = "wb97x-d3"
    X3LYP_VWN5      = "x3lyp(vwn5)"
    XLYP            = "xlyp"


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
        """ Get the atomization energy matrix"""
        energies = self.atom_energies_dict
        mat = to_e_matrix(energies)
        
        return mat
    
    @property
    def atom_energies_dict(self):
        """ Get the atomization energy dictionary"""
        raise NotImplementedError()


class PotentialMethod(QmMethod): #SPLIT FOR INTERACTIO ENERGIES AND FIX MD17
    B1LYP_VWN5_DZP                      = Functional.B1LYP_VWN5, BasisSet.DZP, 0
    B1LYP_VWN5_SZ                       = Functional.B1LYP_VWN5, BasisSet.SZ, 0
    B1LYP_VWN5_TZP                      = Functional.B1LYP_VWN5, BasisSet.TZP, 0
    B1PW91_VWN5_DZP                     = Functional.B1PW91_VWN5, BasisSet.DZP, 0
    B1PW91_VWN5_SZ                      = Functional.B1PW91_VWN5, BasisSet.SZ, 0
    B1PW91_VWN5_TZP                     = Functional.B1PW91_VWN5, BasisSet.TZP, 0
    B3LYP_VWN5_DZP                      = Functional.B3LYP_VWN5, BasisSet.DZP, 0
    B3LYP_VWN5_SZ                       = Functional.B3LYP_VWN5, BasisSet.SZ, 0
    B3LYP_VWN5_TZP                      = Functional.B3LYP_VWN5, BasisSet.TZP, 0
    B3LYP_S_VWN5_DZP                    = Functional.B3LYP_S_VWN5, BasisSet.DZP, 0
    B3LYP_S_VWN5_SZ                     = Functional.B3LYP_S_VWN5, BasisSet.SZ, 0
    B3LYP_S_VWN5_TZP                    = Functional.B3LYP_S_VWN5, BasisSet.TZP, 0
    B3LYP_D_DZP                         = Functional.B3LYPD, BasisSet.DZP, 0
    B3LYP_D_SZ                          = Functional.B3LYPD, BasisSet.SZ, 0
    B3LYP_D_TZP                         = Functional.B3LYPD, BasisSet.TZP, 0
    B3LYP_D3_BJ_DEF2_TZVP               = Functional.B3LYP_D3_BJ, BasisSet.DEF2_TZVP, 0
    B3LYP_6_31G_D                       = Functional.B3LYP, BasisSet.GSTAR, 0
    B3LYP_DEF2_TZVP                     = Functional.B3LYP, BasisSet.DEF2_TZVP, 0
    B97_1_DZP                           = Functional.B97_1, BasisSet.DZP, 0
    B97_1_SZ                            = Functional.B97_1, BasisSet.SZ, 0
    B97_1_TZP                           = Functional.B97_1, BasisSet.TZP, 0
    B97_2_DZP                           = Functional.B97_2, BasisSet.DZP, 0
    B97_2_SZ                            = Functional.B97_2, BasisSet.SZ, 0
    B97_2_TZP                           = Functional.B97_2, BasisSet.TZP, 0
    B97_D_DZP                           = Functional.B97_D, BasisSet.DZP, 0
    B97_D_SZ                            = Functional.B97_D, BasisSet.SZ, 0
    B97_D_TZP                           = Functional.B97_D, BasisSet.TZP, 0
    B97_DZP                             = Functional.B97, BasisSet.DZP, 0
    B97_SZ                              = Functional.B97, BasisSet.SZ, 0
    B97_TZP                             = Functional.B97, BasisSet.TZP, 0
    BECKE00_X_ONLY_DZP                  = Functional.BECKE00_X_ONLY, BasisSet.DZP, 0
    BECKE00_X_ONLY_SZ                   = Functional.BECKE00_X_ONLY, BasisSet.SZ, 0
    BECKE00_X_ONLY_TZP                  = Functional.BECKE00_X_ONLY, BasisSet.TZP, 0
    BECKE00_DZP                         = Functional.BECKE00, BasisSet.DZP, 0
    BECKE00_SZ                          = Functional.BECKE00, BasisSet.SZ, 0
    BECKE00_TZP                         = Functional.BECKE00, BasisSet.TZP, 0
    BECKE00X_XC_DZP                     = Functional.BECKE00X_XC, BasisSet.DZP, 0
    BECKE00X_XC_SZ                      = Functional.BECKE00X_XC, BasisSet.SZ, 0
    BECKE00X_XC_TZP                     = Functional.BECKE00X_XC, BasisSet.TZP, 0
    BECKE88X_BR89C_DZP                  = Functional.BECKE88X_BR89C, BasisSet.DZP, 0
    BECKE88X_BR89C_SZ                   = Functional.BECKE88X_BR89C, BasisSet.SZ, 0
    BECKE88X_BR89C_TZP                  = Functional.BECKE88X_BR89C, BasisSet.TZP, 0
    BHANDH_DZP                          = Functional.BHANDH, BasisSet.DZP, 0
    BHANDH_SZ                           = Functional.BHANDH, BasisSet.SZ, 0
    BHANDH_TZP                          = Functional.BHANDH, BasisSet.TZP, 0
    BHANDHLYP_DZP                       = Functional.BHANDHLYP, BasisSet.DZP, 0
    BHANDHLYP_SZ                        = Functional.BHANDHLYP, BasisSet.SZ, 0
    BHANDHLYP_TZP                       = Functional.BHANDHLYP, BasisSet.TZP, 0
    BLAP3_DZP                           = Functional.BLAP3, BasisSet.DZP, 0
    BLAP3_SZ                            = Functional.BLAP3, BasisSet.SZ, 0
    BLAP3_TZP                           = Functional.BLAP3, BasisSet.TZP, 0
    BLYP_D_DZP                          = Functional.BLYPD, BasisSet.DZP, 0
    BLYP_D_SZ                           = Functional.BLYPD, BasisSet.SZ, 0
    BLYP_D_TZP                          = Functional.BLYPD, BasisSet.TZP, 0
    BLYP_DZP                            = Functional.BLYP, BasisSet.DZP, 0
    BLYP_SZ                             = Functional.BLYP, BasisSet.SZ, 0
    BLYP_TZP                            = Functional.BLYP, BasisSet.TZP, 0
    BMTAU1_DZP                          = Functional.BMTAU1, BasisSet.DZP, 0
    BMTAU1_SZ                           = Functional.BMTAU1, BasisSet.SZ, 0
    BMTAU1_TZP                          = Functional.BMTAU1, BasisSet.TZP, 0
    BOP_DZP                             = Functional.BOP, BasisSet.DZP, 0
    BOP_SZ                              = Functional.BOP, BasisSet.SZ, 0
    BOP_TZP                             = Functional.BOP, BasisSet.TZP, 0
    BP_DZP                              = Functional.BP, BasisSet.DZP, 0
    BP_SZ                               = Functional.BP, BasisSet.SZ, 0
    BP_TZP                              = Functional.BP, BasisSet.TZP, 0
    BP86_D_DZP                          = Functional.BP86_D, BasisSet.DZP, 0
    BP86_D_SZ                           = Functional.BP86_D, BasisSet.SZ, 0
    BP86_D_TZP                          = Functional.BP86_D, BasisSet.TZP, 0
    CCSD_T_CC_PVDZ                      = Functional.CCSDT, BasisSet.CC_PVDZ, 0
    CCSD_CC_PVDZ                        = Functional.CCSD, BasisSet.CC_PVDZ, 0
    DFT3B                               = Functional.DFT3B, BasisSet.NONE, 0
    DSD_BLYP_D3_BJ_DEF2_TZVP            = Functional.DSD_BLYP_D3_BJ, BasisSet.DEF2_TZVP, 0     
    FT97_DZP                            = Functional.FT97, BasisSet.DZP, 0
    FT97_SZ                             = Functional.FT97, BasisSet.SZ, 0
    FT97_TZP                            = Functional.FT97, BasisSet.TZP, 0
    GFN1_XTB                            = Functional.GFN1_XTB, BasisSet.NONE, 0
    GFN2_XTB                            = Functional.GFN2_XTB, BasisSet.NONE, 0
    HCTH_120_DZP                        = Functional.HCTH_120, BasisSet.DZP, 0
    HCTH_120_SZ                         = Functional.HCTH_120, BasisSet.SZ, 0
    HCTH_120_TZP                        = Functional.HCTH_120, BasisSet.TZP, 0
    HCTH_147_DZP                        = Functional.HCTH_147, BasisSet.DZP, 0
    HCTH_147_SZ                         = Functional.HCTH_147, BasisSet.SZ, 0
    HCTH_147_TZP                        = Functional.HCTH_147, BasisSet.TZP, 0
    HCTH_407_DZP                        = Functional.HCTH_407, BasisSet.DZP, 0
    HCTH_407_SZ                         = Functional.HCTH_407, BasisSet.SZ, 0
    HCTH_407_TZP                        = Functional.HCTH_407, BasisSet.TZP, 0
    HCTH_93_DZP                         = Functional.HCTH_93, BasisSet.DZP, 0
    HCTH_93_SZ                          = Functional.HCTH_93, BasisSet.SZ, 0
    HCTH_93_TZP                         = Functional.HCTH_93, BasisSet.TZP, 0
    HF_DEF2_TZVP                        = Functional.HF, BasisSet.DEF2_TZVP, 0
    KCIS_MODIFIED_DZP                   = Functional.KCIS_MODIFIED, BasisSet.DZP, 0
    KCIS_MODIFIED_SZ                    = Functional.KCIS_MODIFIED, BasisSet.SZ, 0
    KCIS_MODIFIED_TZP                   = Functional.KCIS_MODIFIED, BasisSet.TZP, 0
    KCIS_ORIGINAL_DZP                   = Functional.KCIS_ORIGINAL, BasisSet.DZP, 0
    KCIS_ORIGINAL_SZ                    = Functional.KCIS_ORIGINAL, BasisSet.SZ, 0
    KCIS_ORIGINAL_TZP                   = Functional.KCIS_ORIGINAL, BasisSet.TZP, 0
    KMLYP_VWN5_DZP                      = Functional.KMLYP_VWN5, BasisSet.DZP, 0
    KMLYP_VWN5_SZ                       = Functional.KMLYP_VWN5, BasisSet.SZ, 0
    KMLYP_VWN5_TZP                      = Functional.KMLYP_VWN5, BasisSet.TZP, 0
    KT1_DZP                             = Functional.KT1, BasisSet.DZP, 0
    KT1_SZ                              = Functional.KT1, BasisSet.SZ, 0
    KT1_TZP                             = Functional.KT1, BasisSet.TZP, 0
    KT2_DZP                             = Functional.KT2, BasisSet.DZP, 0
    KT2_SZ                              = Functional.KT2, BasisSet.SZ, 0
    KT2_TZP                             = Functional.KT2, BasisSet.TZP, 0
    LDA_VWN_DZP                         = Functional.LDA_VWN, BasisSet.DZP, 0
    LDA_VWN_SZ                          = Functional.LDA_VWN, BasisSet.SZ, 0
    LDA_VWN_TZP                         = Functional.LDA_VWN, BasisSet.TZP, 0
    M05_2X_DZP                          = Functional.M05_2X, BasisSet.DZP, 0
    M05_2X_SZ                           = Functional.M05_2X, BasisSet.SZ, 0
    M05_2X_TZP                          = Functional.M05_2X, BasisSet.TZP, 0
    M05_DZP                             = Functional.M05, BasisSet.DZP, 0
    M05_SZ                              = Functional.M05, BasisSet.SZ, 0
    M05_TZP                             = Functional.M05, BasisSet.TZP, 0
    M06_2X_DZP                          = Functional.M06_2X, BasisSet.DZP, 0
    M06_2X_SZ                           = Functional.M06_2X, BasisSet.SZ, 0
    M06_2X_TZP                          = Functional.M06_2X, BasisSet.TZP, 0
    M06_L_DZP                           = Functional.M06_L, BasisSet.DZP, 0
    M06_L_SZ                            = Functional.M06_L, BasisSet.SZ, 0
    M06_L_TZP                           = Functional.M06_L, BasisSet.TZP, 0
    M06_DZP                             = Functional.M06, BasisSet.DZP, 0
    M06_SZ                              = Functional.M06, BasisSet.SZ, 0
    M06_TZP                             = Functional.M06, BasisSet.TZP, 0
    MPBE_DZP                            = Functional.MPBE, BasisSet.DZP, 0
    MPBE_SZ                             = Functional.MPBE, BasisSet.SZ, 0
    MPBE_TZP                            = Functional.MPBE, BasisSet.TZP, 0
    MPBE0KCIS_DZP                       = Functional.MPBE0KCIS, BasisSet.DZP, 0
    MPBE0KCIS_SZ                        = Functional.MPBE0KCIS, BasisSet.SZ, 0
    MPBE0KCIS_TZP                       = Functional.MPBE0KCIS, BasisSet.TZP, 0
    MPBE1KCIS_DZP                       = Functional.MPBE1KCIS, BasisSet.DZP, 0
    MPBE1KCIS_SZ                        = Functional.MPBE1KCIS, BasisSet.SZ, 0
    MPBE1KCIS_TZP                       = Functional.MPBE1KCIS, BasisSet.TZP, 0
    MPBEKCIS_DZP                        = Functional.MPBEKCIS, BasisSet.DZP, 0
    MPBEKCIS_SZ                         = Functional.MPBEKCIS, BasisSet.SZ, 0
    MPBEKCIS_TZP                        = Functional.MPBEKCIS, BasisSet.TZP, 0
    MPW_DZP                             = Functional.MPW, BasisSet.DZP, 0
    MPW_SZ                              = Functional.MPW, BasisSet.SZ, 0
    MPW_TZP                             = Functional.MPW, BasisSet.TZP, 0
    MPW1K_DZP                           = Functional.MPW1K, BasisSet.DZP, 0
    MPW1K_SZ                            = Functional.MPW1K, BasisSet.SZ, 0
    MPW1K_TZP                           = Functional.MPW1K, BasisSet.TZP, 0
    MPW1PW_DZP                          = Functional.MPW1PW, BasisSet.DZP, 0
    MPW1PW_SZ                           = Functional.MPW1PW, BasisSet.SZ, 0
    MPW1PW_TZP                          = Functional.MPW1PW, BasisSet.TZP, 0
    MVS_DZP                             = Functional.MVS, BasisSet.DZP, 0
    MVS_SZ                              = Functional.MVS, BasisSet.SZ, 0
    MVS_TZP                             = Functional.MVS, BasisSet.TZP, 0
    MVSX_DZP                            = Functional.MVSX, BasisSet.DZP, 0
    MVSX_SZ                             = Functional.MVSX, BasisSet.SZ, 0
    MVSX_TZP                            = Functional.MVSX, BasisSet.TZP, 0
    O3LYP_VWN5_DZP                      = Functional.O3LYP_VWN5, BasisSet.DZP, 0
    O3LYP_VWN5_SZ                       = Functional.O3LYP_VWN5, BasisSet.SZ, 0
    O3LYP_VWN5_TZP                      = Functional.O3LYP_VWN5, BasisSet.TZP, 0
    OLAP3_DZP                           = Functional.OLAP3, BasisSet.DZP, 0
    OLAP3_SZ                            = Functional.OLAP3, BasisSet.SZ, 0
    OLAP3_TZP                           = Functional.OLAP3, BasisSet.TZP, 0
    OLYP_DZP                            = Functional.OLYP, BasisSet.DZP, 0
    OLYP_SZ                             = Functional.OLYP, BasisSet.SZ, 0
    OLYP_TZP                            = Functional.OLYP, BasisSet.TZP, 0
    OPBE_DZP                            = Functional.OPBE, BasisSet.DZP, 0
    OPBE_SZ                             = Functional.OPBE, BasisSet.SZ, 0
    OPBE_TZP                            = Functional.OPBE, BasisSet.TZP, 0
    OPBE0_DZP                           = Functional.OPBE0, BasisSet.DZP, 0
    OPBE0_SZ                            = Functional.OPBE0, BasisSet.SZ, 0
    OPBE0_TZP                           = Functional.OPBE0, BasisSet.TZP, 0
    OPERDEW_DZP                         = Functional.OPERDEW, BasisSet.DZP, 0
    OPERDEW_SZ                          = Functional.OPERDEW, BasisSet.SZ, 0
    OPERDEW_TZP                         = Functional.OPERDEW, BasisSet.TZP, 0
    PBE_D_DZP                           = Functional.PBE_D, BasisSet.DZP, 0
    PBE_D_SZ                            = Functional.PBE_D, BasisSet.SZ, 0
    PBE_D_TZP                           = Functional.PBE_D, BasisSet.TZP, 0
    PBE_D3_BJ_DEF2_TZVP                 = Functional.PBE_D3_BJ, BasisSet.DEF2_TZVP, 0
    PBE_DEF2_TZVP                       = Functional.PBE, BasisSet.DEF2_TZVP, 0
    PBE_DZP                             = Functional.PBE, BasisSet.DZP, 0
    PBE_SZ                              = Functional.PBE, BasisSet.SZ, 0
    PBE_TZP                             = Functional.PBE, BasisSet.TZP, 0
    PBE0_DZP                            = Functional.PBE0, BasisSet.DZP, 0
    PBE0_DEF2_TZVP                      = Functional.PBE0, BasisSet.DEF2_TZVP, 0
    PBE0_SZ                             = Functional.PBE0, BasisSet.SZ, 0
    PBE0_TZP                            = Functional.PBE0, BasisSet.TZP, 0
    PBESOL_DZP                          = Functional.PBESOL, BasisSet.DZP, 0
    PBESOL_SZ                           = Functional.PBESOL, BasisSet.SZ, 0
    PBESOL_TZP                          = Functional.PBESOL, BasisSet.TZP, 0
    PKZB_DZP                            = Functional.PKZB, BasisSet.DZP, 0
    PKZB_SZ                             = Functional.PKZB, BasisSet.SZ, 0
    PKZB_TZP                            = Functional.PKZB, BasisSet.TZP, 0
    PKZBX_KCISCOR_DZP                   = Functional.PKZBX_KCISCOR, BasisSet.DZP, 0
    PKZBX_KCISCOR_SZ                    = Functional.PKZBX_KCISCOR, BasisSet.SZ, 0
    PKZBX_KCISCOR_TZP                   = Functional.PKZBX_KCISCOR, BasisSet.TZP, 0
    PM6                                 = Functional.PM6, BasisSet.NONE, 0
    PW91_DZP                            = Functional.PW91, BasisSet.DZP, 0
    PW91_SZ                             = Functional.PW91, BasisSet.SZ, 0
    PW91_TZP                            = Functional.PW91, BasisSet.TZP, 0
    REVPBE_D3_BJ_DEF2_TZVP              = Functional.REVPBE_D3_BJ, BasisSet.DEF2_TZVP, 0
    REVPBE_DZP                          = Functional.REVPBE, BasisSet.DZP, 0
    REVPBE_SZ                           = Functional.REVPBE, BasisSet.SZ, 0
    REVPBE_TZP                          = Functional.REVPBE, BasisSet.TZP, 0
    REVTPSS_DZP                         = Functional.REVTPSS, BasisSet.DZP, 0
    REVTPSS_SZ                          = Functional.REVTPSS, BasisSet.SZ, 0
    REVTPSS_TZP                         = Functional.REVTPSS, BasisSet.TZP, 0
    RGE2_DZP                            = Functional.RGE2, BasisSet.DZP, 0
    RGE2_SZ                             = Functional.RGE2, BasisSet.SZ, 0
    RGE2_TZP                            = Functional.RGE2, BasisSet.TZP, 0
    RPBE_DZP                            = Functional.RPBE, BasisSet.DZP, 0
    RPBE_SZ                             = Functional.RPBE, BasisSet.SZ, 0
    RPBE_TZP                            = Functional.RPBE, BasisSet.TZP, 0
    SSB_D_DZP                           = Functional.SSB_D, BasisSet.DZP, 0
    SSB_D_SZ                            = Functional.SSB_D, BasisSet.SZ, 0
    SSB_D_TZP                           = Functional.SSB_D, BasisSet.TZP, 0
    SVWN_DEF2_TZVP                      = Functional.SVWN, BasisSet.DEF2_TZVP, 0
    TMGGA_DZP                           = Functional.TMGGA, BasisSet.DZP, 0
    TMGGA_SZ                            = Functional.TMGGA, BasisSet.SZ, 0
    TMGGA_TZP                           = Functional.TMGGA, BasisSet.TZP, 0
    TAU_HCTH_HYBRID_DZP                 = Functional.TAU_HCTH_HYBRID, BasisSet.DZP, 0
    TAU_HCTH_HYBRID_SZ                  = Functional.TAU_HCTH_HYBRID, BasisSet.SZ, 0
    TAU_HCTH_HYBRID_TZP                 = Functional.TAU_HCTH_HYBRID, BasisSet.TZP, 0
    TAU_HCTH_DZP                        = Functional.TAU_HCTH, BasisSet.DZP, 0
    TAU_HCTH_SZ                         = Functional.TAU_HCTH, BasisSet.SZ, 0
    TAU_HCTH_TZP                        = Functional.TAU_HCTH, BasisSet.TZP, 0
    TPSSD_DZP                           = Functional.TPSSD, BasisSet.DZP, 0
    TPSSD_SZ                            = Functional.TPSSD, BasisSet.SZ, 0
    TPSSD_TZP                           = Functional.TPSSD, BasisSet.TZP, 0
    TPSS_DZP                            = Functional.TPSS, BasisSet.DZP, 0
    TPSS_SZ                             = Functional.TPSS, BasisSet.SZ, 0
    TPSS_TZP                            = Functional.TPSS, BasisSet.TZP, 0
    TPSSH_DEF2_TZVP                     = Functional.TPSSH, BasisSet.DEF2_TZVP, 0
    TPSSH_DZP                           = Functional.TPSSH, BasisSet.DZP, 0
    TPSSH_SZ                            = Functional.TPSSH, BasisSet.SZ, 0
    TPSSH_TZP                           = Functional.TPSSH, BasisSet.TZP, 0
    TTM2_1_F                            = Functional.TTM2_1_F, BasisSet.NONE, 0
    VS98_X_XC_DZP                       = Functional.VS98_X_XC, BasisSet.DZP, 0
    VS98_X_XC_SZ                        = Functional.VS98_X_XC, BasisSet.SZ, 0
    VS98_X_XC_TZP                       = Functional.VS98_X_XC, BasisSet.TZP, 0
    VS98_X_ONLY_DZP                     = Functional.VS98_X_ONLY, BasisSet.DZP, 0
    VS98_X_ONLY_SZ                      = Functional.VS98_X_ONLY, BasisSet.SZ, 0
    VS98_X_ONLY_TZP                     = Functional.VS98_X_ONLY, BasisSet.TZP, 0
    VS98_DZP                            = Functional.VS98, BasisSet.DZP, 0
    VS98_SZ                             = Functional.VS98, BasisSet.SZ, 0
    VS98_TZP                            = Functional.VS98, BasisSet.TZP, 0
    WB97M_D3BJ_DEF2_TZVPPD              = Functional.WB97M_D3BJ, BasisSet.DEF2_TZVPPD, 0
    WB97X_D_DEF2_SVP                    = Functional.WB97X_D, BasisSet.DEF2_SVP, 0
    WB97X_D3_DEF2_TZVP                  = Functional.WB97X_D3, BasisSet.DEF2_TZVP, 0
    WB97X_6_31G_D                       = Functional.WB97X, BasisSet.GSTAR, 0
    X3LYP_VWN5_DZP                      = Functional.X3LYP_VWN5, BasisSet.DZP, 0
    X3LYP_VWN5_SZ                       = Functional.X3LYP_VWN5, BasisSet.SZ, 0
    X3LYP_VWN5_TZP                      = Functional.X3LYP_VWN5, BasisSet.TZP, 0
    XLYP_DZP                            = Functional.XLYP, BasisSet.DZP, 0
    XLYP_SZ                             = Functional.XLYP, BasisSet.SZ, 0
    XLYP_TZP                            = Functional.XLYP, BasisSet.TZP, 0

    @property
    def atom_energies_dict(self):
        """ Get the atomization energy dictionary"""
        key = str(self)
        try:
            # print(key)
            energies = atom_energy_collection.get(key, {})
            if len(energies) == 0: raise
        except:
            logger.info(f"No available atomization energy for the QM method {key}. All values are set to 0.")
        
        return energies
    

class InteractionMethod(QmMethod):
    CCSD_T_NN                           = Functional.CCSDT, BasisSet.NN, 0
    CCSD_T_CBS                          = Functional.CCSDT, BasisSet.CBS, 0
    CCSD_T_CC_PVDZ                      = Functional.CCSDT, BasisSet.CC_PVDZ, 0
    DCCSDT_HA_DZ                        = Functional.DCCSDT, BasisSet.HA_DZ, 0
    DCCSDT_HA_TZ                        = Functional.DCCSDT, BasisSet.HA_TZ, 0
    DLPNO_CCSDT                         = Functional.DLPNO_CCSDT, BasisSet.NONE, 0
    DLPNO_CCSDT0                        = Functional.DLPNO_CCSDT0, BasisSet.NONE, 
    FN_DMC                              = Functional.FN_DMC, BasisSet.NONE, 0  
    FIXED                               = Functional.FIXED, BasisSet.NONE, 0   
    LNO_CCSDT                           = Functional.LNO_CCSDT, BasisSet.NONE, 0
    MP2_CBS                             = Functional.MP2, BasisSet.CBS, 0
    MP2_CC_PVDZ                         = Functional.MP2, BasisSet.CC_PVDZ, 0
    MP2_CC_PVQZ                         = Functional.MP2, BasisSet.CC_PVQZ, 0
    MP2_CC_PVTZ                         = Functional.MP2, BasisSet.CC_PVTZ, 0
    MP2_5_CBS_ADZ                       = Functional.MP2_5, BasisSet.CBS_ADZ, 0
    MP2C_CBS                            = Functional.MP2C, BasisSet.CBS, 0
    QCISDT_CBS                          = Functional.QCISDT, BasisSet.CBS, 0
    SAPT0_AUG_CC_PWCVXZ                 = Functional.SAPT0, BasisSet.AUG_CC_PWCVXZ, 0
    SAPT0_JUN_CC_PVDZ                   = Functional.SAPT0, BasisSet.JUN_CC_PVDZ, 0
    SAPT0_JUN_CC_PVDDZ                  = Functional.SAPT0, BasisSet.JUN_CC_PVDDZ, 0
    SAPT0_AUG_CC_PVDDZ                  = Functional.SAPT0, BasisSet.AUG_CC_PVDDZ, 0

    @property
    def atom_energies_dict(self):
        """ Get an empty atomization energy dictionary because Interaction methods don't require this"""
        return {}

if __name__ ==  "__main__":
    for method in PotentialMethod:
        (str(method), len(method.atom_energies_dict))