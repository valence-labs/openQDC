class DataConfigFactory:
    ani = dict(
        dataset_name="ani",
        links={
            "ani1.hdf5.gz": "https://zenodo.org/record/3585840/files/214.hdf5.gz",
            "ani1x.hdf5.gz": "https://zenodo.org/record/4081694/files/292.hdf5.gz",
            "ani1ccx.hdf5.gz": "https://zenodo.org/record/4081692/files/293.hdf5.gz",
        },
    )

    comp6 = dict(
        dataset_name="comp6",
        links={
            "gdb7_9.hdf5.gz": "https://zenodo.org/record/3588361/files/208.hdf5.gz",
            "gdb10_13.hdf5.gz": "https://zenodo.org/record/3588364/files/209.hdf5.gz",
            "drugbank.hdf5.gz": "https://zenodo.org/record/3588361/files/207.hdf5.gz",
            "tripeptides.hdf5.gz": "https://zenodo.org/record/3588368/files/211.hdf5.gz",
            "ani_md.hdf5.gz": "https://zenodo.org/record/3588341/files/205.hdf5.gz",
            "s66x8.hdf5.gz": "https://zenodo.org/record/3588367/files/210.hdf5.gz",
        },
    )

    gdml = dict(
        dataset_name="gdml",
        links={"gdml.hdf5.gz": "https://zenodo.org/record/3585908/files/219.hdf5.gz"},
    )

    solvated_peptides = dict(
        dataset_name="solvated_peptides",
        links={"solvated_peptides.hdf5.gz": "https://zenodo.org/record/3585804/files/213.hdf5.gz"},
    )

    iso_17 = dict(
        dataset_name="iso_17",
        links={"iso_17.hdf5.gz": "https://zenodo.org/record/3585907/files/216.hdf5.gz"},
    )

    sn2_rxn = dict(
        dataset_name="sn2_rxn",
        links={"sn2_rxn.hdf5.gz": "https://zenodo.org/record/3585800/files/212.hdf5.gz"},
    )

    # FROM: https://sites.uw.edu/wdbase/database-of-water-clusters/
    waterclusters3_30 = dict(
        dataset_name="waterclusters3_30",
        links={"W3-W30_all_geoms_TTM2.1-F.zip": "https://drive.google.com/uc?id=18Y7OiZXSCTsHrQ83GCc4fyE_abbL6E_n"},
    )

    geom = dict(
        dataset_name="geom",
        links={"rdkit_folder.tar.gz": "https://dataverse.harvard.edu/api/access/datafile/4327252"},
    )

    molecule3d = dict(
        dataset_name="molecule3d",
        links={"molecule3d.zip": "https://drive.google.com/uc?id=1C_KRf8mX-gxny7kL9ACNCEV4ceu_fUGy"},
    )

    orbnet_denali = dict(
        dataset_name="orbnet_denali",
        links={
            "orbnet_denali.tar.gz": "https://figshare.com/ndownloader/files/28672287",
            "orbnet_denali_targets.tar.gz": "https://figshare.com/ndownloader/files/28672248",
        },
    )

    qm7x = dict(
        dataset_name="qm7x",  # https://zenodo.org/record/4288677/files/1000.xz?download=1
        links={f"{i}000.xz": f"https://zenodo.org/record/4288677/files/{i}000.xz" for i in range(1, 9)},
    )

    qmugs = dict(
        dataset_name="qmugs",
        links={
            "summary.csv": "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv",
            "structures.tar.gz": "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz",
        },
    )

    spice = dict(
        dataset_name="spice",
        links={"SPICE-1.1.4.hdf5": "https://zenodo.org/record/8222043/files/SPICE-1.1.4.hdf5"},
    )

    dess = dict(
        dataset_name="dess5m",
        links={
            "DESS5M.zip": "https://zenodo.org/record/5706002/files/DESS5M.zip",
            "DESS370.zip": "https://zenodo.org/record/5676266/files/DES370K.zip",
        },
    )

    tmqm = dict(
        dataset_name="tmqm",
        links={
            x: f"https://raw.githubusercontent.com/bbskjelstad/tmqm/master/data/{x}"
            for x in ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_y.csv", "Benchmark2_TPSSh_Opt.xyz"]
        },
    )

    misato = dict(
        dataset_name="misato",
        links={
            "MD.hdf5": "https://zenodo.org/record/7711953/files/MD.hdf5",
            "QM.hdf5": "https://zenodo.org/record/7711953/files/QM.hdf5",
        },
    )

    nabladft = dict(
        dataset_name="nabladft",
        links={"nabladft.db": "https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/moses_db/dataset_full.db"},
        cmd=[
            "axel -n 10 --output=dataset_full.db https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/moses_db/dataset_full.db"
        ],
    )

    pubchemqc = dict(
        dataset_name="pubchemqc",
        links={
            "pqcm_b3lyp_2017.tar.gz": "https://chibakoudai.sharepoint.com/:u:/s/stair02/Ed9Z16k0ctJKk9nQLMYFHYUBp_E9zerPApRaWTrOIYN-Eg"
        },
        cmd=[
            'wget "https://chibakoudai.sharepoint.com/:u:/s/stair06/EcWMtOpIEqFLrHcR1dzlZiMBLhTFY0RZ0qPaqC4lhRp51A?download=1" -O b3lyp_pm6_ver1.0.1-postgrest-docker-compose.tar.xz.rclone_chunk.001',
            'wget "https://chibakoudai.sharepoint.com/:u:/s/stair06/EbJe-SlL4oNPhOpOtA8mxLsB1F3eI2l-5RS315hIZUFNwQ?download=1" -O b3lyp_pm6_ver1.0.1-postgrest-docker-compose.tar.xz.rclone_chunk.002',
            "cat b3lyp_pm6_ver1.0.1-postgrest-docker-compose.tar.xz.rclone_chunk.001 b3lyp_pm6_ver1.0.1-postgrest-docker-compose.tar.xz.rclone_chunk.002 | tar xvfJ - ",
        ],
    )

    available_datasets = [k for k in locals().keys() if not k.startswith("__")]

    def __init__(self):
        pass

    def __call__(self, dataset_name):
        return getattr(self, dataset_name)
