import gzip
import os
import shutil
import socket
import tarfile
import urllib.error
import urllib.request
import zipfile

import fsspec
import gdown
import requests
import tqdm
from loguru import logger
from sklearn.utils import Bunch

from openqdc.utils.io import get_local_cache


def download_url(url, local_filename):
    """
    Download a file from a url to a local file.
    Parameters
    ----------
        url : str
            URL to download from.
        local_filename : str
            Local path for destination.
    """
    logger.info(f"Url: {url}  File: {local_filename}")
    if "drive.google.com" in url:
        gdown.download(url, local_filename, quiet=False)
    elif "raw.github" in url:
        r = requests.get(url, allow_redirects=True)
        with open(local_filename, "wb") as f:
            f.write(r.content)
    else:
        r = requests.get(url, stream=True)
        with fsspec.open(local_filename, "wb") as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=16384)):
                if chunk:
                    f.write(chunk)


def decompress_tar_gz(local_filename):
    """
    Decompress a tar.gz file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    parent = os.path.dirname(local_filename)
    with tarfile.open(local_filename) as tar:
        logger.info(f"Verifying archive extraction states: {local_filename}")
        all_names = tar.getnames()
        all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
        if not all_extracted:
            logger.info(f"Extracting archive: {local_filename}")
            tar.extractall(path=parent)
        else:
            logger.info(f"Archive already extracted: {local_filename}")


def decompress_zip(local_filename):
    """
    Decompress a zip file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    parent = os.path.dirname(local_filename)

    logger.info(f"Verifying archive extraction states: {local_filename}")
    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        all_names = zip_ref.namelist()
        all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
        if not all_extracted:
            logger.info(f"Extracting archive: {local_filename}")
            zip_ref.extractall(parent)
        else:
            logger.info(f"Archive already extracted: {local_filename}")


def decompress_gz(local_filename):
    """
    Decompress a gz file.
    Parameters
    ----------
        local_filename : str
            Path to local file to decompress.
    """
    logger.info(f"Verifying archive extraction states: {local_filename}")
    out_filename = local_filename.replace(".gz", "")
    if out_filename.endswith("hdf5"):
        out_filename = local_filename.replace("hdf5", "h5")

    all_extracted = os.path.exists(out_filename)
    if not all_extracted:
        logger.info(f"Extracting archive: {local_filename}")
        with gzip.open(local_filename, "rb") as f_in, open(out_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        logger.info(f"Archive already extracted: {local_filename}")


def fetch_file(url, local_filename, overwrite=False):
    """
    Download a file from a url to a local file. Useful for big files.
    Parameters
    ----------
    url : str
        URL to download from.
    local_filename : str
        Local file to save to.
    overwrite : bool
        Whether to overwrite existing files.
    Returns
    -------
    local_filename : str
        Local file.
    """
    try:
        if os.path.exists(local_filename) and not overwrite:
            logger.info("File already exists, skipping download")
        else:
            download_url(url, local_filename)

        # decompress archive if necessary
        parent = os.path.dirname(local_filename)
        if local_filename.endswith("tar.gz"):
            decompress_tar_gz(local_filename)

        elif local_filename.endswith("zip"):
            decompress_zip(local_filename)

        elif local_filename.endswith(".gz"):
            decompress_gz(local_filename)

        elif local_filename.endswith("xz"):
            logger.info(f"Extracting archive: {local_filename}")
            os.system(f"cd {parent} && xz -d *.xz")

        else:
            pass

    except (socket.gaierror, urllib.error.URLError) as err:
        raise ConnectionError("Could not download {} due to {}".format(url, err))

    return local_filename


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
        links={"sn2_rxn.hdf5.gz": "https://zenodo.org/records/2605341/files/sn2_reactions.npz"},
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

    l7 = dict(
        dataset_name="l7",
        links={
            "l7.yaml": "http://cuby4.molecular.cz/download_datasets/l7.yaml",
            "geometries.tar.gz": "http://cuby4.molecular.cz/download_geometries/L7.tar",
        },
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
        dataset_name="qm7x",
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
    spicev2 = dict(
        dataset_name="spicev2",
        links={"spice-2.0.0.hdf5": "https://zenodo.org/records/10835749/files/SPICE-2.0.0.hdf5?download=1"},
    )

    splinter = dict(
        dataset_name="splinter",
        links={
            "dimerpairs.0.tar.gz": "https://figshare.com/ndownloader/files/39449167",
            "dimerpairs.1.tar.gz": "https://figshare.com/ndownloader/files/40271983",
            "dimerpairs.2.tar.gz": "https://figshare.com/ndownloader/files/40271989",
            "dimerpairs.3.tar.gz": "https://figshare.com/ndownloader/files/40272001",
            "dimerpairs.4.tar.gz": "https://figshare.com/ndownloader/files/40272022",
            "dimerpairs.5.tar.gz": "https://figshare.com/ndownloader/files/40552931",
            "dimerpairs.6.tar.gz": "https://figshare.com/ndownloader/files/40272040",
            "dimerpairs.7.tar.gz": "https://figshare.com/ndownloader/files/40272052",
            "dimerpairs.8.tar.gz": "https://figshare.com/ndownloader/files/40272061",
            "dimerpairs.9.tar.gz": "https://figshare.com/ndownloader/files/40272064",
            "dimerpairs_nonstandard.tar.gz": "https://figshare.com/ndownloader/files/40272067",
            "lig_interaction_sites.sdf": "https://figshare.com/ndownloader/files/40272070",
            "lig_monomers.sdf": "https://figshare.com/ndownloader/files/40272073",
            "prot_interaction_sites.sdf": "https://figshare.com/ndownloader/files/40272076",
            "prot_monomers.sdf": "https://figshare.com/ndownloader/files/40272079",
            "merge_monomers.py": "https://figshare.com/ndownloader/files/41807682",
        },
    )

    dess = dict(
        dataset_name="dess5m",
        links={
            "DESS5M.zip": "https://zenodo.org/record/5706002/files/DESS5M.zip",
            "DESS370.zip": "https://zenodo.org/record/5676266/files/DES370K.zip",
        },
    )

    des370k_interaction = dict(
        dataset_name="des370k_interaction",
        links={
            "DES370K.zip": "https://zenodo.org/record/5676266/files/DES370K.zip",
        },
    )

    des5m_interaction = dict(
        dataset_name="des5m_interaction",
        links={
            "DES5M.zip": "https://zenodo.org/records/5706002/files/DESS5M.zip?download=1",
        },
    )

    tmqm = dict(
        dataset_name="tmqm",
        links={
            x: f"https://raw.githubusercontent.com/bbskjelstad/tmqm/master/data/{x}"
            for x in ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_y.csv", "Benchmark2_TPSSh_Opt.xyz"]
        },
    )

    metcalf = dict(
        dataset_name="metcalf",
        links={"model-data.tar.gz": "https://zenodo.org/records/10934211/files/model-data.tar?download=1"},
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

    multixcqm9 = dict(
        dataset_name="multixcqm9",
        links={
            "xyz.zip": "https://data.dtu.dk/ndownloader/files/35143624",
            "xtb.zip": "https://data.dtu.dk/ndownloader/files/42444300",
            "dzp.zip": "https://data.dtu.dk/ndownloader/files/42443925",
            "tzp.zip": "https://data.dtu.dk/ndownloader/files/42444129",
            "sz.zip": "https://data.dtu.dk/ndownloader/files/42441345",
            "failed_indices.dat": "https://data.dtu.dk/ndownloader/files/37337677",
        },
    )

    transition1x = dict(
        dataset_name="transition1x",
        links={"Transition1x.h5": "https://figshare.com/ndownloader/files/36035789"},
    )

    des_s66 = dict(
        dataset_name="des_s66",
        links={"DESS66.zip": "https://zenodo.org/records/5676284/files/DESS66.zip?download=1"},
    )

    des_s66x8 = dict(
        dataset_name="des_s66x8",
        links={"DESS66x8.zip": "https://zenodo.org/records/5676284/files/DESS66x8.zip?download=1"},
    )
    revmd17 = dict(
        dataset_name="revmd17",
        links={"revmd17.zip": "https://figshare.com/ndownloader/articles/12672038/versions/3"},
    )

    x40 = dict(
        dataset_name="x40",
        links={
            "x40.yaml": "http://cuby4.molecular.cz/download_datasets/x40.yaml",
            "geometries.tar.gz": "http://cuby4.molecular.cz/download_geometries/X40.tar",
        },
    )

    available_datasets = [k for k in locals().keys() if not k.startswith("__")]

    def __init__(self):
        pass

    def __call__(self, dataset_name):
        return getattr(self, dataset_name)


class DataDownloader:
    """Download data from a remote source.
    Parameters
    ----------
    cache_path : str
        Path to the cache directory.
    overwrite : bool
        Whether to overwrite existing files.
    """

    def __init__(self, cache_path=None, overwrite=False):
        if cache_path is None:
            cache_path = get_local_cache()

        self.cache_path = cache_path
        self.overwrite = overwrite

    def from_config(self, config: dict):
        b_config = Bunch(**config)
        data_path = os.path.join(self.cache_path, b_config.dataset_name)
        os.makedirs(data_path, exist_ok=True)

        logger.info(f"Downloading the {b_config.dataset_name} dataset")
        for local, link in b_config.links.items():
            outfile = os.path.join(data_path, local)

            fetch_file(link, outfile)

    def from_name(self, name):
        cfg = DataConfigFactory()(name)
        return self.from_config(cfg)
