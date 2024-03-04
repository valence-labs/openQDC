"""Download funtionalities for PubChemQC."""

import hashlib
import os
import pickle as pkl
from glob import glob

import click
import numpy as np
from tqdm import tqdm

from openqdc.datasets.pcqm import read_archive
from openqdc.utils.io import get_local_cache


@click.group()
def cli():
    pass


def download_b3lyp_pm6_item(i, dirname, web_cookie, method="b3lyp"):
    try:
        step_size = 25000
        start = str(i * step_size + 1).rjust(9, "0")
        stop = str((i + 1) * step_size).rjust(9, "0")

        enc_ext = "%2Esha256sum"
        ext = ".sha256sum"

        cmd_b3lyp = (
            lambda check: f"""wget --header="Host: chibakoudai.sharepoint.com"
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36" --header="Accept:
text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/
webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
--header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7"
--header="Referer: https://chibakoudai.sharepoint.com/sites/stair02/
Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fstair02%2F
Shared%20Documents%2Fdata%2FPubChemQC%2FB3LYP%5FPM6%2Fb3lyp%5Fpm6%5F
ver1%2E0%2E0%2Fjson%2Fall%2FCompound%5F{start}%5F{stop}%2Etar%2E
xz{enc_ext if check else ''}
&viewid=f6d34767%2D64f0%2D480e%2Dab70%2Dd8524dbdc74e&parent=%2Fsites
%2Fstair02%2FShared%20Documents%2Fdata%2FPubChemQC%2FB3LYP%5FPM6%2Fb3lyp
%5Fpm6%5Fver1%2E0%2E0%2Fjson%2Fall" --header="Cookie: MicrosoftApplic
ationsTelemetryDeviceId=cec40b8a-9870-4c4f-bb71-838a300c8685; MSFPC=GUID
=511089efdbeb49d3923fdc7e6404bd9b&HASH=5110&LV=202303&V=4&LU=1678287635332;
WSS_FullScreenMode=false; {web_cookie}
--header="Connection: keep-alive" "https://chibakoudai.sharep
oint.com/sites/stair02/_layouts/15/download.aspx?SourceUrl=%2Fsites%2Fsta
ir02%2FShared%20Documents%2Fdata%2FPubChemQC%2FB3LYP%5FPM6%2Fb3lyp%5Fpm6%
5Fver1%2E0%2E0%2Fjson%2Fall%2FCompound%5F{start}%5F{stop}%2Etar%2Exz{enc_ext if check else ''}"
 -c -O 'Compound_{start}_{stop}.tar.xz{ext if check else ''}' > /dev/null 2>&1"""
        )

        cmd_pm6 = (
            lambda check: f"""wget --header="Host: chibakoudai.sharepoint.com"
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36" --header="Accept:
text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,
image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;
q=0.7" --header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8,fr;
q=0.7" --header="Referer: https://chibakoudai.sharepoint.com/sites
/stair01/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2F
stair01%2FShared%20Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5F
ver2%2E0%2E0%2Fjson%2Fall%2FCompound%5F{start}%5F{stop}%2E
tar%2Exz{enc_ext if check else ''}
&viewid=2a7fb7f8%2Df3f8%2D4ad2%2D931e%2Dfc786e938ea8&parent=%2F
sites%2Fstair01%2FShared%20Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5F
ver2%2E0%2E0%2Fjson%2Fall" --header="Cookie: MicrosoftApplications
TelemetryDeviceId=cec40b8a-9870-4c4f-bb71-838a300c8685; MSFPC=GUID=
511089efdbeb49d3923fdc7e6404bd9b&HASH=5110&LV=202303&V=4&LU=1678287635332;
WSS_FullScreenMode=false; {web_cookie}
--header="Connection: keep-alive" "https://chibakoudai.sharepoint.com/sites
/stair01/_layouts/15/download.aspx?SourceUrl=%2Fsites%2Fstair01%2FShared%20
Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5Fver2%2E0%2E0%2Fjson%2Fall%2F
Compound%5F{start}%5F{stop}%2Etar%2Exz{enc_ext if check else ''}" -c -O
'Compound_{start}_{stop}.tar.xz{ext if check else ''}' > /dev/null 2>&1"""
        )

        # download and read checksum
        f_name_checksum = os.path.join(dirname, f"Compound_{start}_{stop}.tar.xz.sha256sum")
        if not os.path.exists(f_name_checksum):
            cmd_checksum = cmd_b3lyp(True) if method == "b3lyp" else cmd_pm6(True)
            cmd_checksum = cmd_checksum.replace("\n", "")
            os.system(cmd_checksum)

        with open(f_name_checksum, "r") as f:
            checksum = f.read().split(" ")[0]

        # download archive
        f_name = os.path.join(dirname, f"Compound_{start}_{stop}.tar.xz")
        download_file = True

        if os.path.exists(f_name):
            with open(f_name, "rb") as f:
                bytes = f.read()
                checksum_arxiv = hashlib.sha256(bytes).hexdigest()

            if checksum_arxiv == checksum:
                download_file = False
                print(f"Checksum match: {os.path.basename(f_name)}")
            else:
                print(f"Checksum mismatch: {os.path.basename(f_name)}")

        if download_file:
            cmd = cmd_b3lyp(False) if method == "b3lyp" else cmd_pm6(False)
            cmd = cmd.replace("\n", "")
            print(f"Downloading: {os.path.basename(f_name)}")
            os.system(cmd)

    except Exception as e:
        print(e)
        raise

    # else:
    #     print(f"Downloaded: Compound_{start}_{stop}.tar.xz")


def download_b3lyp_pm6(web_cookie, start=0, stop=10000, method="b3lyp"):
    path = os.path.join(get_local_cache(), "pubchemqc", method)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    ixs = list(range(start, stop))
    for i in tqdm(ixs):
        download_b3lyp_pm6_item(i, method=method, dirname=path, web_cookie=web_cookie)
    return 0


@cli.command("download")
@click.option("--id", "-i", type=int, default=0, help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=50, help="Chunk size to divide and conquer.")
@click.option("--method", "-m", type=str, default="pm6", help="QM Method used for the calculations.")
@click.option(
    "--cookie-path", "-c", type=str, default="", help="Path to a text file containing the Cookie to access the website."
)
def download(id, chunk_size, method, cookie_path):
    start = id * chunk_size
    stop = (id + 1) * chunk_size
    with open(cookie_path, "r") as f:
        web_cookie = f.read()
    download_b3lyp_pm6(start=start, stop=stop, method=method, web_cookie=web_cookie)
    return 0


def preprocess_archive(start=0, stop=30, method="b3lyp"):
    path = os.path.join(get_local_cache(), "pubchemqc", method, "*.tar.xz")
    arxiv_paths = np.array(sorted(glob(path)))
    print(f"Found {len(arxiv_paths)} archives.")
    ixs = list(range(start, stop))
    if start > len(arxiv_paths):
        return
    for arxiv in tqdm(arxiv_paths[ixs]):
        out = arxiv.replace(".tar.xz", ".pkl")
        if os.path.exists(out):
            continue
        try:
            res = read_archive(arxiv)
            with open(out, "wb") as f:
                pkl.dump(res, f)
        except Exception as e:
            print(e)

    return 0


@cli.command("arxiv2json")
@click.option("--id", "-i", type=int, default=0, help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=50, help="Chunk size to divide and conquer.")
@click.option("--method", "-m", type=str, default="pm6", help="QM Method used for the calculations.")
def arxiv2json(id, chunk_size, method):
    start = id * chunk_size
    stop = (id + 1) * chunk_size
    preprocess_archive(start=start, stop=stop, method=method)
    return 0


if __name__ == "__main__":
    cli()
