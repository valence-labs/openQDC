import os

import click
from tqdm import tqdm

from openqdc.utils.io import get_local_cache


def download_b3lyp_pm6_item(i, method="b3lyp"):
    try:
        step_size = 25000
        start = str(i * step_size + 1).rjust(9, "0")
        stop = str((i + 1) * step_size).rjust(9, "0")

        cmd_b3lyp = f"""wget --header="Host: chibakoudai.sharepoint.com"
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36" --header="Accept:
text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image
/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
--header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7"
--header="Referer: https://chibakoudai.sharepoint.com/sites/stair02/Shared
%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fstair02%2FShared%20
Documents%2Fdata%2FPubChemQC%2FB3LYP%5FPM6%2Fb3lyp%5Fpm6%5Fver1%2E0%2E0%2F
json%2Fall%2FCompound%5F{start}%5F{stop}%2Etar%2Exz&viewid=f6d34767%2
D64f0%2D480e%2Dab70%2Dd8524dbdc74e&parent=%2Fsites%2Fstair02%2FShared%20
Documents%2Fdata%2FPubChemQC%2FB3LYP%5FPM6%2Fb3lyp%5Fpm6%5Fver1%2E0%2E0%2F
json%2Fall" --header="Cookie: MicrosoftApplicationsTelemetryDeviceId=cec40b8a
-9870-4c4f-bb71-838a300c8685; MSFPC=GUID=511089efdbeb49d3923fdc7e6404bd9b&
HASH=5110&LV=202303&V=4&LU=1678287635332; WSS_FullScreenMode=false; FedAuth=
77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWV
tYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzI1YTJjOTIyMjVkMWNlNzlkOWVmN2NjYmRjNzc5Y
WI1MmJhMDY2N2E5NDRmZTg3NGFmZTFhZjRjMjQ0OGE3ZTUsMCMuZnxtZW1iZXJzaGlwfHVybiU
zYXNwbyUzYWFub24jMjVhMmM5MjIyNWQxY2U3OWQ5ZWY3Y2NiZGM3NzlhYjUyYmEwNjY3YTk0N
GZlODc0YWZlMWFmNGMyNDQ4YTdlNSwxMzM0MTM0MDE0OTAwMDAwMDAsMCwxMzM0MTYwOTY0NjE
yNDYxODcsMC4wLjAuMCwyNTgsNTIyZTlhNzYtYWNiZC00MDJiLWEyZmMtN2NmNjg5ZGRmNTkwL
CwsOTA3NGUyYTAtNDAyNS0yMDAwLWE2NDEtYjdiNDU2N2JlNzI5LDc2MjNlM2EwLWQwYzEtMjA
wMC1hNjQxLWI4MGEyYmU3YmExZSxSNFZHUmtMSXdrT3RETDI0alZUSm9RLDAsMCwwLCwsLDI2N
TA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxOTU2NzYsR0FkeFdYM3FnLXBsUDRlOVhCUDF
5MTZpZmpVLFN2QXdUYjI3b0MrM0RKa2hsODdRNnhkVFVpQ2l5U0tqU2RxZ3EzNUFsa2lOcmczQ
0NJZWplSmNCR1dteCtWRS8zL1lacmZFYVk3eGJGVDFSWHoxREhXVE5oK0dUSzhiQ0FYOUUxQ20
yUXpPVG5jZm5MNDdpWUVOLzRzUzVTdnFpbnZ1eDh3L2FrQmZISW01Zlpqbk02c25KOWs5V294b
24wY1F1dUgvY1d0UUNOTkJ2WmtvRkVReitUVldBSmtQRmtxNUlibXFyL2hMUzcreGlqS3FWeXd
WZldIeGp3Q25iUTlzYitjcnhqcDlYR2szLzZ1YUFUeTMyVi9MVFBBdmM4am9wL2hRdjV4bXBnZ
k95M1cvSkljNXpPTlBlbmdQVkl2MXJtb0EwS0h6QVpCNjBnY3pEM1BaYWZVZHFsdGV6RndRTTV
xSFB3Q1hqelJ3SDRyL0Vsdz09PC9TUD4=" --header="Connection: keep-alive" "https
://chibakoudai.sharepoint.com/sites/stair02/_layouts/15/download.aspx?
SourceUrl=%2Fsites%2Fstair02%2FShared%20Documents%2Fdata%2FPubChemQC%2FB3LYP
%5FPM6%2Fb3lyp%5Fpm6%5Fver1%2E0%2E0%2Fjson%2Fall%2FCompound%5F{start}%5F
{stop}%2Etar%2Exz" -c -O 'Compound_{start}_{stop}.tar.xz' > /dev/null  2>&1"""

        cmd_pm6 = f"""wget --header="Host: chibakoudai.sharepoint.com"
--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36" --header="Accept:
text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/
webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
--header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7"
--header="Referer: https://chibakoudai.sharepoint.com/sites/stair01/
Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fstair01%2F
Shared%20Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5Fver2%2E0%2E0%2F
json%2Fall%2FCompound%5F{start}%5F{stop}%2Etar%2Exz&viewid=2a7fb7f8
%2Df3f8%2D4ad2%2D931e%2Dfc786e938ea8&parent=%2Fsites%2Fstair01%2FShared
%20Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5Fver2%2E0%2E0%2Fjson%2Fall"
--header="Cookie: MicrosoftApplicationsTelemetryDeviceId=cec40b8a-9870-
4c4f-bb71-838a300c8685; MSFPC=GUID=511089efdbeb49d3923fdc7e6404bd9b&
HASH=5110&LV=202303&V=4&LU=1678287635332; WSS_FullScreenMode=false;
FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+
VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzI1YTJjOTIyMjVkMWNl
NzlkOWVmN2NjYmRjNzc5YWI1MmJhMDY2N2E5NDRmZTg3NGFmZTFhZjRjMjQ0OGE3ZTUs
MCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jMjVhMmM5MjIyNWQxY2U3OWQ5
ZWY3Y2NiZGM3NzlhYjUyYmEwNjY3YTk0NGZlODc0YWZlMWFmNGMyNDQ4YTdlNSwxMzM0
MTM0MDE0OTAwMDAwMDAsMCwxMzM0MTYyODU0MDg5NTI4NjAsMC4wLjAuMCwyNTgsNTIy
ZTlhNzYtYWNiZC00MDJiLWEyZmMtN2NmNjg5ZGRmNTkwLCwsOTA3NGUyYTAtNDAyNS0y
MDAwLWE2NDEtYjdiNDU2N2JlNzI5LDdiMzVlM2EwLWYwYmMtMjAwMC05ZmM0LWU4ODFi
NmM4NGNjZSxSNFZHUmtMSXdrT3RETDI0alZUSm9RLDAsMCwwLCwsLDI2NTA0Njc3NDM5
OTk5OTk5OTksMCwsLCwsLCwwLCwxOTU2NzYsR0FkeFdYM3FnLXBsUDRlOVhCUDF5MTZp
ZmpVLFNRKzRNWHJYNzRaSHUxMUxVcE9adVZTT1BiK0xJTllwdHY3YTBIM2hLOEdPNThw
L1F1VDZ2K1FTWUZWekpqL3FFblp1TUhlVjFWaytxQ2lhSC9tWXNkMXlRM1N6YVRJaUtx
cHVsWkhTUEVsWmg4TmtHMDhzT3ZXN2J5dW1OMmY4dFJMUVNmekFYQnREVzdnN1hUMUgy
MUsyVlFyUys3WEtHSXpvMmFjQU5XQVNMUTQwRTJFVEd5SlhjRE9ya09HS2ZiSThDVWk4
bHNwaFRVZTJ6UjBPbjRZaGVFSDUrYTJsSVB4bUNLdG0weXBsS1V6M2pEakxHcml0Rk5l
dWdUdEk0WUpZY3ZOcGZENmZDU0M3dGFhOXlXYmpZUU1QMlhmbXd1bGtkRCs1aUdYRjZi
SFNBNXlNY1FuUXBCVWZjSjgwcDZXSmtlbXlzMWlWZXA5RGU4UHpvZz09PC9TUD4="
--header="Connection: keep-alive" "https://chibakoudai.sharepoint.com/
sites/stair01/_layouts/15/download.aspx?SourceUrl=%2Fsites%2Fstair01%2F
Shared%20Documents%2Fdata%2FPubChemQC%2FPM6%2Fpm6opt%5Fver2%2E0%2E0%2F
json%2Fall%2FCompound%5F{start}%5F{stop}%2Etar%2Exz" -c -O
'Compound_{start}_{stop}.tar.xz' > /dev/null  2>&1"""

        cmd = cmd_b3lyp if method == "b3lyp" else cmd_pm6
        cmd = cmd.replace("\n", "")
        os.system(cmd)
    except Exception:
        pass
    # else:
    #     print(f"Downloaded: Compound_{start}_{stop}.tar.xz")


def download_b3lyp_pm6(start=0, stop=10000, method="b3lyp"):
    path = os.path.join(get_local_cache(), "pubchemqc", method)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    ixs = list(range(start, stop))
    for i in tqdm(ixs):
        download_b3lyp_pm6_item(i, method=method)


@click.command()
@click.option("--id", "-i", type=int, default=0, help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=50, help="Chunk size to divide and conquer.")
@click.option("--method", "-m", type=str, default="pm6", help="QM Method used for the calculations.")
def main(id, chunk_size, method):
    start = id * chunk_size
    stop = (id + 1) * chunk_size
    download_b3lyp_pm6(start=start, stop=stop, method=method)


if __name__ == "__main__":
    main()
