# Installation

This document details the steps required to install the requisite packages via a Python environment, assuming that an Anaconda distribution is used on a Windows PC. If using native Python (without any distributions): [download Python here](https://www.python.org/downloads/release/python-387/) and install; and go directly to step 6 (until step 9).

1.	Open Anaconda Navigator

2.	Open a new Console window (aka Terminal; or also a Powershell Window)

3.	Create a new Python environment (named: py38) using:

`conda create -y --name py38 python=3.8`

4.	Activate the new environment:

`conda activate py38`

5.	Install pip (in case it wasn’t installed with the py38):

`conda install pip`

6.	On the Terminal or Console, navigate to the folder where the [requirements.txt](https://github.com/anandpr1602/Electrode_Thickness_Calculator/blob/main/requirements.txt) is downloaded – commonly using the command:

`cd <path>/<folder>`

7.	Install all the required packages using Pip (Pip is required because some packages, like keyboard, are not available in the conda-forge channels):

`pip install -r requirements.txt` or `python -m pip install -r requirements.txt`

8.  If special image file types (e.g., HDR, RAW, etc.) need to be handled, you may install the FreeImage plugin with ImageIO:
    * On Python `import imageio` followed by `imageio.plugins.freeimage.download()`
    * On the command line `imageio_download_bin freeimage`

9.	You may now execute the “Electrode_Thickness_Calculator.py” file by following the [usage commands](https://github.com/anandpr1602/Electrode_Thickness_Calculator#usage).

10. You may deactivate the Anaconda environment and return to the base installation using:

`conda deactivate`
