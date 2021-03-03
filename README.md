# Electrode_Thickness_Calculator
is a Python script that loads a stack of X-ray CT images from a directory and measure the thickness of each electrode layer assembled between two current collectors. The script has been developed aimed at high-resolution CT scans for e.g., from Zeiss Xradia 620 Versa or 810 Ultra CT imaging systems.

Upon loading the entire image stack, the central slice of the orthogonal projection is selected for interrogation. The brightest layers (current collectors) are segmented via [Otsu binary thresholding](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu). The remaining layers are then classified using [K-Means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) --> both electrodes into 1 cluster and separator layer into the second cluster. Because both electrodes have similar X-ray absorption values, they are clustered into one label.

The threshold and the K-Means clustering values are then used to segment the pixels into user defined classes in all the slices. Each class is assigned a unique label and the contours between each class identified using a [Marching-squares algorithm](https://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html).

The script then saves the contours of the central slice as an Excel file. The average thickness of each layer and their standard deviation from all the slices are computed and saved as another Excel file.

* **NB 1: This version uses the `multiprocessing` module that uses all the available CPU cores to process slices in parallel.**

* **NB 2: The `classes` variable in line 20 of the python file denotes the number of clusters to categorise the pixels (i.e., the number of layers to segment in the image stack). The default is 3 clusters.**

* **NB 3: The `separator_dilate` variable in line 23 of the python file denotes the diameter by which to dilate the pores of the separator layer for accurate estimation of contours. This function can be decreased if the contrast between the separator and electrodes is poor; or increased to obtain a smoother contour of the separator layer.**

# Installation:

* Please see the [Installation.md](https://github.com/anandpr1602/Electrode_Thickness_Calculator/blob/main/Installation.md) file for step-by-step instructions on performing a clean installation using a Python 3.8 environment.

# Requirements:
## Tkinter:
### From Python 3.1 - Tkinter is included with all standard Python distribution.
### For Python 3.x use:
`import tkinter.Tk()` and NOT `import Tkinter.Tk()`

## Numpy:
`pip install numpy` or `python -m pip install numpy`

## Pandas:
`pip install pandas` or `python -m pip install pandas`

## Scipy:
`pip install scipy` or `python -m pip install scipy`

## Matplotlib:
`pip install matplotlib` or `python -m pip install matplotlib`

## PIL:
`pip install Pillow` or `python -m pip install Pillow`

## Scikit-Image:
`pip install scikit-image` or `python -m pip install scikit-image`

## Scikit-Learn:
`pip install scikit-learn` or `python -m pip install scikit-learn`

## ImageIO:
`pip install imageio` or `python -m pip install imageio`
* For compatible file formats: see https://imageio.readthedocs.io/en/stable/formats.html or use `imageio.formats.show()`

## Use the FFMPEG plugin to open a variety of video formats:
`pip install imageio-ffmpeg` or
`python -m pip install imageio-ffmpeg`
* See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#ffmpeg

## Use the FreeImage plugin to open a variety of image formats such as .RAW, .HDR, etc.:
On Python `imageio.plugins.freeimage.download()` or on command line `imageio_download_bin freeimage`
* See http://freeimage.sourceforge.net/

## Use the ITK plugin to open a variety of image formats such as .HDF5, .HDR, .NHDR, etc.:
`pip install itk` or `python -m pip install itk`
* See https://imageio.readthedocs.io/en/stable/format_itk.html#itk

# Usage:
## Run the [Electrode_Thickness_Calculator.py](https://github.com/anandpr1602/Electrode_Thickness_Calculator/blob/main/Electrode_Thickness_Calculator.py) directly:
* On the command line or in a Python console run:

`python Electrode_Thickness_Calculator.py` or `Electrode_Thickness_Calculator.py`

