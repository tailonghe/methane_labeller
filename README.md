# Methane plume labeller
A GUI for labelling methane plumes from O&G super emitters, using images from Landsat satellites (Landsat 4, 5, 7, and 8) and Sentinel-2.

## Steps to set up the environment
* Install the gcloud CLI following instructions here: https://cloud.google.com/sdk/docs/install#linux
* Install Google Earth Engine API following these instructions: https://developers.google.com/earth-engine/guides/python_install-conda. Don't skip the "Setting up authentication credentials" step.
* Set up the Python environment using the environment file provided: ```conda env create -f environment_ch4_labeller.yml```
* X11 forwarding is required if GUI is run remotely.

## Steps to use the GUI
* Run the GUI: ```python ch4_labeller.py'''. Below is a screenshot of the interface.
* Choose a satellite and provide information about longitude, latitude, year, and month.
* Click search. Satellite images available for the 1-month time window will show up in a few seconds for Landsat data. Searching for Sentinel-2 images takes a longer time due to the higher revisit frequency. Below is a screenshot of the interface after data searching is complete.
* The message box in the bottom-right will show an error message if no images are available for the month. 
* You can use the right arrow (">>") button and left arrow ("<<") to switch between different images. Drawn masks will not be lost due to the switch.
* Click and drag using Left Mouse Button to draw masks over methane plumes. Click and drag using Right Mouse Button to erase.
* Use the "Clear mask" button to completely remove the mask you draw.
* Click "Save" to store the data, the mask, and a screenshot of the labeling for the current canvas. #You would have to click "Save" after labeling each satellite image.#
