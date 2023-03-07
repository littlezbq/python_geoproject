# Python_GeoProject
This is a project based on Python, using Flask and Cesium to build a digital terrain feature analysis system on
traditional villages.<br>
<br>Main functions include village edge detection, reachableArea generatation, denglinPoint calculate and exposivePoint calculate etc.
Details for technical words are explained below.

## Project Structure
* **Up_to_Domain**: *algorithm to calculate and generate reachableArea*.<br>
* **ascensionpoint_generate**: *algorithm based on viewshed generated to calculate ascensionpoint*.<br>
* **road_intersection**: *algorithm based on deeplearning and tensor voting to extract road intersection from remote sensing data*.<br>
* **static**: *directory for static resources and running results eg. js files, pics, raw data, algorithm result etc*.<br>
* **commonutils**: *directory for some useful tools*.<br>
* **config**: *directory for storing params, including save path and some hypeparameters*.<br>

* **run.py**: *main entrance to run the system*.<br>

other parts like village_space_quantization holds no
## Quick Start
### Prerequisites
* Python == 3.9.12
* CUDA >= 10 <br>
**WARNING**: Difference version of Python or other libs may cause install failed or other errors<br>

First, create a virtual environment or install dependencies directly with:<br>
``` pip3 install -r requirements.txt```

### Data preparation
There are two main data source in the project. First is the **remote and DEM** data of the village waiting for analyze; Second is the data for **village edge detection module**.
