# Python_GeoProject
This is a project based on Python, using Flask and Cesium to build a digital terrain feature analysis system on
traditional villages.<br>
<br>Main functions include village edge detection, reachableArea generatation, denglinPoint calculate and exposivePoint calculate etc.
Details for technical words are explained below.

## Project Structure
* **3Dplot**: *some 3D visualization effects(testing)*.<br>
* **Up_to_Domain**: *algorithm to calculate and generate reachableArea*.<br>
* **calculate_denglinPoint**: *algorithm based on viewshed generated to calculate denglinPoint*.<br>
* **static**: *directory for static resources eg. js files, pics, raw data, algorithm result etc*.<br>
* **templates**: *directory to store html files*.<br>
* **app.py**: *main entrance to run the system*.<br>

## Quick Start
### Prerequisites
* Python == 3.7.10
* CUDA >= 10 <br>
**WARNING**: Difference version of Python or other libs may cause install failed or other errors<br>

First, create a virtual environment or install dependencies directly with:<br>
``` pip3 install -r requirements.txt```

### Data preparation
There are two main data source in the project. First is the **remote and DEM** data of the village waiting for analyze; Second is the data for **village edge detection module**.
