## Dataset

### Data availability
The data collection process is still ongoing. Once the campaign has finished, we 
plan to host the dataset on a publicly accessible server and make it available 
upon request. In the meantime, please contact Dr. Diego Hiadalgo-Mazzei
([dahidalg@clinic.cat](mailto:dahidalg@clinic.cat)) to inquire about the dataset 
used in this work.

### File structure
- the `dataset/` folder should have the following structure
  ```
  ts4h2022/
    dataset/
      - raw_data/
        - unzip/
          - 1274370/
            - ACC.csv
            - BVP.csv
            - EDA.csv
            - HR.csv
            - IBI.csv
            - info.txt
            - tags.csv
            - TEMP.csv
          - 1311603/
          - ...
        1274370.zip
        1311603.zip
        ...
      - README.md
  ```
  
  
### Raw data information for Empatica E4
```
.csv files in this archive are in the following format:
The first row is the initial time of the session expressed as unix timestamp in UTC.
The second row is the sample rate expressed in Hz.

TEMP.csv
Data from temperature sensor expressed degrees on the Celsius (°C) scale.

EDA.csv
Data from the electrodermal activity sensor expressed as microsiemens (μS).

BVP.csv
Data from photoplethysmograph.

ACC.csv
Data from 3-axis accelerometer sensor. The accelerometer is configured to measure acceleration in the range [-2g, 2g]. Therefore the unit in this file is 1/64g.
Data from x, y, and z axis are respectively in first, second, and third column.

IBI.csv
Time between individuals heart beats extracted from the BVP signal.
No sample rate is needed for this file.
The first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s).
The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).

HR.csv
Average heart rate extracted from the BVP signal.The first row is the initial time of the session expressed as unix timestamp in UTC.
The second row is the sample rate expressed in Hz.

tags.csv
Event mark times.
Each row corresponds to a physical button press on the device; the same time as the status LED is first illuminated.
The time is expressed as a unix timestamp in UTC and it is synchronized with initial time of the session indicated in the related data files from the corresponding session.
```