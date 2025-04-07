# Compound_Drought_Heatwave

This repository hosts the code for detecting extreme events in summer (June to August) and identifying the characteristics (frequency, days, mean duration, mean severity) annually based on Run theory. 
In addition, the code for quantifying the impacts of extreme events on vegetation index anomaly is also provided.

## code list
- 01_heatwave.py: indentify characteristics of summer heatwave events
- 02_drought.py: indentify characteristics of extreme drought events
- 03_compound.py: indentify characteristics of summer compound extreme drought-heatwave events
- 04_csif.py: CSIF 4-day clear-sky data processing (monthly integration)
- 05_vegetation.py: seperate the VIs anomaly into 4 types, including *natural VIs anomaly*、*heat-induced VIs anomaly*、*drought-induced VIs anomaly*、*compound-induced VIs anomaly*
