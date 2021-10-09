# Face detection - DNN
Application to detect faces with or without mask on images.
### Usage:
- Clone this repo

- Install requirements: ```pip3 install requirements.txt```

- On the main folder, Run:
  - ```python3 detec.py [xxx].jpg```
  
  For detec one single picture, or
  
  - ```python3 detec.py dataset/[xxx]/```
  
  To loop throughout folder and generate a log file with the time and filename along with the pic results on the "[xxx]/output/".
  
### Extra:  
- /dataset/base20 and ../base20  contains the dataset sent to robocup 2020 and 2021.
- /dataset/test could be used for test. 
- /models/ contais the models used to detec faces and mask.
