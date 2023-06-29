# Face detection - DNN
Application to detect faces with or without mask on images.
### How to use:
- Clone this repo using:
-  - ```git clone https://github.com/pinguimbotsathome/Face-detection.git```

- Install the requirements under the file requirements.txt

- On the main folder, Run:

  To detect one single picture use:
  
  - ```python3 detec.py [xxx].jpg```

  To loop throughout folder and detect pictures:

  - ```python3 detec.py dataset/[xxx]/```

  It will generate a log file with the time and filename along <br />
  with the pic results on the "[xxx]/output/" .
  


  
### Extra:  
- /dataset/base20 and ../base20  contains the dataset sent to robocup 2020 and 2021.
- /dataset/test could be used for test. 
- /models/ contais the models used to detec faces and mask.
