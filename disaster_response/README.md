
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instr)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
- python
- json
- plotly
- pandas
- flask
- sqlalchemy
- ntlk
- numpy

## Project Motivation<a name="motivation"></a>

This Project is based on data provided by figure eight. This data contains the messages sent by people during natural disaster which can be analyze for future prediction:

1. Who are the people should be given priority during disaster
2. Categories type emergencies.
3. Identify situtation and repsonse in minimum time.

## File Descriptions <a name="files"></a>
In this repository following files included:



 
 * app
   * templates
      * go.html # main page of web app
      * master.html # classification result page of web app
   * run.py
   
 * data
   * disaster_categories.csv # data to process 
   * disaster_messages.csv # data to process
   * process_data.py
   * DisasterResponse.db # database to save clean data to
   
 * models
 * train_classifier.py
 


## Instructions<a name="instr"></a>
 - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credit to figure eight for providing the data and Udacity for guidance and instructions. 

[contact](https://github.com/ksekara/disaster_response)
  
