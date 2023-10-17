## DeepDreaming
Hello visitor, 
we are the students of KIT college of Engineering who are creating this deepdream model as part of our internship

#### What we are doing?
we want to create one deep dream learning model that will help students and learners to understand how deep dream as well as image classification works

#### What will be there in our model ?:
The model will first display the information about the image classifiacation and deep dream. Also there will be option to try deep dream if you click to try deep dream it will ask on which image classification model you want to apply deep dream then it will ask to upload the image and also to select the specific layer of selected model to apply deep dream (we are interested in specific layer because the specific layer of CNN detects the specific feature from image while performing image classification). the the model will display the original image and deep dream image side by side which will be helpful for users to understanf deep dream and image classification

#### How to run project?:
open the final.ipynb file in google colab run all cells one by one. Once you reach the command :
    !npx localtunnel --port 8501
then after executing this command you will get one url follow that url, one page will open in new tab which will ask for local tunnel endpoint. You can get this endpoint in log.txt file of colab project. Copy the IPv4 address from external url,in my case in logs.txt the url is :
  External URL: http://34.125.156.61:8501
thus the local tunnel endpoint will be :
  34.125.156.61
once you will enter the endpoint the project will run successfully and you can use the model😊😊

#### Contents of file:
1.final.ipynb : contains final project which can be used to run and see our model.
2.deep_dream.ipynb : contains initial python notebook version of project that can process with two model (inceptionV3 and resnet50 but has some format error).
3.dream.py : contains the logic for deep dream.
4.index.py : contains streamlit frontend for our app which can process signle model.
5.main.py : contains streamlit frontend for our app which can process two models.

#### Link for project report:
https://www.overleaf.com/read/mfjgxmsnhkpc

#### Streamlit Reference Link:
https://youtu.be/VqgUkExPvLY?si=qOMT9QrsQUrynK6w
