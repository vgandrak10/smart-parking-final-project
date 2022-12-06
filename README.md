# CAMERA BASED SMART PARKING LOT

![alt text](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/parking-lot-mess.jpg )
<sup> Image credits: [CALIFORNIA CLEAN ENERGY COMMITTEE](https://www.californiacleanenergy.org/cruising_for_park) </sup>

#### Table of Contents  
&ensp;[BACKGROUND](#background)  
&ensp;[SOLUTION](#solution)  
&ensp;[METHOD](#method)  
&ensp;[RESULTS](#results)    
&ensp;[STEPS TO RUN THE CODE](#steps-to-run-the-code)  
&ensp;[DESIGN GAP](#design-gap)  
&ensp;[FUTURE RESEARCH](#future-research)

## BACKGROUND
Urbanization is leading to more vehicles on the road in several ways. With the increasing number of cars, managing parking lots during peak hours in busy areas is becoming hard and inefficient. On an average, a driver spends close to 10 minutes searching for a free parking space in a mall. According to an IBM study, this is also responsible for up to 30% of traffic congestion in cities. Traffic congestion is a factor in a number of economic and environmental issues, including noise and pollution. Numerous studies have demonstrated that the smart parking system not only reduces traffic issues but also stimulates economic growth and corporate expansion.

## SOLUTION
Smart parking lot management is the use of technology to improve the efficiency, safety, and security of parking lots. This includes the use of sensors, cameras, and computer networks to detect and monitor vehicles entering, leaving, and parked in the lot. Smart parking lot management systems can provide real-time updates on available spaces, alert drivers to potential hazards, and direct them to free spaces. 
Smart parking Solutions based on IoT and AI hold the key to resolving this issue. The smart parking IoT systems that are most frequently used include cameras, overhead radars/lidars, and ground sensors. In this report, I discuss the work I did to approach this problem using cameras.

## METHOD
Nowadays, it seems that cameras, driven by recent advancements in computer vision and AI, seem to be the most direct method. Camera-based smart parking systems can be further classified into two types:  
**1. Cloud-based/server processing:** Streaming the video or series of snapshots to the cloud or on-premise server.  
**2. On-board processing:** Capable of executing vehicle recognition locally on only sending the parking events and a limited number of images when required.

I took the first approach as it allows for a more complex AI model to be developed and deployed in the cloud, where powerful computing resources are available. This will result in a more accurate AI model, without the need to worry about the limitations of an edge device. I drew this idea from the classroom lectures where I worked on AWS cloud instances and realized its computational power.

### Client module
The code is written in Python and uses the OpenCV library and the Mosquitto MQTT broker.
The code captures frames from a webcam, processes them using OpenCV, and then publishes them to the MQTT broker on a particular topic. On the receiving end, the MQTT broker streams the frames to the remote server. The remote server receives the frames for further processing. I was inspired by the camera streaming module discussed in the class. The classroom lectures have motivated me to go for a Pub/Sub architecture using websockets for a better FPS rather than a client/server model using HTTP.

### Remote Server
I developed a remote server using Python that subscribes to a Mosquitto MQTT topic. This server was able to capture frames using OpenCV and then process them to extract relevant information. Afterward, the server streams the results, so the other devices connected to the same network can view the results.  
These frames are then passed through an Object Detection module. Tensorflow Object Detection can be used to detect free parking spaces in a given area. It can detect the presence of cars in a parking lot, and then classify the ones that are unoccupied as free parking spaces. This can be done by training the model on labeled images of cars in various parking lot scenarios, and then using it to classify the images as free or occupied. 
  
#### Model Training 
I downloaded the PKLot dataset that consists of 10000 labeled images. I used the Tensorflow’s [SSD-MobileNet-V2 model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz) which was trained on the COCO dataset, and fine-tuned it on the [PKLot dataset](https://public.roboflow.com/object-detection/pklot). The model can be further tuned to recognize patterns in the images, such as the colors of the cars and their size, in order to improve accuracy. This can be used to help drivers identify and locate free parking spaces, reducing the time it takes for them to reach their destination.  
  
Below are the Tensorboard results of the training process. These images show the training loss, regularization loss and the time taken for each training step.
I trained the model for **26K steps** on **Ubuntu 20.04** with a **Intel® Core™ i7 Processor** .    


![Image2](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-55-36%20TensorBoard.png)


![Image3](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-55-52%20TensorBoard.png)  


![Image4](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-56-06%20TensorBoard.png)  

![Image5](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-56-19%20TensorBoard.png)  

![Image6](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-56-31%20TensorBoard.png) 

![Image7](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screenshot%202022-12-05%20at%2015-56-47%20TensorBoard.png)  

## RESULTS

### Demo Video

[Click Here to Download](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/Screencast%20from%2012-05-2022%2004_14_00%20PM.webm)

![Image8](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/result1.png)  

![Image8](https://github.com/vgandrak10/smart-parking-final-project/blob/main/Results/result2.png)  


## STEPS TO RUN THE CODE

Software requirements : 
- Ubuntu 20.04
- Python 3.7 or above [Installation Guide](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-ubuntu-20-04-server)
- Conda [Installation Guide](https://docs.anaconda.com/anaconda/install/linux/)  

Steps

1. Clone this github repo.
2. Install all the dependencies using [environment_tf2.yml](https://github.com/vgandrak10/smart-parking-final-project/blob/main/environment_tf2.yml) , and run this command:  
`conda env create --name <give-a-name> --file=<path-to-environment.yml>`
3. Activate the python environment using this command  
`conda activate <your-env-name>`
4. Clone the Tensorflow models repository. Please follow instructions given [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-install)  
After the installation, the folder tree should look something like this:  
```
 smart-parking-final-project/
   ├─ exported_model/
   ├─ models/
   ├─ mqtt/
   ├─ Results/
   └── ...
```
5. Open a new terminal and repeat step #3
6. Type `ifconfig` in the terminal and copy the `IP Address`
7. In `mqtt` folder, open `client.py` and add paste the IP Address in `line #9`  
In `line #13`, add you camera index or give the path to input video file. Sample videos are available in this repo. 
8. In the same folder, open `Obj_Det_Server_Flask.py`, and:
   - In `line #30`, add path to Saved_model. `<repo-dir>/exported_model/saved_model`
   - In `line #31`, add path to labels file. `<repo-dir>/`
   - In `line #142`, add add your IP Address
   - In `line #209`, add add your IP Address
9. After making sure you have two open terminals with conda environment and `PYTHONPATH` activated in both,   
first excute `python Obj_Det_Server_Flask.py` in one terminal  
and then `python client.py` in other terminal.

## DESIGN GAP
Initially I planned to take the ‘On-board processing’ approach as discussed in the Method section. My plan was to develop a light weight computer Object Detection algorithm that can run on low-power, resource-constrained devices with limited RAM, storage, and processing power. However, the TensorFlow Lite models that are used for this purpose are not accurate. Hence I decided to take the remote server approach and thereby gave myself the scope to create a more robust and accurate Object Detection model. This approach also proved to have better scalability.

## FUTURE RESEARCH
Future research in smart parking solutions could focus on improving the accuracy and reliability of existing systems, such as by exploring new technologies or algorithms that could be used to better identify parking availability. Additionally, research could also explore potential applications of artificial intelligence and machine learning to better predict parking availability and optimize parking allocation. Additionally, research could focus on the development of secure payment systems to allow drivers to pay for parking using their mobile device, as well as explore new technologies for implementing enforcement and toll collection. Finally, research could also focus on the development of systems that can better integrate with existing public transit options to enable end-to-end travel experiences and reduce the reliance on personal vehicles.
