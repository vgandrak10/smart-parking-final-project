# Importing Libraries
import cv2 as cv
import paho.mqtt.client as mqtt
import base64
import time


# Raspberry PI IP address
MQTT_BROKER =  "Your-System-IP-Address" # In terminal type ifconfig
# Topic on which frame will be published
MQTT_SEND = "home/server"
# Object to capture the frames
cap = cv.VideoCapture("Path-to-the-video-file")

# Set FPS
#cap.set(cv.CAP_PROP_FPS,10) 

# Phao-MQTT Clinet
client = mqtt.Client()
# Establishing Connection with the Broker
client.connect(MQTT_BROKER)
try:
 while True:
  start = time.time()
  # Read Frame
  _, frame = cap.read()
  # Encoding the Frame
  _, buffer = cv.imencode('.jpg', frame)
  # Converting into encoded bytes
  jpg_as_text = base64.b64encode(buffer)
  # Publishig the Frame on the Topic home/server
  client.publish(MQTT_SEND, jpg_as_text)
  end = time.time()
  t = end - start
  fps = 1/t
  print(fps)
  time.sleep(1/100)
except:
 cap.release()
 client.disconnect()
 print("\nNow you can restart fresh")