# BusScan: A Simulation Platform for Evaluating Bus Based On-street Parking Detection Strategies.
We have developed a simulating system of the micro-urban transportation system: BusScan. BusScan is an open-source simulation framework that models real-time parking detection through bus-mounted sensors while accounting for bus dynamics and driver behavior patterns. The main model could be seperated into three parts.
## 1. Bus Route Simulation
Using **Hough Transformation** together with customized city rule design modeling (boundary design, urban functional zoning, traffic configuration), users only need to upload a bus route map of the city they like to bulid models, and seperate the city into different regions with distinct characteristics. The codes are displayed in [https://github.com/oscakywalker/BusScan-A-Simulation-Platform-for-Evaluating-Bus-Based-On-street-Parking-Detection-Strategies/tree/main/Bus%20Route%20Simulation]. 
We have uploaded different city map demos for users to test. The output is a dynamic graph with the simulation of bus movement trajectory. Here is the effect of Chicago.
![Bus Movement](https://github.com/user-attachments/assets/170d6037-920c-49ce-96ff-0f4d4b849fb7)
## 2. Parking Behavior Simulation

Using **Kaplan-Meier Method** to draw the survival functions of different states of parking behaviors (idle=0, occupied=1)
