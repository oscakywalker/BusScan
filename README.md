# BusScan: A Simulation Platform for Evaluating Bus Based On-street Parking Detection Strategies.
We have developed a simulating system of the micro-urban transportation system: BusScan. BusScan is an open-source simulation framework that models real-time parking detection through bus-mounted sensors while accounting for bus dynamics and driver behavior patterns. The main model could be seperated into three parts.
## 1. Bus Route Simulation
Using **Hough Transformation** together with customized city rule design modeling (boundary design, urban functional zoning, traffic configuration), users only need to upload a bus route map of the city they like to bulid models, and seperate the city into different regions with distinct characteristics. The codes are displayed in [https://github.com/oscakywalker/BusScan-A-Simulation-Platform-for-Evaluating-Bus-Based-On-street-Parking-Detection-Strategies/tree/main/Bus%20Route%20Simulation]. 

We have uploaded different city map demos for users to test. The output is a dynamic graph with the simulation of bus movement trajectory. Here is the effect of Chicago.
![Bus Movement](https://github.com/user-attachments/assets/170d6037-920c-49ce-96ff-0f4d4b849fb7)
## 2. Parking Behavior Simulation
Using **Kaplan-Meier Method** to draw the survival functions of different states of parking behaviors (idle=0, occupied=1). Using **Kernel Density Estimation** to model idle and occupied duration. The model is trained by the hostorical parking data in The Chinese University of Hong Kong, Shenzhen.

The codes are displayed in [https://github.com/oscakywalker/BusScan-A-Simulation-Platform-for-Evaluating-Bus-Based-On-street-Parking-Detection-Strategies/tree/main/Parking%20Behavior%20Simulation]. Here is the pulse diagrams of 20 parking spaces between 0(idle) and 1(occupled).
![Emulation of 20 parking spots in one day](https://github.com/user-attachments/assets/0094e404-77cc-44cf-914c-7f53c924071d)
## 3. Deployment and Cross-city Validation
This section demonstrates BusScan's generalizability through implementation in Melbourne's central business area. (It is also stable in other cities with public data) The steps are as follows:
   1. Upload the image of bus route map.
   2. Upload the public data of Melbourne.
   3. Design the city boundary, region boundary and the average speed.
   4. Using Kaplan-Meier Estimator to learn the survival function, using the Kernal Density Estimation to generate the duration of each state.
   5. Deploy parking spaces into the city.
   6. Record the 
