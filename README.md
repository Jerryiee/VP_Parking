# VP_Parking

#### This project is combination of Data analysis from UNIZA.park.IN.temp,UNIZA.park.OUT sensors and computer vision network.

![Screenshot 2022-08-24 093806](https://user-images.githubusercontent.com/39840269/186360953-110d638c-a17e-4dd6-8c74-baaec0e4dc9c.png)


Detecting vehicles using Yolov5
https://github.com/ultralytics/yolov5

![Screenshot 2022-08-23 141741](https://user-images.githubusercontent.com/39840269/186155954-ca5e52df-2b4d-497b-8b0f-029fe8de62bc.png)


**Future plans:**

Jetson nano ✔️ <br />
Line for counting vehicles ✔️<br />
Data analysis <br />
Identification ✔️ <br />
Draw lines ✔️ <br />
Possibility count more lines at once ✔️ <br />
Entry camera ✔️ <br /> 
Two reference lines ✔️<br />
TensorRT

### Installation Jetson nano <br />

1. First of all install Jetpack os via sdk manager, instructions here [sdk](https://www.waveshare.com/wiki/JETSON-NANO-DEV-KIT) - requieres second computer with ubuntu <br />
2. Enable SD card for J101 carrier board. Be aware, use this guide only for enable sd card, we falsh system to sd card in next step [enable sd card](https://wiki.seeedstudio.com/J101_Enable_SD_Card/) <br />
3. Boot from sd card - download [change_rootfs_storage.zip](https://github.com/Jerryiee/VP_Parking/files/9893646/change_rootfs_storage.zip)<br/>                       ```
sudo ./change-rootfs-storage.sh [for example /dev/mmcblk1p1]                                                                                                    ```<br/>reboot and everything is ready <br />
4. Install SDK componets - use sdk manager again <br />
5. YOLOv5 install <br />
[instructions](https://wiki.seeedstudio.com/YOLOv5-Object-Detection-Jetson/) step 1 - 8 <br />


### Counting lines <br />
https://user-images.githubusercontent.com/39840269/188903893-9096fc75-efd7-4844-b93e-e8abec851ef7.mp4

### Identification<br />
![Screenshot 2022-09-08 193235](https://user-images.githubusercontent.com/39840269/189187889-78906253-27f8-431d-b966-14bf213a78c4.png)
![image](https://user-images.githubusercontent.com/39840269/189189265-78040d38-c9c9-48f2-bdd0-8f60a3766861.png)

### Draw lines <br />
https://user-images.githubusercontent.com/39840269/189973714-8ec64ca3-15a4-4162-8c2a-124359a99444.mp4

### Possibility count more lines at once  <br />
https://user-images.githubusercontent.com/39840269/190453627-fb09665a-1f2e-4b45-880e-5106aa487944.mp4


### Two reference lines  <br />


