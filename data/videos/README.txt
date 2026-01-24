Template to understand the format of the videos: 
- RG (Right-side Good)
= RB (Right-side Bad)
- LG (Left-side Good)
= LB (Left-side Bad) 
- MG (Middle-side Good) 
- MB (Middle-side Bad)
 
-  The file is named after #.RB etc for example:
- (1st video) Good right intubation is:   1.RG.avi


// to convert the videos into images (frames) please use the command, if confused use this webstie:
https://github.com/wkentaro/labelme/blob/main/examples/video_annotation/README.md 

video-toimg nameofvideo.avi 

--> If for some reason the command doesn't work please use the following command to force the path to be using the .venv folder

source .venv/bin/activate

--- then the line above ( video-toimg nameofvideo.avi ) ---

--> To open files using the terminal and IINA use the following command:

open -a IINA name_file.avi 

--> STEPS TO TURN JSON FILES INTO VIEWABLE VIDEOS:
1. Run the script "normrectangles.py" in the terminal ex: python3 normrectangles.py "input_folder"
2. Run the command: labelmetk json-to-visualization "input_folder" -- this should produce a folder with a .export at the end
3. Run the following command to turn the frames into a video: video-fromimg -i "name_file.export/*/visualization.jpg" --fps 30 output_file.mp4
--------

Current List: (12 Good/ 9 Bad) <- please update as we add more videos 

1.	Right side (performer) good 
2.	Middle (performer) good
3.	Left (p) good
4.	Right (p… right is better?) bad 
1.	Nor far enough 
5.	Middle good
1.	Hit vocals 
6.	Right … good 
7.	Bad glidescope insert…. Tube inserted into esophagus 
8.	Middle … bad … tube in the esophagus 
9.	Identifying epiglottis (contains perfect view) --> OMIT
1.	Halfway to see it 
10.	Middle angle… bad… hitting epiglottis 
11.	Middle good 
12.	Middle bad … glidescope too deep and inserted into stomach 
13.	Middle… bad (deciding which angle is best) multiple attempts and inserted into stomach 
1.	Middle angle makes you miss seeing epiglottis
14.	Right angle (tongue less in the way) good 
15.	Left angle g… bad visualization 
16.	Right… bad.. difficult insertion 
17.	Right … bad … difficult insertion 
18.	Right … good 
19.	Left … good 
1.	Metal (styleates???) should not be visible past tube 
20.	Middle … good ? (Hit wall between esophagus and vocal cords) blade angled to the left so that the tube could be inserted easier 
21.	Right… good… faster… tube can be inserted straight instead of at an angle 
22.	Right … good… educational explanations happening during video --> OMIT
23.	Right… good (long video) 
24.	
== Current plan, use CV to train 20, test on 1.