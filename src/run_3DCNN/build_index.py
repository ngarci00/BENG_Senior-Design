import os, json
from config import raw_frames_dir, ann_dir, index_json_path, LABEL_TO_INT

img_extensions = ['.jpg', '.jpeg', '.png']

def list_sorted_images(frames_dir):
    names = [f for f in os.listdir(frames_dir) if f.lower().endswith(img_extensions)]#sort by frame number
    def key_func(name):
        stem = os.path.splitext(name)[0]#get the part of the filename without the extension 
        try: 
            return (0,int(stem))
        except:
            return (1,stem)
    names.sort(key=key_func)
    return names #returns list of image file names 

def build_one(video_id, label_str, ann_dir, frames_dir):
    frame_names = list_sorted_images(frames_dir)

    #Annotated frames are those with a matching Json file stem
    annotated = []
    ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith('.json')]#list of json files in the annotation directory
    ann_stems = set(os.path.splitext(f)[0] for f in ann_files)
    for fn in frame_names:
        stem = os.path.splitext(fn)[0]
        if stem in ann_stems:
            annotated.append(fn)
    item = {
        'video_id': video_id,
        'label': int(LABEL_TO_INT[label_str]),
        'frames_dir': os.path.abspath(frames_dir),
        'annotated_dir': os.path.abspath(ann_dir),
        'frame_names': frame_names,
        'annotated_frames_names': annotated,
    }
    return item#returns a dictionary with video_id, label, frames_dir, annotated_dir, frame_names, and annotated_frames_names

def main():#builds an index of the dataset by iterating through the annotation directories for each label and video, and creating a list of dictionaries containing information about each video and its frames.
    items = []

    for label_str in ['PASS', 'FAIL']:
        label_dir = os.path.join(ann_dir, label_str)
        if not os.path.isdir(label_dir):#check if the label directory exists
            print(f"Warning: Label directory {label_dir} does not exist. Skipping.")
            continue

        for video_id in os.listdir(label_dir):
            ann_dir = os.path.join(label_dir,video_id)
            if not os.path.isdir(ann_dir):#check if the annotation directory for the video exists
                print(f"Warning: Annotation directory {ann_dir} does not exist. Skipping.")
                continue

            frames_dir = os.path.join(raw_frames_dir, video_id)
            if not os.path.isdir(frames_dir):#check if the frames directory for the video exists
                raise FileNotFoundError(f"Warning: Frames directory {frames_dir} does not exist. Skipping...")

            items.append(build_one(video_id,label_str,ann_dir,frames_dir))#builds a dictionary for the video and appends it to the index list
        if len(items) == 0:
            raise RuntimeError(f"No valid videos found for label {label_str}. Please check the directory structure and contents.")
        
        os.makedirs(os.path.join(index_json_path), exist_ok=True)
        with open(index_json_path, 'w') as f:
            json.dump(items, f, indent=2)#saves the index list as a JSON file in the specified index directory
        print(f"Index built successfully with {len(items)} items. Saved to {index_json_path}.")

if __name__ == "__main__":
    main()#runs the main function to build the index when the script is executed directly