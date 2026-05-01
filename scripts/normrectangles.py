# To be able to run " labelmetk json-to-visualization" we need to normalize the rectangles in the json files
# labelme is sensitive to the order of the points in the rectangle, therefore giving us an error
# this script should fix that and needs to be run before the json-to-visualization command - nico
import json, glob, os, sys 

#Main function to Norm rectangle points
def normalize_rectangle_points(json_path: str) -> bool:
    with open(json_path, "r") as f: #open the json file
        data = json.load(f)
    changed = False
    
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue #only process rectangles
            
        points = shape.get("points", [])
        if len(points) != 2:
            continue #not a valid rectangle
        try:
            (x0, y0), (x1, y1) = points
        except Exception: #in case points are not in expected format
            continue

        #Normalize points to top-left and bottom-right(new order)
        nx0, nx1 = sorted([x0, x1])
        ny0, ny1 = sorted([y0, y1])

        if (nx0, ny0) != (x0, y0) or (nx1, ny1) != (x1, y1):
            shape["points"] = [[nx0, ny0], [nx1, ny1]]
            changed = True
    if changed: 
        with open(json_path, "w") as f: #write back the changes
            json.dump(data, f, indent=2)
    return changed

def main(folder:str):
    json_files = glob.glob(os.path.join(folder, "*.json"))
    fixed = 0
    total = 0
    for json in json_files:
        total += 1
        if normalize_rectangle_points(json):
            fixed += 1
            print(f"Fixed rectangle points in: {json}")
    #Summary of changes in the files
    print(f"Folder: {folder}")
    print(f"Total JSON files processed: {total}")
    print(f"Files with fixed rectangles: {fixed}")

#Command line execution
if __name__ == "__main__": #only run when executed directly
    if len(sys.argv) !=2:
        print("Usage: python normrectangles.py <folder_with_json_files>")
        sys.exit(1)
    main(sys.argv[1]) #pass the folder containing json files as argument