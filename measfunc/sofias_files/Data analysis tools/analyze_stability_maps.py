import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.backend_bases 
import json 

def pair_points_to_lines(points):
    """
    points = [(x1,y1), (x2,y2), (x3,y3), ...]
    lines = [[(x1,x2), (y1,y2)], [(x3,x4), (y3,y4)]]
    """
    if (len(points)%2 != 0): 
        raise Warning("Number of points is not even. Skipping the last point")
    lines = [[points[i_p], points[i_p+1]] 
              for i_p in range(0,len(points),2)]
    return lines 
    
def click_points_to_save_lines(file_path:str, file_name:str, timeout:float=180):
    points = plt.ginput(n=-1, timeout=timeout, mouse_stop=matplotlib.backend_bases.MouseButton.RIGHT) # n = negative -> click until input is terminated with right-click
    lines = pair_points_to_lines(points)
    try:
        with open(os.path.join(file_path, file_name), 'w') as json_file:
            json.dump(lines, json_file)
        print("Saved lines successfully!")
        return lines 
    except: 
        return lines 