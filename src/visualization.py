import tkinter as tk
import cv2
import os
import sys
from detect.detect import Detector
from interpret.combine import combine
from interpret.convert import convert
from lib.frame_grabber import FrameGrabber
from PIL import Image, ImageTk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500
PADDING_X = CANVAS_WIDTH // 10
PADDING_Y = CANVAS_HEIGHT // 10
min_x = 0
min_y = 0
max_x = 0
max_y = 0

last_mtime = None
annotating = False
current_shape = None
original_image_size = None
active_index = 0
frames = []
raw_data = {}
combined_data = {}

def draw():
    global min_x, min_y, max_x, max_y

    if combined_data:
        markers = combined_data["markers"]
        houses = combined_data["houses"]
        paths = combined_data["paths"]
        min_x, min_y = markers[0]["position"] if len(markers) > 0 else (0,0)
        max_x, max_y = min_x, min_y
        for marker in markers[1:]:
            x,y = marker["position"]

            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        canvas.delete('all')

        draw_markers(markers)
        draw_houses(houses)
        draw_paths(paths)

def scale_point(point):
    x,y = point
    sx = (x - min_x) * (CANVAS_WIDTH - PADDING_X * 2) / (max_x - min_x) + PADDING_X
    sy = (y - min_y) * (CANVAS_HEIGHT - PADDING_Y * 2) / (max_y - min_y) + PADDING_Y
    return (sx, sy)

def draw_markers(markers):
    for marker in markers:
        pos = marker["position"]
        sx, sy = scale_point(pos)
        r = 5
        canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill='red')
        canvas.create_text(sx + 10, sy, text=marker["id"], anchor='w')

def draw_paths(paths):
    for path in paths:
        c1, c2, c3, c4 = path['points']
        canvas.create_polygon(*scale_point(c1), *scale_point(c2), *scale_point(c3), *scale_point(c4), outline="yellow", width=0, fill='yellow')

def draw_houses(houses):
    for house in houses:
        c1, c2, c3, c4 = house['points']
        cx = (c1[0] + c2[0] + c3[0] + c4[0]) / 4
        cy = (c1[1] + c2[1] + c3[1] + c4[1]) / 4
        r = 5
        #canvas.create_oval(scx - r, scy - r, scx + r, scy + r, fill='red')
        canvas.create_polygon(*scale_point(c1), *scale_point(c2), *scale_point(c3), *scale_point(c4), outline="blue", width=2, fill='')
        #canvas.create_text(scx + 10, scy, text=house['class'], anchor='w')

selected_shape_index = None

def update_sidebar():
    global raw_data

    sidebar_list.delete(0, tk.END)  
    houses = raw_data["areas"][active_index]["houses"]
    for house in houses:
        sidebar_list.insert(tk.END, house["class"])

    if selected_shape_index is not None and selected_shape_index < len(houses):
        sidebar_list.selection_set(selected_shape_index)

def on_shape_select(_event):
    global selected_shape_index, raw_data, shape_name_var
    houses = raw_data["areas"][active_index]["houses"]

    selection = sidebar_list.curselection()
    if selection:
        selected_shape_index = selection[0]
        print(selected_shape_index)
        shape_name_var.set(houses[selected_shape_index]["class"])
        update_annotations()

def update_shape_name(_event):
    global selected_shape_index, raw_data, shape_name_var
    houses = raw_data["areas"][active_index]["houses"]

    if selected_shape_index is not None and selected_shape_index < len(houses):
        new_name = shape_name_var.get()
        raw_data["areas"][active_index]["houses"][selected_shape_index]["class"] = new_name
        update_sidebar()

def delete_selected_shape():
    global selected_shape_index, raw_data
    if selected_shape_index is not None:
        del raw_data["areas"][active_index]["houses"][selected_shape_index]
        selected_shape_index = None
        update_sidebar()
        update_annotations()

def new_shape():
    global current_shape
    current_shape = []

def complete_shape():
    global current_shape, raw_data

    raw_data["areas"][active_index]["houses"].append({
        "points": current_shape,
        "class": "New Shape",
        "confidence": 100.0
    })
    
    current_shape = None

def on_feed_click(event):
    if current_shape is not None:
        x, y = event.x, event.y
        current_shape.append([x, y])
        
        r = 3
        feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                               fill='red', tags="annotation")
        
        if len(current_shape) > 1:
            x1, y1 = current_shape[-2]
            x2, y2 = current_shape[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill='blue', width=2, tags="annotation")
        
        if len(current_shape) >= 3:
            complete_shape_button.config(state=tk.NORMAL)
            x1, y1 = current_shape[0]
            x2, y2 = current_shape[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill='green', width=2, tags="annotation")

def prev_frame():
    global active_index
    if active_index == 0:
        active_index = len(grabbers) - 1
    else:
        active_index -= 1

def next_frame():
    global active_index
    if active_index == len(grabbers) - 1:
        active_index = 0
    else:
        active_index += 1

def update_feed_image():
    global frames, active_index
    try:
        frame = frames[active_index]
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            img_tk = ImageTk.PhotoImage(img_pil)
            
            feed_canvas.img_tk = img_tk  
            feed_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk, tags="background")
            feed_canvas.tag_lower("background")

            feed_canvas.config(width=img_pil.width, height=img_pil.height)
            
    except Exception as e:
        print("Error updating feed:", e)

def update_frames():
    global frames

    frames = []
    for grabber in grabbers:
        frame = grabber.read()
        frames.append(frame)

def update_detections():
    global raw_data, frames
    raw_data = detector.detect(frames)

def update_path_detections():
    global raw_data, frames
    paths = detector.detect_paths(frames)
    for i, area in enumerate(raw_data["areas"]):
        area["paths"] = paths[i]

def update_annotations(scope="all"):
    global active_index, raw_data

    if scope == "all":
        feed_canvas.delete("path")
        feed_canvas.delete("annotation")
    else:
        feed_canvas.delete(scope)

    area = raw_data["areas"][active_index]
    houses = area["houses"]
    markers = area["markers"]
    paths = area["paths"]

    for path in paths:
        points = path["points"]
        for i, point in enumerate(points):
            r = 3
            x,y = point
            if i > 0:
                prev_x, prev_y = points[i-1]
                feed_canvas.create_line(prev_x, prev_y, x, y, 
                                      fill="yellow", width=2, tags="path")
        if len(points) >= 3:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill="yellow", width=2, tags="path")

    for i, house in enumerate(houses):
        curr = i == selected_shape_index
        points = house["points"]
        for i, point in enumerate(points):
            r = 3
            x,y = point
            feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                                    fill='yellow' if curr else 'red', tags="annotation")
            if i > 0:
                prev_x, prev_y = points[i-1]
                feed_canvas.create_line(prev_x, prev_y, x, y, 
                                      fill='yellow' if curr else 'red', width=2, tags="annotation")
        if len(points) >= 3:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill='yellow' if curr else 'red', width=2, tags="annotation")

    for marker in markers:
        pos = marker["position"]
        x,y = pos
        r = 5
        feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                               fill='blue', tags="annotation")

update_counter = 0
def loop():
    global update_counter
    update_frames()
    update_feed_image()
    if (update_counter < 20):
        update_counter += 1
        update_detections()
        update_annotations()
    else:
        update_path_detections()
        update_sidebar()
        update_annotations("path")
    
    print("filler")
    root.after(100, loop)

def interpret():
    global combined_data
    converted = convert(raw_data)
    combined = combine(converted)
    combined_data = combined

    draw()

    #export
    root.after(1000, interpret)

def refresh_data():
    global last_mtime
    
    #try:
    #    mtime = os.path.getmtime(RAW_JSON)
    #    if last_mtime is None or mtime != last_mtime:
    #        last_mtime = mtime
    #        update_annotations()
    #        
    #except Exception as e:
    #    print("Error loading data:", e)

grabbers = []
for i in range(1, len(sys.argv)):
    grabber = FrameGrabber(int(sys.argv[i]))
    grabbers.append(grabber)
if len(sys.argv) == 1:
    grabbers.append(FrameGrabber())

detector = Detector(len(grabbers))

root = tk.Tk()
root.title("visualization")
root.geometry("1400x800")

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(right_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack(fill=tk.BOTH, expand=True, pady=10)

frame_buttons = tk.Frame(left_frame)
frame_buttons.pack(pady=10)

frame_left = tk.Button(frame_buttons, text="left", command=prev_frame)
frame_left.pack(fill=tk.X, padx=5, pady=5)
frame_right = tk.Button(frame_buttons, text="right", command=next_frame)
frame_right.pack(fill=tk.X, padx=5, pady=5)

feed_frame = tk.LabelFrame(left_frame, text="Video Feed")
feed_frame.pack(fill=tk.BOTH, expand=True, pady=10)

feed_canvas = tk.Canvas(feed_frame, width=640, height=480, bg='black')
feed_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
feed_canvas.bind("<Button-1>", on_feed_click)

sidebar_frame = tk.LabelFrame(right_frame, text="Annotation shapes", width=200)
sidebar_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

list_frame = tk.Frame(sidebar_frame)
list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

sidebar_list = tk.Listbox(list_frame)
sidebar_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(list_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
sidebar_list.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=sidebar_list.yview)
sidebar_list.bind('<<ListboxSelect>>', on_shape_select)

name_frame = tk.Frame(sidebar_frame)
name_frame.pack(fill=tk.X, padx=5, pady=5)

tk.Label(name_frame, text="Shape name:").pack(side=tk.LEFT)

shape_name_var = tk.StringVar()
name_entry = tk.Entry(name_frame, textvariable=shape_name_var)
name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
name_entry.bind("<KeyRelease>", update_shape_name)

button_frame_sidebar = tk.Frame(sidebar_frame)
button_frame_sidebar.pack(fill=tk.X, padx=5, pady=5)

delete_button = tk.Button(sidebar_frame, text="Delete selected", command=delete_selected_shape)
delete_button.pack(fill=tk.X, padx=5, pady=5)

button_frame = tk.Frame(left_frame)
button_frame.pack(pady=10)

new_shape_button = tk.Button(button_frame, text="New shape", command=new_shape)
new_shape_button.pack(side=tk.LEFT, padx=5)

complete_shape_button = tk.Button(button_frame, text="Complete shape", command=complete_shape, state=tk.DISABLED)
complete_shape_button.pack(side=tk.LEFT, padx=5)

status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

loop()
interpret()

root.mainloop()
