import tkinter as tk
import cv2
import json
import os
from detect import Detector
from frame_grabber import FrameGrabber
from PIL import Image, ImageTk

JSON_FILE = "data.json"

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500
SCALE_X = CANVAS_WIDTH
SCALE_Y = CANVAS_HEIGHT

last_mtime = None
annotating = False
current_shape = []
shapes = []
original_image_size = None

def scale_point(x, y):
    sx = x * SCALE_X
    sy = (1 - y) * SCALE_Y
    return sx, sy

def draw_area(canvas, area):
    points = []
    for x, y, _ in area:
        sx, sy = scale_point(x, y)
        points.extend([sx, sy])
    canvas.create_polygon(points, outline='blue', fill='', width=2)

def draw_houses(canvas, houses):
    for house in houses:
        cx, cy, _ = house['center']
        c1, c2, c3, c4 = house['corners']
        scx, scy = scale_point(cx, cy)
        r = 5
        canvas.create_oval(scx - r, scy - r, scx + r, scy + r, fill='red')
        canvas.create_polygon(*c1, *c2, *c3, *c4, outline="blue", width=2)
        canvas.create_text(scx + 10, scy, text=house['class'], anchor='w')

def load_data():
    with open(JSON_FILE, "r") as f:
        return json.load(f)


selected_shape_index = None
shape_name_var = None

def update_sidebar():
    sidebar_list.delete(0, tk.END)  
    for i, shape in enumerate(shapes):
        name = shape.get('name', f"Shape {i+1}")
        sidebar_list.insert(tk.END, f"{name}")
    
    if selected_shape_index is not None and selected_shape_index < len(shapes):
        sidebar_list.selection_set(selected_shape_index)

def on_shape_select(_event):
    global selected_shape_index, shape_name_var
    selection = sidebar_list.curselection()
    if selection:
        selected_shape_index = selection[0]
        redraw_annotations()
        if selected_shape_index < len(shapes):
            shape = shapes[selected_shape_index]
            name = shape.get('name', f"Shape {selected_shape_index+1}")
            shape_name_var.set(name)
        status_bar.config(text=f"Selected: {name}")

def update_shape_name(_event):
    global selected_shape_index
    if selected_shape_index is not None and selected_shape_index < len(shapes):
        new_name = shape_name_var.get()
        shapes[selected_shape_index]['name'] = new_name
        update_sidebar()
        status_bar.config(text=f"Renamed to: {new_name}")

def delete_selected_shape():
    global selected_shape_index, shapes
    if selected_shape_index is not None and shapes:
        del shapes[selected_shape_index]
        selected_shape_index = None
        update_sidebar()
        redraw_annotations()
        status_bar.config(text="Shape deleted")

def start_annotation():
    global annotating, current_shape
    annotating = True
    current_shape = []
    annotate_button.config(state=tk.DISABLED)
    done_button.config(state=tk.NORMAL)
    new_shape_button.config(state=tk.NORMAL)
    status_bar.config(text="Annotation started")
    update_sidebar()

def new_shape():
    global current_shape
    if current_shape:
        shape_num = len(shapes) + 1
        shapes.append({
            "points": current_shape.copy(),
            "name": f"Shape {shape_num}"  
        })
        current_shape = []
        status_bar.config(text="Started a new shape")
    update_sidebar()

def done_annotation():
    global annotating, current_shape, shapes
    annotating = False
    
    if current_shape:
        shape_num = len(shapes) + 1
        shapes.append({
            "points": current_shape.copy(),
            "name": f"Shape {shape_num}"  
        })
    
    annotate_button.config(state=tk.NORMAL)
    done_button.config(state=tk.DISABLED)
    new_shape_button.config(state=tk.DISABLED)
    update_sidebar()
    
    status_bar.config(text="Annotation completed")
    print("\nAll annotations with names:")
    for i, shape in enumerate(shapes, 1):
        name = shape.get('name', f"Shape {i}")
        print(f"{name}")
        for j, (x, y) in enumerate(shape['points'], 1):
            print(f"  Point {j}: ({x}, {y})")
    detector.update_shapes(shapes)

def clear_annotations():
    global shapes, current_shape
    shapes = []
    current_shape = []
    feed_canvas.delete("annotation")
    status_bar.config(text="All annotations cleared")
    update_sidebar()
    print("All annotations cleared")

def on_feed_click(event):
    if annotating:
        x, y = event.x, event.y
        current_shape.append((x, y))
        status_bar.config(text=f"Added point: ({x}, {y})")
        
        r = 3
        feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                               fill='red', tags="annotation")
        
        if len(current_shape) > 1:
            x1, y1 = current_shape[-2]
            x2, y2 = current_shape[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill='blue', width=2, tags="annotation")
        
        if len(current_shape) >= 3:
            x1, y1 = current_shape[0]
            x2, y2 = current_shape[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill='green', width=2, tags="annotation")

def update_feed_image():
    global original_image_size
    
    try:
        frame = grabber.read()
        if frame is not None:
            if original_image_size is None:
                original_image_size = (frame.shape[1], frame.shape[0])

            
            drawn_frame = detector.detect(frame)
            
            frame_rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            img_tk = ImageTk.PhotoImage(img_pil)
            
            feed_canvas.img_tk = img_tk  
            feed_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk, tags="background")
            
            feed_canvas.config(width=img_pil.width, height=img_pil.height)
            
            redraw_annotations()
            
    except Exception as e:
        print("Error updating feed:", e)
    
    root.after(100, update_feed_image)

def redraw_annotations():
    feed_canvas.delete("annotation")
    
    for idx, shape in enumerate(shapes):
        points = shape["points"]
        point_color = 'orange' if idx == selected_shape_index else 'red'
        line_color = 'yellow' if idx == selected_shape_index else 'blue'
        close_color = 'gold' if idx == selected_shape_index else 'green'
        
        for i, (x, y) in enumerate(points):
            r = 3
            feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                                  fill=point_color, tags="annotation")
            
            if i > 0:
                prev_x, prev_y = points[i-1]
                feed_canvas.create_line(prev_x, prev_y, x, y, 
                                      fill=line_color, width=2, tags="annotation")
        
        if len(points) >= 3:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            feed_canvas.create_line(x1, y1, x2, y2, 
                                  fill=close_color, width=2, tags="annotation")
    
    for i, (x, y) in enumerate(current_shape):
        r = 3
        feed_canvas.create_oval(x - r, y - r, x + r, y + r, 
                              fill='red', tags="annotation")
        if i > 0:
            prev_x, prev_y = current_shape[i-1]
            feed_canvas.create_line(prev_x, prev_y, x, y, 
                                  fill='blue', width=2, tags="annotation")

def refresh_data():
    global last_mtime
    
    try:
        mtime = os.path.getmtime(JSON_FILE)
        if last_mtime is None or mtime != last_mtime:
            last_mtime = mtime
            data = load_data()
            canvas.delete("all")
            draw_area(canvas, data['area'])
            draw_houses(canvas, data['houses'])
            
    except Exception as e:
        print("Error loading data:", e)
    
    root.after(1000, refresh_data)

url = "http://192.168.0.19:4747/video"
grabber = FrameGrabber(url)
detector = Detector()

root = tk.Tk()
root.title("visualization")
root.geometry("1400x800")

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(right_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack(fill=tk.BOTH, expand=True, pady=10)

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

annotate_button = tk.Button(button_frame, text="Start annotation", command=start_annotation)
annotate_button.pack(side=tk.LEFT, padx=5)

new_shape_button = tk.Button(button_frame, text="New shape", command=new_shape, state=tk.DISABLED)
new_shape_button.pack(side=tk.LEFT, padx=5)

done_button = tk.Button(button_frame, text="Done annotation", command=done_annotation, state=tk.DISABLED)
done_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(button_frame, text="Clear annotations", command=clear_annotations)
clear_button.pack(side=tk.LEFT, padx=5)

status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

update_feed_image()
refresh_data()

root.mainloop()
