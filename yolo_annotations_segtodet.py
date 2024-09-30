import os
import cv2
import numpy as np

def yolo_segmentation_to_bbox(label_path, img_width, img_height):
    """
    Convert YOLO segmentation labels (text format) into bounding box coordinates.
    
    Args:
        label_path (str): Path to the text file containing YOLO segmentation labels.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
    
    Returns:
        bbox_list (list): List of bounding boxes in (class_id, x_min, y_min, width, height) format.
    """
    bbox_list = []
    
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        elements = line.strip().split()        
        class_id = int(elements[0])        
        coords = list(map(float, elements[1:]))
        
        # Separate x and y coordinates and denormalize
        polygon_points = np.array([
            [coords[i] * img_width, coords[i + 1] * img_height] 
            for i in range(0, len(coords), 2)
        ], dtype=np.int32)

        x_min, y_min, width, height = cv2.boundingRect(polygon_points)
        bbox_list.append((class_id, x_min, y_min, width, height))
    
    return bbox_list

def process_images_and_annotations(image_folder, annotation_folder, output_folder):
    """
    Process images and their corresponding YOLO segmentation annotations, 
    and convert segmentation to bounding box annotations. Saves the new
    images with bounding boxes and corresponding label files.
    
    Args:
        image_folder (str): Path to the folder containing images.
        annotation_folder (str): Path to the folder containing YOLO segmentation annotations.
        output_folder (str): Path to save the resulting images with bounding boxes and label files.
    
    Returns:
        None
    """
    image_output_dir = os.path.join(output_folder, 'images')
    label_output_dir = os.path.join(output_folder, 'labels')
    images_with_annotations_dir = os.path.join(output_folder, 'images_with_annotations')

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(images_with_annotations_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        
        # Find the corresponding annotation file
        annotation_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found for {img_file}")
            continue
        
        bboxes = yolo_segmentation_to_bbox(annotation_path, img_width, img_height)
        
        # Save bounding boxes as new annotation in label output directory
        label_output_path = os.path.join(label_output_dir, annotation_file)
        with open(label_output_path, 'w') as label_file:
            for bbox in bboxes:
                class_id, x_min, y_min, width, height = bbox
                
                # Convert back to normalized YOLO format: x_center, y_center, width, height (normalized)
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                label_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
        
        for bbox in bboxes:
            class_id, x_min, y_min, width, height = bbox
            image_with_annotations = cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)
            cv2.putText(image_with_annotations, f"Class {class_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
               
        image_with_annotation_path = os.path.join(images_with_annotations_dir, img_file)
        cv2.imwrite(image_with_annotation_path, image_with_annotations)
        
        image_output_path = os.path.join(image_output_dir, img_file)
        cv2.imwrite(image_output_path, image)
        
        print(f"Processed and saved image: {image_output_path}")
        print(f"Saved label: {label_output_path}")

image_folder = 'path/to/your/image/folder'
annotation_folder = 'path/to/your/labels/folder'
output_folder = 'save_folder'

process_images_and_annotations(image_folder, annotation_folder, output_folder)
