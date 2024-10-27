#!/usr/bin/env python3
"""
Dataset Generator for COCO and TFOD formats
-----------------------------------------
This script generates synthetic datasets in both COCO and TensorFlow Object Detection formats.
It creates dummy data that maintains consistency between both formats.

Requirements:
    pip install numpy
"""

import json
import csv
import random
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

class DatasetGenerator:
    def __init__(self, num_images=100, annotations_per_image=5, image_width=640, image_height=480):
        self.num_images = num_images
        self.annotations_per_image = annotations_per_image
        self.image_width = image_width
        self.image_height = image_height
        
        # Fixed categories for our synthetic dataset
        self.categories = [
            {"id": 1, "name": "person", "supercategory": "living"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "dog", "supercategory": "animal"},
            {"id": 4, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 5, "name": "chair", "supercategory": "furniture"}
        ]
        
        # Initialize random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

    def generate_bbox(self):
        """Generate a random bounding box that fits within image dimensions."""
        # Generate random width and height (between 10% and 30% of image dimensions)
        width = random.uniform(0.1 * self.image_width, 0.3 * self.image_width)
        height = random.uniform(0.1 * self.image_height, 0.3 * self.image_height)
        
        # Generate random x,y (ensuring bbox fits within image)
        x = random.uniform(0, self.image_width - width)
        y = random.uniform(0, self.image_height - height)
        
        return [x, y, width, height]  # COCO format [x,y,width,height]

    def coco_to_tfod_bbox(self, bbox):
        """Convert COCO format bbox [x,y,width,height] to TFOD format [ymin,xmin,ymax,xmax]."""
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        
        # Normalize coordinates for TFOD format
        ymin_norm = ymin / self.image_height
        xmin_norm = xmin / self.image_width
        ymax_norm = ymax / self.image_height
        xmax_norm = xmax / self.image_width
        
        return [ymin_norm, xmin_norm, ymax_norm, xmax_norm]

    def generate_coco_dataset(self):
        """Generate COCO format dataset."""
        images = []
        annotations = []
        ann_id = 1
        
        for img_id in range(1, self.num_images + 1):
            # Generate image entry
            image_name = f"image_{img_id:06d}.jpg"
            images.append({
                "id": img_id,
                "width": self.image_width,
                "height": self.image_height,
                "file_name": image_name,
                "license": 1,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Generate annotations for this image
            for _ in range(self.annotations_per_image):
                bbox = self.generate_bbox()
                category_id = random.choice(self.categories)["id"]
                
                # Calculate area (required for COCO format)
                area = bbox[2] * bbox[3]
                
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # Empty for dummy data
                })
                ann_id += 1
        
        # Construct COCO format dictionary
        coco_dict = {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Synthetic dataset for testing",
                "contributor": "Dataset Generator",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "dummy_license", "url": ""}],
            "images": images,
            "annotations": annotations,
            "categories": self.categories
        }
        
        return coco_dict

    def generate_tfod_csv(self, coco_dict):
        """Generate TensorFlow Object Detection CSV format dataset."""
        tfod_rows = []
        
        # Create category id to name mapping
        cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        
        # Create image id to filename mapping
        img_id_to_name = {img["id"]: img["file_name"] for img in coco_dict["images"]}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_dict["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Generate rows for each image
        for img_id, annotations in annotations_by_image.items():
            filename = img_id_to_name[img_id]
            
            # Convert all bboxes for this image
            for ann in annotations:
                tfod_bbox = self.coco_to_tfod_bbox(ann["bbox"])
                class_name = cat_id_to_name[ann["category_id"]]
                
                tfod_rows.append({
                    "filename": filename,
                    "width": self.image_width,
                    "height": self.image_height,
                    "class": class_name,
                    "xmin": tfod_bbox[1],
                    "ymin": tfod_bbox[0],
                    "xmax": tfod_bbox[3],
                    "ymax": tfod_bbox[2]
                })
        
        return tfod_rows

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic COCO and TFOD datasets')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images in the dataset')
    parser.add_argument('--annotations_per_image', type=int, default=5,
                        help='Number of annotations per image')
    parser.add_argument('--output_dir', type=str, default='generated_dataset',
                        help='Output directory for the generated files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = DatasetGenerator(
        num_images=args.num_images,
        annotations_per_image=args.annotations_per_image
    )
    
    # Generate COCO format dataset
    coco_dict = generator.generate_coco_dataset()
    
    # Save COCO format JSON
    coco_output = output_dir / 'coco_annotations.json'
    with open(coco_output, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    
    # Generate and save TFOD format CSV
    tfod_rows = generator.generate_tfod_csv(coco_dict)
    tfod_output = output_dir / 'tfod_annotations.csv'
    
    with open(tfod_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'width', 'height', 'class', 
                                             'xmin', 'ymin', 'xmax', 'ymax'])
        writer.writeheader()
        writer.writerows(tfod_rows)
    
    print(f"Generated datasets in {output_dir}:")
    print(f"- COCO format: {coco_output}")
    print(f"- TFOD format: {tfod_output}")
    print(f"\nDataset statistics:")
    print(f"- Number of images: {args.num_images}")
    print(f"- Annotations per image: {args.annotations_per_image}")
    print(f"- Total annotations: {args.num_images * args.annotations_per_image}")

if __name__ == "__main__":
    main()
