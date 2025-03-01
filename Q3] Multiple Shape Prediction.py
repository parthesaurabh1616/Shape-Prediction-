#!/usr/bin/env python
# coding: utf-8

# # STEP 1: CREATE DATABASE AND PREPROCESS

# In[1]:


import cv2
import numpy as np
import os

def detect_shapes(image):
    # Step 2.1: Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2.2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3.1: Threshold the image to get a binary image
    ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    # Step 4.1: Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Step 5.1: Compute the perimeter of the contour
        perimeter = cv2.arcLength(cnt, True)
        
        # Step 5.2: Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        shape = "unidentified"
        
        # Step 5.3.1: Identify a triangle (3 vertices)
        if len(approx) == 3:
            shape = "triangle"
        # Step 5.3.2: Identify quadrilaterals and further classify as square or rectangle (4 vertices)
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        # Step 5.3.3: For shapes with more than 4 vertices, check if it is a circle or polygon
        elif len(approx) > 4:
            area = cv2.contourArea(cnt)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            if abs(area - circle_area) < 0.2 * circle_area:
                shape = "circle"
            else:
                shape = "polygon"
        
        # Step 5.3.4: Optionally, fit an ellipse to detect ellipses (if enough points)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            if abs(axes[0] - axes[1]) > 10:  # Threshold to differentiate from a circle
                shape = "ellipse"
        
        # Step 6.1: Draw the contour and the shape name on the image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.putText(image, shape, (cX - 20, cY - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


# # STEP 2: DETECTION AND CLASSIFICATION

# In[2]:


def create_shape_database(output_folder):
    # Step 1.1: Check if folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shapes = ["circle", "square", "rectangle", "triangle", "ellipse", "polygon"]
    
    for shape in shapes:
        # Step 1.2: Create a blank white image
        img = np.ones((400, 400, 3), dtype="uint8") * 255
        
        # Draw each shape based on its type
        if shape == "circle":
            cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)
        elif shape == "square":
            cv2.rectangle(img, (150, 150), (250, 250), (0, 255, 0), -1)
        elif shape == "rectangle":
            cv2.rectangle(img, (100, 150), (300, 250), (255, 0, 0), -1)
        elif shape == "triangle":
            pts = np.array([[200, 100], [100, 300], [300, 300]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 255, 255))
        elif shape == "ellipse":
            cv2.ellipse(img, (200, 200), (100, 50), 0, 0, 360, (255, 0, 255), -1)
        elif shape == "polygon":
            pts = np.array([[100, 150], [200, 100], [300, 150], [350, 250],
                            [300, 350], [200, 400], [100, 350], [50, 250]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 165, 255))
        
        # Save the generated image in the database folder
        filename = os.path.join(output_folder, f"{shape}.png")
        cv2.imwrite(filename, img)


# # STEP 3: LABELLING

# In[3]:


def process_database(input_folder, output_folder):
    # Step 6.2: Check if results folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Step 2.1: Process each image file in the database folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            # Step 5 & 6: Detect shapes and annotate the image
            processed_img = detect_shapes(img)
            # Save the processed image
            output_path = os.path.join(output_folder, "processed_" + filename)
            cv2.imwrite(output_path, processed_img)


# # STEP 4: SAVE RESULT

# In[4]:


if __name__ == "__main__":
    # Step 1: Create a folder for the shape database and generate images
    db_folder = "shape_database"
    create_shape_database(db_folder)
    
    # Step 6: Process the images to detect and label shapes
    results_folder = "results"
    process_database(db_folder, results_folder)
    
    # Step 7: Inform the user to package the folders along with the code into a ZIP file for submission.
    print("Shape database images have been created and processed. Check the 'shape_database' and 'results' folders.")


# In[ ]:




