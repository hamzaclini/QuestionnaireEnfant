import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile

# ---------- Your Original Functions (slightly modified to accept image directly) ----------
def show_image(img, caption="Image"):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_column_width=True)

def extract_red_areas(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if len(cnt) > 70]
    result = img.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    return filtered_contours, result

def divide_into_boxes(img, box_height_px, box_width_px, rows, cols, start_x=700, start_y=700):
    height, width, _ = img.shape
    boxes = []
    for i in range(rows):
        for j in range(cols):
            top_left = (start_x + j * box_width_px, start_y + i * box_height_px)
            bottom_right = (start_x + (j + 1) * box_width_px, start_y + (i + 1) * box_height_px)
            boxes.append((top_left, bottom_right))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    return boxes, img

def find_red_circle_coordinates(img, box_height_px, box_width_px, rows, cols, start_x=700, start_y=700):
    contours, contour_img = extract_red_areas(img.copy())
    boxes, boxed_img = divide_into_boxes(img.copy(), box_height_px, box_width_px, rows, cols, start_x, start_y)

    box_contour_counts = {i: [0] * cols for i in range(rows)}

    for contour in contours:
        for point in contour:
            cx, cy = point[0]
            for i, (top_left, bottom_right) in enumerate(boxes):
                x_min, y_min = top_left
                x_max, y_max = bottom_right
                if x_min <= cx < x_max and y_min <= cy < y_max:
                    row = i // cols
                    col = i % cols
                    box_contour_counts[row][col] += 1
                    cv2.circle(boxed_img, (cx, cy), 5, (0, 255, 255), -1)

    max_contours_boxes = []
    for row in range(rows):
        if sum(box_contour_counts[row]) == 0:
            max_contours_boxes.append((row + 1, None))
        else:
            max_contours_col = max(range(cols), key=lambda col: box_contour_counts[row][col])
            max_contours_boxes.append((row + 1, max_contours_col + 1))

    return max_contours_boxes, boxed_img

# ---------- Streamlit App ----------
st.title("Red Circle Detector with Editable Table")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Parameters
    box_height = st.number_input("Box Height (px)", value=72)
    box_width = st.number_input("Box Width (px)", value=94)
    rows = st.number_input("Number of Rows", value=16, step=1)
    cols = st.number_input("Number of Columns", value=4, step=1)
    start_x = st.number_input("Start X", value=1067)
    start_y = st.number_input("Start Y", value=932)

    # Process Image
    max_boxes, processed_img = find_red_circle_coordinates(
        image_bgr, box_height, box_width, rows, cols, start_x, start_y
    )

    show_image(processed_img, "Detected Red Circles and Boxes")

    # Create editable DataFrame
    df = pd.DataFrame(max_boxes, columns=["Question (Row)", "Selected Option (Column)"])
    edited_df = st.data_editor(df, num_rows="fixed", use_container_width=True)

    if st.button("Submit"):
        st.success("Answers submitted successfully!")
        st.dataframe(edited_df)

        # Optionally allow downloading
        csv = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="answers.csv", mime="text/csv")
