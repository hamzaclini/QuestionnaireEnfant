import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile

image_params = [
    {"box_height": 72, "box_width": 94, "rows": 16, "cols": 4, "start_x": 1067, "start_y": 932},
    {"box_height": 72, "box_width": 94, "rows": 26, "cols": 4, "start_x": 1064, "start_y": 186},
    {"box_height": 72, "box_width": 94, "rows": 26, "cols": 4, "start_x": 1064, "start_y": 186},
    {"box_height": 72, "box_width": 94, "rows": 22, "cols": 4, "start_x": 1064, "start_y": 186},
]

# ---------- Your Original Functions (slightly modified to accept image directly) ----------
def show_image(img, caption="Image"):
    #st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_column_width=True)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)

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
#st.set_page_config(layout="wide")

st.title("Questionnaire pour enfant")

uploaded_files = st.file_uploader(
    "Téléchargez 4 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 4:
    all_results = []
    question_counter = 1  # Start from Q1

    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Page {idx + 1}")

        # Grab parameters for this image
        params = image_params[idx]
        box_height = params["box_height"]
        box_width = params["box_width"]
        rows = params["rows"]
        cols = params["cols"]
        start_x = params["start_x"]
        start_y = params["start_y"]

        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        max_boxes, processed_img = find_red_circle_coordinates(
            image_bgr, box_height, box_width, rows, cols, start_x, start_y
        )

        # Assign global question numbers
        question_numbers = list(range(question_counter, question_counter + len(max_boxes)))
        question_counter += len(max_boxes)

        df = pd.DataFrame(max_boxes, columns=["_", "Reponse"])
        df["Question"] = question_numbers
        df = df[["Question", "Reponse"]]  # Reorder columns

        # Display results
        col1, col2 = st.columns([2, 1])
        with col1:
            show_image(processed_img, f"Detected Circles - Image {idx + 1}")
        with col2:
            #df = pd.DataFrame(max_boxes, columns=["Question", "Reponse"])
            edited_df = st.data_editor(df, num_rows="fixed", use_container_width=True, hide_index=True, key=f"editor_{idx}")
            all_results.append(edited_df)

    if st.button("Tout soumettre"):
        final_df = pd.concat(all_results, ignore_index=True)

        # Remove duplicates: keep the first response per question
        #final_df = final_df.drop_duplicates(subset=["Question"])

        # Sort questions
        final_df = final_df.sort_values(by="Question").reset_index(drop=True)

        # Map responses
        mapping = {1: 2, 2: 1, 3: 0, 4: "NSP"}
        final_df["Mapped"] = final_df["Reponse"].map(mapping)

        # Create wide format: questions as columns, one row for answers
        questions = [f"Q{q}" for q in final_df["Question"]]
        answers = final_df["Mapped"].tolist()

        display_df = pd.DataFrame([answers], columns=questions, index=["Answer"])

        st.success("Réponses traitées avec succès!")
        st.dataframe(display_df)

        # Optional: download
        #csv = display_df.to_csv().encode("utf-8")
        #st.download_button("Download Answers Table", data=csv, file_name="answers_table.csv", mime="text/csv")
        
        #final_df = pd.concat(all_results, keys=[f"Image {i+1}" for i in range(4)])
        #st.success("All answers submitted successfully!")
        #st.dataframe(final_df)

        #csv = final_df.to_csv().encode("utf-8")
        #st.download_button("Download All Answers as CSV", data=csv, file_name="all_answers.csv", mime="text/csv")
