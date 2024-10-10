import streamlit as st 
import datetime
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import glob  

# Define the CSV directory
csv_dir = 'Face_recog_Attendance/'

# Function to create Attendance.csv if it does not exist
def make_csv():
    # Create the Attendance.csv if it does not exist
    attendance_file = os.path.join(csv_dir, 'Attendance.csv')
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as attendance_csv:
            attendance_writer = csv.writer(attendance_csv)
            attendance_writer.writerow(["Class", "Name", "Date"])  # Headers for attendance file

# Navigation
# Set the page title
st.sidebar.title("Attendance System")
nav = st.sidebar.radio("Navigate", ["Mark Attendance", "Students", "Dashboard"])

# Create the Attendance.csv file as soon as the program runs
make_csv()  # Ensure Attendance.csv is created at the start

# Current date
st.sidebar.write(f"Date: {datetime.now().strftime('%B %d, %Y')}")


# Function to check for duplicate user ID or name
def is_duplicate(phone, name, csv_file='Face_recog_Attendance/StudentDetails.csv'):
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # Ensure the row is not empty
                    if str(row[0]) == str(phone) or row[1] == name:
                        return True
    return False

# Function to compare the captured face with existing faces
def is_face_same(new_face_img, existing_faces, existing_ids):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists('Face_recog_Attendance/Trainer.yml'):
        return None
    
    recognizer.read('Face_recog_Attendance/Trainer.yml')

    # Resize the captured image to match the size used in training (optional)
    new_face_img = cv2.resize(new_face_img, (200, 200))

    for face, number in zip(existing_faces, existing_ids):
        conf = recognizer.predict(new_face_img)[1]
        if conf < 60:  # Experiment with a slightly higher threshold
            return number
    return None

# Define CSV file path globally or pass it as a parameter
csv_file = 'Face_recog_Attendance/StudentDetails.csv'

def capture_images(name, number, path='Face_recog_Attendance/TrainingImages/'):
    # Check duplicates in CSV
    if is_duplicate(number, name):
        st.error(f"Number {number} or name '{name}' already exists in CSV.")
        return

    # Create directory for images if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    cam = cv2.VideoCapture(0)
    sample_num = 0
    detector = cv2.CascadeClassifier('Face_recog_Attendance/haarcascade_frontalface_default.xml')

    # Load existing faces and their IDs for comparison
    existing_faces, existing_ids = get_images_and_labels(path)

    # Initialize the image count to 0
    image_count = 0

    # Placeholder to display the number of images captured
    image_count_placeholder = st.empty()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_num += 1
            new_face_img = gray[y:y + h, x:x + w]

            matched_number = is_face_same(new_face_img, existing_faces, existing_ids)
            if matched_number is not None:
                matched_name = get_name_by_id(matched_number)  # Get the name of the matched user
                st.error(f"Face already exists for {matched_name}. Deleting all captured images for {name}.")
                
                # Delete all images for the new student
                delete_existing_images(name, number, path)

                cam.release()
                cv2.destroyAllWindows()
                return

            cv2.imwrite(f"{path}{name}.{number}.{sample_num}.jpg", new_face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Increment image count and update display
            image_count += 1
            image_count_placeholder.text(f"Images Captured: {image_count}")

            # Overlay image count on the frame
            cv2.putText(img, f'Images Captured: {image_count}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Capture Image', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sample_num >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Create directory for CSV files if it does not exist
    csv_dir = 'Face_recog_Attendance/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Save user details to StudentDetails.csv
    student_details_file = os.path.join(csv_dir, 'StudentDetails.csv')
    if not os.path.exists(student_details_file):
        # Create the CSV file with headers
        with open(student_details_file, 'w', newline='') as file:
            writer = csv.writer(file)
    try:
        with open(student_details_file, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([number, name, student_class])  # Replace 'student_class' with actual class
            st.success(f"Student {name} with number {number} added successfully!")  # Success message
            
            # Create the Attendance.csv if it does not exist
            attendance_file = os.path.join(csv_dir, 'Attendance.csv')
            if not os.path.exists(attendance_file):
                with open(attendance_file, 'w', newline='') as attendance_csv:
                    attendance_writer = csv.writer(attendance_csv)
                    attendance_writer.writerow(["Class", "Name", "Date"])  # Headers for attendance file

    except Exception as e:
        st.error(f"Error appending student to CSV: {e}")

    image_count_placeholder.empty()  # Clear the image count text


# Function to train the recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('Face_recog_Attendance/haarcascade_frontalface_default.xml')
    faces, ids = get_images_and_labels('Face_recog_Attendance/TrainingImages')
    recognizer.train(faces, np.array(ids))
    recognizer.save('Face_recog_Attendance/Trainer.yml')
    

# Helper function to get face images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        number = int(os.path.split(image_path)[-1].split(".")[1])
        face_samples.append(img_np)
        ids.append(number)
    return face_samples, ids

# Function to get the name by user ID
def get_name_by_id(number, csv_file='Face_recog_Attendance/StudentDetails.csv'):
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Check if the row has at least 2 columns (for Number and Name)
                if len(row) >= 2:
                    if str(row[0]) == str(number):
                        return row[1]  # Return the name corresponding to the user ID
    return None  # Return None if user ID is not found


# Function to delete existing images for the user
def delete_existing_images(name, phone, csv_file='Face_recog_Attendance/StudentDetails.csv'):
    # Generate the pattern to search for images for the new student
    file_pattern = f"{csv_file}{name}.{phone}.*.jpg"
    for file_path in glob.glob(file_pattern):
        if os.path.exists(file_path):
            os.remove(file_path)
            st.info(f"Deleted existing image: {file_path}")

# Function to display student details and a bar chart  
def display_student_details(search_name=None, search_class=None):
    filename = 'Face_recog_Attendance/StudentDetails.csv'

    if os.path.isfile(filename):
        # Read the CSV file
        df = pd.read_csv(filename, names=["Number", "Name", "Class"])

        # Filter DataFrame by search name and class if provided
        if search_name:
            df = df[df['Name'].str.contains(search_name, case=False, na=False)]
        if search_class:
            df = df[df['Class'].str.contains(search_class, case=False, na=False)]
        st.subheader("Student details")   
        # Display the DataFrame as HTML without the index
        if not df.empty:
            # Reset index and add a serial number column
            df.reset_index(drop=True, inplace=True)
            df.index += 1  # Start serial numbers from 1
            df['Sr No'] = df.index

            # Reorder columns to show 'Sr No', 'Number', 'Name', 'Class'
            columns_to_show = ['Sr No', 'Name', 'Number', 'Class']

            # Create custom CSS for wide tables
            st.markdown(
                """
                <style>
                .dataframe {
                width: 100% !important;
                }
                th {
                    text-align: center !important;
                }
                td {
                    text-align: left;  /* Keeps the table data left-aligned */
                }
                .dataframe {
                    width: 100%;  /* Adjust table width */
                    table-layout: auto;  /* Make table columns flexible */
                    border-collapse: collapse;  /* Merge table borders */
                }
                .dataframe td, .dataframe th {
                    padding: 10px;  /* Increase padding for better readability */
                    text-align: left;  /* Align text to the left */
                    border: 1px solid #ddd;  /* Add border to cells */
                }
                    /* Column-specific widths */
                .dataframe th:nth-child(1), .dataframe td:nth-child(1) {
                    width: 60px; /* Width for Sr No */
                }
                .dataframe th:nth-child(2), .dataframe td:nth-child(2) {
                    width: 300px; /* Width for Name */
                }
                .dataframe th:nth-child(3), .dataframe td:nth-child(3) {
                    width: 180px; /* Width for Number */
                }
                .dataframe th:nth-child(4), .dataframe td:nth-child(4) {
                    width: 180px; /* Width for Class */
                }
                
                </style>
                """, 
                unsafe_allow_html=True
            )
            # Display the DataFrame as HTML without the index
            st.write(df[columns_to_show].to_html(index=False), unsafe_allow_html=True)
        else:
            st.error("Student details file not found.")

# Function to create a bar chart of class-wise total students 
def bar_class_wise_totalstudent():
    # Load the student details from the CSV file
    filename = 'Face_recog_Attendance/StudentDetails.csv'
    
    if os.path.isfile(filename):
        # Read the CSV file
        df = pd.read_csv(filename, names=["Number", "Name", "Class"])
        
        # Create a bar chart of class-wise student count
        class_counts = df['Class'].value_counts()
        st.subheader("Class-wise Student Count")
        # Plotting the bar chart
        plt.figure(figsize=(10, 5))
        bars = plt.bar(class_counts.index, class_counts.values, color='skyblue')
        plt.title('Number of Students in Each Class')
        plt.xlabel('Class')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        
        # Annotate each bar with the count of students
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                     ha='center', va='bottom')  # va='bottom' to position text above the bar
        
        # Display the plot in Streamlit
        st.pyplot(plt)
    else:
        st.error("Student details file not found.")

# Function to delete a student's details from the CSV file
def delete_student_from_csv(name, csv_file='Face_recog_Attendance/StudentDetails.csv'):
    if not os.path.exists(csv_file):
        st.warning("CSV file does not exist.")
        return False  # Return False if file does not exist

    temp_file = 'Face_recog_Attendance/StudentDetails_temp.csv'
    student_found = False  # Flag to check if student is found
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Write all rows except the one with the matching name to a temporary file
        with open(temp_file, 'w', newline='') as temp_file_obj:
            writer = csv.writer(temp_file_obj)
            for row in rows:
                if row:
                    if row[1] == name:  # Assuming name is in the second column
                        student_found = True  # Student is found
                    else:
                        writer.writerow(row)

        if student_found:
            # Replace the original file with the temp file
            os.replace(temp_file, csv_file)
            return True  # Return True if student was deleted
        else:
            os.remove(temp_file)  # Remove the temp file if no student was found
            return False  # Return False if no student found
    except Exception as e:
        st.error(f"Error deleting student from CSV: {e}")
        return False  # Return False on error

# Function to delete student's attendance and images
def delete_student_data(name, retrain=True):
    # Path for attendance records
    attendance_file = 'Face_recog_Attendance/Attendance.csv'
    temp_file = 'Face_recog_Attendance/Attendance_temp.csv'
    
    # Delete attendance records
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        with open(temp_file, 'w', newline='') as temp_file_obj:
            writer = csv.writer(temp_file_obj)
            for row in rows:
                if row and row[1] != name:  # Assuming name is in the second column
                    writer.writerow(row)
        os.replace(temp_file, attendance_file)
    else:
        st.warning(f"No attendance records found for '{name}'.")

    # Path for training images
    image_dir = "Face_recog_Attendance/TrainingImages/"

    # Delete training images
    if os.path.exists(image_dir):
        for image_file in os.listdir(image_dir):
            if image_file.startswith(f"{name}."):
                os.remove(os.path.join(image_dir, image_file))
    # If retraining is required after deleting the student's images
    if retrain:
        retrain_model()

# Function to retrain the model after deleting a student
def retrain_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels('Face_recog_Attendance/TrainingImages/')
    
    # Check if we have enough faces for training
    if len(faces) < 2:  # Adjust the number based on your requirements
        st.warning("Not enough training data to retrain the model. At least 2 samples are required.")
        return  # Early exit if not enough samples
    
    recognizer.train(faces, np.array(ids))
    recognizer.write('Face_recog_Attendance/Trainer.yml')

# Function to get unique classes from the CSV
def get_unique_classes(csv_file='Face_recog_Attendance/StudentDetails.csv'):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, names=["Number", "Name", "Class"])
        return df['Class'].unique().tolist()
    return []

# Function to get unique student names from the CSV
def get_unique_names(csv_file='Face_recog_Attendance/StudentDetails.csv'):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, names=["Number", "Name", "Class"])
        return df['Name'].unique().tolist()
    return []  # Return an empty list if file doesn't exist


# Streamlit app for managing students
if nav == "Students":
    st.header("Add New Student")
    make_csv()
    
    # Input fields for student details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("Name")
    
    with col2:
        number = st.text_input("Number")

    with col3:
        student_class = st.text_input("Class")

    # Button to add a new student
    if st.button("Add Student"):
        st.warning("Press 'q' to stop the camera.")
        if name and number:
            # Capture images and train recognizer
            capture_images(name, number)  # Function to capture face images
            train_recognizer()  # Function to train the face recognizer
            # Clear input fields
            st.session_state[name] = ''
            st.session_state[number] = ''
            st.session_state[student_class] = ''
        else:
            st.error("Please fill in all fields.")

    # Search functionality
    st.subheader("Search Student")
    
    # Input fields for search functionality
    col4, col5 = st.columns(2)
    
    with col4:
         # Input for searching a specific student by name
        unique_names = get_unique_names()
        search_name = st.selectbox("Enter Name to search:", options=[""] + unique_names)
        
    with col5:
        # Get unique classes for the selectbox
        unique_classes = get_unique_classes()
        search_class = st.selectbox("Select Class to search:", options=[""] + unique_classes)

    # Delete button to remove student record
    if st.button("Delete Student"):
        if search_name:
            
            # Check if student exists before deleting
            student_deleted = delete_student_from_csv(search_name)
            if student_deleted:
                delete_student_data(search_name)
                st.success(f"Student '{search_name}' deleted successfully.")
            else:
                st.warning(f"No student found with the name '{search_name}' to delete.")
        else:   
            # Automatically show all student details if no search
            display_student_details()
            st.error("Please provide a student name to delete.")

    # Display student details at the end
    display_student_details(search_name, search_class)
    # Call the function to create the bar chart
    bar_class_wise_totalstudent()


##################################################################################

marked_students = set()  

# Function to mark attendance in CSV
def mark_attendance(number, name):
    date = datetime.now().strftime('%Y-%m-%d')  # Current date
    filename = 'Face_recog_Attendance/Attendance.csv'

    # Read StudentDetails.csv to get the class
    student_details_file = 'Face_recog_Attendance/StudentDetails.csv'
    if os.path.isfile(student_details_file):
        student_df = pd.read_csv(student_details_file, names=["Number", "Name", "Class"])
        student_row = student_df[student_df['Number'] == number]
        if not student_row.empty:
            student_class = student_row['Class'].values[0]
        else:
            st.error(f"Class information for {name} is not available.")
            return
    else:
        st.error("Student details file not found.")
        return

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the attendance file exists
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

        # Check if the student already exists in the file
        student_exists = df[(df['Name'] == name) & (df['Class'] == student_class)]
        
        if not student_exists.empty:
            # Student exists, append the new date to the existing dates
            existing_dates = student_exists['Date'].values[0]
            
            # If the current date is not already in the record, prepend it to the existing dates
            if date not in existing_dates.split(', '):
                updated_dates = date + ', ' + existing_dates  # Prepend today's date
                df.loc[(df['Name'] == name) & (df['Class'] == student_class), 'Date'] = updated_dates
                st.success(f"Attendance updated for {name} (Class: {student_class}) with new date: {date}.")
            else:
                # Check if the student has already been marked today
                    if name not in marked_students:
                        st.warning(f"Attendance already marked for {name} today.")
                        marked_students.add(name)
                        
        else:
            # Student doesn't exist, add a new row
            new_record = pd.DataFrame({'Class': [student_class], 'Name': [name], 'Date': [date]})
            df = pd.concat([df, new_record], ignore_index=True)
            st.success(f"Attendance marked for {name} (Class: {student_class}) on {date}.")
    else:
        # File doesn't exist, create a new file with the attendance record
        df = pd.DataFrame([{'Class': student_class, 'Name': name, 'Date': date}])
        st.success(f"Attendance marked for {name} (Class: {student_class}) on {date}.")

    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)

    # Provide a message if the attendance CSV is not found
    if not os.path.isfile(filename):
        st.warning("Attendance file not found. Please register students first.")

# Function to track and recognize faces
def track_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Face_recog_Attendance/Trainer.yml')
    face_cascade = cv2.CascadeClassifier('Face_recog_Attendance/haarcascade_frontalface_default.xml')
    
    cam = cv2.VideoCapture(0)
    df = pd.read_csv('Face_recog_Attendance/StudentDetails.csv', names=["Number", "Name", "Class"])
    
    frame_count = 0  # Frame counter for skipping frames
    # Initialize a set to keep track of unrecognized phone numbers
    unrecognized_number = set()
    # Streamlit display for camera feed
    FRAME_WINDOW = st.image([])

    while True:
        ret, img = cam.read()
        frame_count += 1
        
        if not ret:
            print("Failed to capture image")
            break

        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            number, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                # Check if the number exists in the DataFrame
                if not df.loc[df['Number'] == number].empty:
                    name = df.loc[df['Number'] == number]['Name'].values[0]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
                    # Mark attendance in both CSV and SQL
                    mark_attendance(number, name)
                else:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    # Only show warning once for each unrecognized phone
                    if number not in unrecognized_number:
                        st.warning(f"Unrecognized face with phone: {number}")
                        unrecognized_number.add(number)  # Add phone to the set
            else:
                # Handle unknown number case
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Unknown", (x + 6, y + h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Track Attendance', img)
        # Convert the image to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img_rgb)

        if cv2.waitKey(1) == ord('q') or st.session_state.get('stop_camera', False):
            break

    cam.release()
    cv2.destroyAllWindows()


# Function to load and use the face recognizer
def load_trainer_yml(trainer_file='Face_recog_project/Trainer.yml'):
    # Check if the Trainer.yml file exists
    if not os.path.exists(trainer_file):
        return None
    try:
        # Initialize the recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Read the Trainer.yml file
        recognizer.read(trainer_file)
        return recognizer
    except Exception as e:
        return None
    
    # Streamlit UI for 'Mark Attendance'
if nav == "Mark Attendance":
    st.header("Mark Attendance")

    # Load the CSV file (make sure to replace with your actual file path)
    df = pd.read_csv('Face_recog_Attendance/Attendance.csv')
    
    # Get today's date in the same format as in your CSV (e.g., 'YYYY-MM-DD')
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Split the Date column into lists of individual dates (in case there are multiple dates for each student)
    df['Date'] = df['Date'].astype(str).apply(lambda x: x.split(', '))

    # Check if today's date is in the list of dates for each student
    present_today = df[df['Date'].apply(lambda dates: today_date in dates)]

    # Get the total number of students present today
    total_present = present_today.shape[0]

    # Display the total present today in Streamlit
    st.metric("Total Students Present Today", total_present)

    # Create columns for Start and Stop buttons
    col1, col2 = st.columns(2)

    # Stop the camera when "Stop Camera" is pressed
    with col2:
        if st.button("Stop Camera"):
            st.session_state['stop_camera'] = True

# Start the camera when "Start Camera" is pressed
    with col1:
        if st.button("Start Camera"):
            # Check if the StudentDetails.csv file exists to determine if students are registered
            student_details_file = 'Face_recog_Attendance/StudentDetails.csv'
            if not os.path.isfile(student_details_file):
                st.warning("No students registered. Please register students first.")
            else:
                recognizer = load_trainer_yml()  # Load the trainer file
                if recognizer is None:
                    st.error("No student is registered. Please register a face before starting the camera.")
                else:
                    st.session_state['stop_camera'] = False  # Reset stop flag to False

    # Run track_attendance only if the camera is supposed to be running
    if not st.session_state.get('stop_camera', True):
        track_attendance()  # Start tracking attendance


##################################################################################


def display_attendance_records(month, year, search_class): 
    filename = 'Face_recog_Attendance/Attendance.csv'

    if not os.path.isfile(filename):
        st.error("Attendance file not found.")
        return

    try:
        df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        st.error("The attendance file is empty.")
        return
    except pd.errors.ParserError:
        st.error("Error parsing the attendance file. Please check the file format.")
        return

    if 'Date' not in df.columns:
        st.error("The 'Date' column is missing from the attendance file.")
        return

    # Convert the 'Date' column to string
    df['Date'] = df['Date'].astype(str)

    # Split the dates and filter by the selected month and year
    df['Date'] = df['Date'].apply(lambda x: x.split(', '))

    # Explode the 'Date' column so that each date is in a separate row
    df = df.explode('Date')

    # Convert the 'Date' column to datetime for filtering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter by selected month and year
    filtered_df = df[(df['Date'].dt.month == month) & (df['Date'].dt.year == year)]

    # Filter by class if provided
    if search_class:
        if 'Class' in df.columns:
            filtered_df = filtered_df[filtered_df['Class'].str.contains(search_class, case=False, na=False)]
        else:
            st.warning("The 'Class' column is not found in the file.")
            return

    if filtered_df.empty:
        st.warning("No records found for the selected criteria.")
        return
    st.subheader("Attendance")
    # Count attendance occurrences
    attendance_count = filtered_df.groupby(['Name', 'Class'])['Date'].count().reset_index(name='Total')

    # Concatenate the dates back together for display
    attendance_dates = filtered_df.groupby(['Name', 'Class'])['Date'].apply(lambda x: ', '.join(x.dt.strftime('%d'))).reset_index(name='Date')

    # Merge attendance count and dates
    filtered_df = pd.merge(attendance_count, attendance_dates, on=['Name', 'Class'])

    # Reset index and add serial number
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.index += 1
    filtered_df['Sr No'] = filtered_df.index

    # Custom CSS for table display
    st.markdown(
        """
        <style>
        .dataframe { width: 100% !important; table-layout: auto; border-collapse: collapse; }
        th { text-align: center !important; }
        td { text-align: left; }
        .dataframe th:nth-child(1), .dataframe td:nth-child(1) { width: 60px; }
        .dataframe th:nth-child(2), .dataframe td:nth-child(2) { width: 180px; }
        .dataframe th:nth-child(3), .dataframe td:nth-child(3) { width: 100px; }
        .dataframe th:nth-child(4), .dataframe td:nth-child(4) { width: 300px; }
        .dataframe th:nth-child(5), .dataframe td:nth-child(5) { width: 60px; }
        .dataframe td, .dataframe th { padding: 10px; border: 1px solid #ddd; }
        </style>
        """, 
        unsafe_allow_html=True
    )

    columns_to_show = ['Sr No', 'Name', 'Class', 'Date', 'Total']
    st.write(filtered_df[columns_to_show].to_html(index=False), unsafe_allow_html=True)

def bar_chart_todays_present(): 
    # Get today's date in the same format as in your CSV
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Check if the CSV has been read properly
    filename = 'Face_recog_Attendance/Attendance.csv'
    if not os.path.isfile(filename):
        st.error("Attendance file not found.")
        return

    df = pd.read_csv(filename)

    # Ensure the Date column is treated as strings and split correctly
    df['Date'] = df['Date'].astype(str).apply(lambda x: x.split(','))  # Ensure correct split

    # Check if today's date is in the list of dates for each student
    present_today = df[df['Date'].apply(lambda dates: today_date in dates)]

    # Check if present_today is empty
    if present_today.empty:
        st.warning("No students present today.")
        return  # Exit the function if no students are present today

    # Group by Class and count the number of students present
    attendance_by_class = present_today['Class'].value_counts()

    # Check if attendance_by_class is empty
    if attendance_by_class.empty:
        st.warning("No attendance records available.")
        return  # Exit the function if no attendance records

    st.subheader("Class-wise Today's Attendance")

    # Plotting the bar chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(attendance_by_class.index, attendance_by_class.values, color='lightgreen')

    # Setting titles and labels
    plt.title('Attendance Today by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Students Present')
    plt.xticks(rotation=45)

    # Annotate each bar with the count of students
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                 ha='center', va='bottom')  # Position text above the bar
        
    # Display the plot in Streamlit
    st.pyplot(plt)

def display_student_attendance(search_name): 
    filename = 'Face_recog_Attendance/Attendance.csv'

    if os.path.isfile(filename):
        try:
            df = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            st.error("The attendance file is empty.")
            return
        except pd.errors.ParserError:
            st.error("Error parsing the attendance file. Please check the file format.")
            return

        # Ensure the correct 'Date' column exists
        if 'Date' not in df.columns:
            st.error("The 'Date' column is missing from the attendance file.")
            return

        # Filter by student name
        filtered_df = df[df['Name'].str.contains(search_name, case=False, na=False)]

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            # Collect all attendance dates separated by commas
            all_dates = []
            for dates in filtered_df['Date']:
                # Split each entry by comma and strip any extra spaces
                all_dates.extend([date.strip() for date in dates.split(',') if date.strip()])

            # Convert the list of dates to a set to get unique attendance days
            unique_dates = set(all_dates)

            # Convert the list of unique dates back to a DataFrame for further processing
            dates_df = pd.DataFrame({'Date': pd.to_datetime(list(unique_dates), errors='coerce')})

            # Extract month and year from the 'Date'
            dates_df['Month'] = dates_df['Date'].dt.month
            dates_df['Year'] = dates_df['Date'].dt.year

            # Count the number of days present per month
            attendance_per_month = dates_df.groupby(['Year', 'Month'])['Date'].count().reset_index(name='Days Present')

            # Create a month-year label for the x-axis
            attendance_per_month['Month-Year'] = attendance_per_month.apply(lambda x: f"{datetime(2000, x['Month'], 1).strftime('%B')} {x['Year']}", axis=1)

            # Plot the bar chart using Matplotlib
            plt.figure(figsize=(10, 5))
            bars = plt.bar(attendance_per_month['Month-Year'], attendance_per_month['Days Present'], color='lightgreen')  # Set color to lightgreen

            # Set chart labels and title
            plt.xlabel('Month-Year')
            plt.ylabel('Number of Days Present')
            plt.title(f'Attendance for {search_name}')
            plt.xticks(rotation=45)

            # Annotate each bar with the count of days present
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                         ha='center', va='bottom')  # Annotate above the bar

            # Display the bar chart in Streamlit
            st.pyplot(plt)

        else:
            st.warning(f"No attendance records found for {search_name}.")
    else:
        st.error("Attendance file not found.")

# Function to count attendance dates for a student or teacher
def count_attendance_dates(name):
    filename = 'Face_recog_Attendance/Attendance.csv'
    
    # Check if the attendance file exists
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        
        # Find the record of the person (either student or teacher)
        person_record = df[(df['Name'] == name) & (df['Class'] == df[df['Name'] == name]['Class'].values[0])]
        
        if not person_record.empty:
            # Collect all attendance dates separated by commas
            all_dates = []
            for dates in person_record['Date']:
                all_dates.extend([date.strip() for date in dates.split(',') if date.strip()])

            # Get unique dates
            unique_dates = set(all_dates)
            attendance_count = len(unique_dates)
            return attendance_count
        else:
            st.error(f"No attendance records found for {name}.")
            return 0
    else:
        st.error("Attendance file not found.")
        return 0

# Function to calculate attendance percentage for a student compared to the teacher
def calculate_attendance_percentage(student_name, teacher_name):
    student_total_days = count_attendance_dates(student_name)
    teacher_total_days = count_attendance_dates(teacher_name)

    # Check if both student and teacher have attendance records
    if student_total_days > 0 and teacher_total_days > 0:
        # Calculate attendance percentage
        attendance_percentage = (student_total_days / teacher_total_days) * 100
        return attendance_percentage
    else:
        st.error("No valid attendance records found for comparison.")
        return None

  
    # Streamlit UI for 'Dashboard'
if nav == "Dashboard":
    st.header("Attendance Dashboard")

    # Load the CSV file (make sure to replace with your actual file path)
    df = pd.read_csv('Face_recog_Attendance/Attendance.csv')

        # Get today's date in the same format as in your CSV (e.g., 'YYYY-MM-DD')
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Split the Date column into lists of individual dates (in case there are multiple dates for each student)
    df['Date'] = df['Date'].astype(str).apply(lambda x: x.split(', '))

    # Check if today's date is in the list of dates for each student
    present_today = df[df['Date'].apply(lambda dates: today_date in dates)]

    # Get the total number of students present today
    total_present = present_today.shape[0]
    
    # Display the total present today in Streamlit
    st.metric("Total Students Present Today", total_present)
    bar_chart_todays_present()
    # Get current month and year
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Create three columns for month, year, and class input
    col1, col2, col3 = st.columns(3)

    with col1:
        # Month selection (default to current month)
        month = st.selectbox(
            "Select Month", 
            list(range(1, 13)), 
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'), 
            index=current_month - 1
        )

    with col2:
        # Year input (default to current year)
        year = st.number_input(
            "Select Year", 
            min_value=2000, 
            max_value=datetime.now().year, 
            value=current_year
        )

    with col3:
        unique_classes = get_unique_classes()
        search_class = st.selectbox("Filter by Class:", options=[""] + unique_classes)

    with col1:
        # Input for searching a specific student by name
        unique_names = get_unique_names()
        search_name = st.selectbox("Select Name to search:", options=[""] + unique_names)
    with col2:
        # Input for searching a specific student by name
        unique_names = get_unique_names()
        search_teacher = st.selectbox("Select teacher:", options=[""] + unique_names)    
    
        # Button logic improvement
    if st.button("Get percentage"):
                if search_name and search_teacher:
                    attendance_percentage = calculate_attendance_percentage(search_name, search_teacher)
                    if attendance_percentage is not None:
                        st.header(f"{attendance_percentage:.2f}%")
                        display_student_attendance(search_name)
                else:
                    st.warning("Please enter both a student and a teacher's name to get percentage.")

                if search_name:
                        display_student_attendance(search_name)
                else:
                    st.warning("Please enter a name to search.")  

    # Automatically show attendance records based on selections
    display_attendance_records(month, year, search_class)

    # Footer
st.markdown("___")
st.markdown("FACE RECOGNITION ATTENDANCE TRACKER.")