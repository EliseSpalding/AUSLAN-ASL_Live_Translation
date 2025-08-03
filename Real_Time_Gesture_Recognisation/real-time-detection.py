# %%
# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw


# %%
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# %%
# Initialize MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def put_aligned_text(cv2_img, text1, font_path1, size1, 
                     text2, font_path2, size2, 
                     start_pos, color=(255,255,255), spacing=10):

    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_rgb)
    draw = ImageDraw.Draw(pil_img)

    font1 = ImageFont.truetype(font_path1, size1)
    font2 = ImageFont.truetype(font_path2, size2)

    x, y = start_pos
    for ch1, ch2 in zip(text1, text2):
        # Draw English letter
        draw.text((x, y), ch1, font=font1, fill=color)

        # Measure English char width/height with getbbox
        bbox = font1.getbbox(ch1)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # Draw ASL letter directly under English
        draw.text((x, y + h + 10), ch2, font=font2, fill=color)

        # Advance x position for next pair
        x += w + spacing

    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def get_text_size(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)  # returns (left, top, right, bottom)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

# Custom font helper
def put_custom_font_text(cv2_img, text, position, font_path, font_size, color):
    # Convert from OpenCV BGR to RGB
    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_rgb)

    # Load font
    font = ImageFont.truetype(font_path, font_size)

    # Create drawing context
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV BGR
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

# Define functions for MediaPipe detection and drawing
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color space
    image.flags.writeable = False
    results = model.process(image)  # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    # # Draw face connections
    # mp_drawing.draw_landmarks(
    #     image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    # )
    # # Draw pose connections
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
    #     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    # )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )



# %%
def extract_keypoints(results):
    # Exclude pose and face landmarks
    # Left Hand
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                         ).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Right Hand
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                          ).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([left_hand, right_hand])


# %%
def normalize_keypoints(sequence):
    # Reshape sequence to (frames, keypoints, coordinates)
    sequence = sequence.reshape(sequence.shape[0], -1, 3)
    normalized_sequence = []
    for frame in sequence:
        # Left Hand
        left_hand = frame[:21, :]
        if np.any(left_hand):
            left_hand_center = left_hand[0, :]  # Wrist
            left_hand = left_hand - left_hand_center  # Center the hand
            hand_size = np.linalg.norm(left_hand[9, :])  # Middle finger MCP
            if hand_size > 0:
                left_hand = left_hand / hand_size  # Scale the hand
            else:
                left_hand = np.zeros((21, 3))
        else:
            left_hand = np.zeros((21, 3))

        # Right Hand
        right_hand = frame[21:, :]
        if np.any(right_hand):
            right_hand_center = right_hand[0, :]  # Wrist
            right_hand = right_hand - right_hand_center  # Center the hand
            hand_size = np.linalg.norm(right_hand[9, :])  # Middle finger MCP
            if hand_size > 0:
                right_hand = right_hand / hand_size  # Scale the hand
            else:
                right_hand = np.zeros((21, 3))
        else:
            right_hand = np.zeros((21, 3))

        # Concatenate normalized keypoints
        frame_normalized = np.concatenate([left_hand, right_hand], axis=0)
        normalized_sequence.append(frame_normalized.flatten())
    return np.array(normalized_sequence)


# %%
# Define data augmentation functions
def add_noise(sequence, noise_level=0.05):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence, scale_factor=0.1):
    scaling = np.random.normal(1.0, scale_factor)
    return sequence * scaling

def time_warp(sequence, sigma=0.2):
    from scipy.interpolate import CubicSpline
    num_frames = sequence.shape[0]
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=num_frames)
    cumulative_warp = np.cumsum(random_warp)
    cumulative_warp = (cumulative_warp - cumulative_warp.min()) / (cumulative_warp.max() - cumulative_warp.min()) * (num_frames - 1)
    cs = CubicSpline(np.arange(num_frames), sequence, axis=0)
    warped_sequence = cs(cumulative_warp)
    return warped_sequence

def augment_sequence(sequence):
    augmented_sequence = sequence.copy()
    if np.random.rand() < 0.5:
        augmented_sequence = add_noise(augmented_sequence)
    if np.random.rand() < 0.5:
        augmented_sequence = scale_sequence(augmented_sequence, scale_factor=0.2)  # Wider scaling factor
    if np.random.rand() < 0.5:
        augmented_sequence = time_warp(augmented_sequence)
    return augmented_sequence

# %%
# Define paths and actions
DATA_PATH = os.path.join('AUSLAN_Data')

# Actions to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
                    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Number of sequences and sequence length
no_sequences = 50
sequence_length = 30

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# %%
print(label_map)

# %% [markdown]
# ### 1. Data Preprocessing
# - Normalization and Centering

# %% [markdown]
# - Load and Normalize Data

# %%
sequences, labels = [], []
for action in actions:
    for sequence_num in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence_num), f"{frame_num}.npy"))
            # Extract only hand keypoints
            res = res[-(21*3*2):]  # Assuming hand keypoints are at the end
            window.append(res)
        window = np.array(window)
        # Normalize keypoints
        window_normalized = normalize_keypoints(window)
        sequences.append(window_normalized)
        labels.append(label_map[action])


# %%
# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# %% [markdown]
# ### 2. Data Augmentation
# - Enhanced Augmentation Functions

# %% [markdown]
# - Apply Augmentation

# %%
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Augment training data
augmented_sequences = []
augmented_labels = []
for seq, label in zip(X_train, y_train):
    augmented_seq = augment_sequence(seq)
    augmented_sequences.append(augmented_seq)
    augmented_labels.append(label)

# Convert augmented data to numpy arrays
augmented_sequences = np.array(augmented_sequences)
augmented_labels = np.array(augmented_labels)

# Combine original and augmented data
X_train_augmented = np.concatenate((X_train, augmented_sequences), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_labels), axis=0)

# Shuffle the augmented training data
X_train_augmented, y_train_augmented = shuffle(X_train_augmented, y_train_augmented)


# %% [markdown]
# ### 3. Model Definition

# %%
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
labels_flat = np.argmax(y_train_augmented, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(labels_flat), y=labels_flat)
class_weights = dict(enumerate(class_weights))

# %%
# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))



# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %% [markdown]
# ### 5. Evaluation

# %%
# Load the best model
model.load_weights('models_weights/best_model_new.keras')

# %%
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')



# %% [markdown]
# ### Real Time Detection Begin:

# %%

# Update the preprocess_frame function
def preprocess_frame(results):
    # Extract keypoints
    keypoints = extract_keypoints(results)
    # Reshape to (1, keypoints)
    keypoints = keypoints.reshape(1, -1)
    # Normalize keypoints
    keypoints = normalize_keypoints(keypoints)
    return keypoints[0]

# %%
colors = [
    (255, 0, 0),      # A
    (0, 255, 0),      # B
    (0, 0, 255),      # C
    (255, 255, 0),    # D
    (0, 255, 255),    # E
    (255, 0, 255),    # F
    (190, 125, 0),    # G
    (0, 190, 125),    # H
    (190, 0, 125),    # I
    (25, 185, 0),     # J
    (0, 25, 185),     # K
    (185, 0, 25),     # L
    (100, 0, 100),    # M
    (0, 100, 100),    # N
    (123, 123, 0),    # O
    (255, 165, 0),    # P
    (75, 0, 130),     # Q
    (255, 20, 147),   # R
    (0, 128, 0),      # S
    (128, 0, 128),    # T
    (0, 0, 128),      # U
    (128, 128, 0),    # V
    (0, 100, 200),    # W
    (28, 20, 50),   # X
    (85, 100, 128),   # Y
    (70, 75, 75)   # Z
]


# %%
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    
    # Get the dimensions of the window
    frame_height, frame_width, _ = output_frame.shape
    
    # Set configuration for dynamic layout
    left_column_letters = 13  # Adjusted number of letters on the left side
    top_row_letters = 0       # No letters on the top side
    label_height = 30         # Vertical spacing for left and right
    label_width = 120         # Maximum width of rectangles for probability bars
    top_spacing = 90          # Adjustable spacing for top row
    top_letter_spacing = 5    # Adjustable spacing between top letters

    # Iterate over the results and position letters accordingly
    for num, prob in enumerate(res):
        if num < left_column_letters:
            # Left-side placement
            x_offset = 0
            y_position = top_spacing + num * label_height
            # Probability bars increasing to the right
            cv2.rectangle(output_frame, (x_offset, y_position), 
                          (x_offset + int(prob * 100), y_position + 20), 
                          colors[num % len(colors)], -1)
            cv2.putText(output_frame, actions[num], (x_offset, y_position + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # Right-side placement
            x_offset = frame_width - label_width  # Adjust the starting point to be near the edge
            y_position = top_spacing + (num - left_column_letters) * label_height
            # Probability bars increasing to the left
            bar_start = x_offset + label_width - int(prob * 100)
            cv2.rectangle(output_frame, 
                          (bar_start, y_position),
                          (x_offset + label_width, y_position + 20), 
                          colors[num % len(colors)], -1)
            cv2.putText(output_frame, actions[num], 
                        (bar_start - 20, y_position + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return output_frame

# %%
# New Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

# Load your trained model
model = tf.keras.models.load_model('models_weights/best_model_new.keras')  # Ensure 'best_model.keras' is your trained model

cap = cv2.VideoCapture(3)

# Access MediaPipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0,  # Simplify the model since we're only using hand landmarks
) as holistic:
    while cap.isOpened():
        # Read the feed
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)  # Uncomment to see the detection results

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Preprocess the frame
        keypoints = preprocess_frame(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames

        # Once we have 30 frames, make a prediction
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualize results if the prediction is consistent
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # Show probability visualization
            image = prob_viz(res, actions, image, colors)

        # Display the result
        cv2.rectangle(image, (0, 0), (640, 90), (245, 117, 16), -1)
        english = ' '.join(sentence)
        asl = ' '.join(sentence)  # assuming the ASL font maps same chars

        image = put_aligned_text(
            image,
            english, "DAYROM__.ttf", 40,       # English font
            asl, "ASLHandsByFrank.otf", 40, # ASL font
            start_pos=(10, 3), 
            color=(255,255,255),
            spacing=15
        )

        # Show to the screen
        cv2.imshow('Frame', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




