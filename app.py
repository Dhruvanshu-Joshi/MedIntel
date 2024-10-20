# # import os
# # import json
# # import cv2
# # import base64
# # import matplotlib.pyplot as plt
# # from flask import Flask, render_template, request, redirect, url_for
# # from werkzeug.utils import secure_filename
# # from path import Path
# # from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# # app = Flask(__name__)

# # # Configurations
# # UPLOAD_FOLDER = 'uploads/'
# # PROCESSED_FOLDER = 'static/images/'
# # OUTPUT_FOLDER = 'output/'
# # DATA_FOLDER = 'data/'
# # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
# # app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# # # Load configurations for the handwritten text recognition pipeline
# # with open(os.path.join(DATA_FOLDER, 'config.json')) as f:
# #     sample_config = json.load(f)

# # with open(os.path.join(DATA_FOLDER, 'words_alpha.txt')) as f:
# #     word_list = [w.strip().upper() for w in f.readlines()]
# # prefix_tree = PrefixTree(word_list)

# # # In-memory list to store user documents (can be replaced with a database)
# # user_documents = []

# # # Function to check file extension
# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/handwritten')
# # def handwritten():
# #     return render_template('handwritten.html')

# # @app.route('/upload', methods=['GET', 'POST'])
# # def upload():
# #     if request.method == 'POST':
# #         if 'file' not in request.files:
# #             return redirect(request.url)
# #         file = request.files['file']
# #         if file and allowed_file(file.filename):
# #             filename = secure_filename(file.filename)
# #             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #             file.save(filepath)

# #             # Process the image and extract text
# #             processed_image_name, extracted_text = process_image(filepath, filename)

# #             # Store the document information
# #             user_documents.append({
# #                 'filename': filename,
# #                 'processed_image': processed_image_name,
# #                 'extracted_text': extracted_text
# #             })

# #             return render_template('upload.html',  
# #                                    processed_image=processed_image_name, 
# #                                    extracted_text=extracted_text)

# #     return render_template('upload.html')

# # @app.route('/dashboard')
# # def dashboard():
# #     return render_template('dashboard.html', documents=user_documents)

# # def process_image(filepath, filename):
# #     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
# #     scale = sample_config.get(filename, {}).get('scale', 1)
# #     margin = sample_config.get(filename, {}).get('margin', 0)

# #     # Reading the image and extracting text
# #     decoder = 'best_path'
# #     read_lines = read_page(img,
# #                            detector_config=DetectorConfig(scale=scale, margin=margin),
# #                            line_clustering_config=LineClusteringConfig(min_words_per_line=2),
# #                            reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

# #     # Save extracted text to a variable
# #     extracted_text = ""
# #     for read_line in read_lines:
# #         line_text = ' '.join(read_word.text for read_word in read_line)
# #         extracted_text += line_text + '\n'

# #     # Save processed image with text bounding boxes
# #     processed_image_name = f"{Path(filename).stem}_processed.png"
# #     processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_name)
# #     plt.imshow(img, cmap='gray')

# #     for read_line in read_lines:
# #         for read_word in read_line:
# #             aabb = read_word.aabb
# #             xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
# #             ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
# #             plt.plot(xs, ys, c='r')
# #             plt.text(aabb.xmin, aabb.ymin - 2, read_word.text, color='white')

# #     plt.savefig(processed_image_path)
# #     plt.close()

# #     # Create a collage of the original and processed images
# #     collage_image_name = f"{Path(filename).stem}_collage.png"
# #     collage_image_path = os.path.join(app.config['PROCESSED_FOLDER'], collage_image_name)

# #     # Create a collage using matplotlib
# #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# #     axs[0].imshow(img, cmap='gray')
# #     axs[0].set_title('Original Image')
# #     axs[0].axis('off')

# #     processed_img = cv2.imread(processed_image_path)
# #     axs[1].imshow(processed_img)
# #     axs[1].set_title('Processed Image')
# #     axs[1].axis('off')

# #     plt.tight_layout()
# #     plt.savefig(collage_image_path)
# #     plt.close()

# #     return collage_image_name, extracted_text

# # @app.route('/save_text', methods=['POST'])
# # def save_text():
# #     corrected_text = request.form['corrected_text']
# #     # Here you can save the corrected text to a file or database
# #     # For simplicity, we'll save it to a file
# #     corrected_text_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'corrected_text.txt')
# #     with open(corrected_text_filename, 'w') as f:
# #         f.write(corrected_text)
# #     return redirect(url_for('home'))

# # if __name__ == '__main__':
# #     app.run(debug=True)

# import os
# import json
# import cv2
# import base64
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from path import Path
# from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# app = Flask(__name__)

# # Configurations
# UPLOAD_FOLDER = 'uploads/'
# PROCESSED_FOLDER = 'static/images/'
# OUTPUT_FOLDER = 'output/'
# DATA_FOLDER = 'data/'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# # Load configurations for the handwritten text recognition pipeline
# with open(os.path.join(DATA_FOLDER, 'config.json')) as f:
#     sample_config = json.load(f)

# with open(os.path.join(DATA_FOLDER, 'words_alpha.txt')) as f:
#     word_list = [w.strip().upper() for w in f.readlines()]
# prefix_tree = PrefixTree(word_list)

# # In-memory list to store user documents (can be replaced with a database)
# user_documents = []

# # Function to check file extension
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def home():
#     return render_template('index.html')

# # Handwritten routes
# @app.route('/handwritten')
# def handwritten():
#     return render_template('handwritten.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             # Process the image and extract text (Handwritten)
#             processed_image_name, extracted_text = process_image(filepath, filename)

#             # Store the document information
#             user_documents.append({
#                 'filename': filename,
#                 'processed_image': processed_image_name,
#                 'extracted_text': extracted_text,
#                 'type': 'handwritten'
#             })

#             return render_template('upload.html',  
#                                    processed_image=processed_image_name, 
#                                    extracted_text=extracted_text)

#     return render_template('upload.html')

# # Printed routes
# @app.route('/printed')
# def printed():
#     return render_template('printed.html')

# @app.route('/upload_printed', methods=['GET', 'POST'])
# def upload_printed():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             # Process the image and extract text (Printed)
#             processed_image_name, extracted_text = process_image(filepath, filename)

#             # Store the document information
#             user_documents.append({
#                 'filename': filename,
#                 'processed_image': processed_image_name,
#                 'extracted_text': extracted_text,
#                 'type': 'printed'
#             })

#             return render_template('upload_printed.html',  
#                                    processed_image=processed_image_name, 
#                                    extracted_text=extracted_text)

#     return render_template('upload_printed.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html', documents=user_documents)

# def process_image(filepath, filename):
#     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     scale = sample_config.get(filename, {}).get('scale', 1)
#     margin = sample_config.get(filename, {}).get('margin', 0)

#     # Reading the image and extracting text
#     decoder = 'best_path'
#     read_lines = read_page(img,
#                            detector_config=DetectorConfig(scale=scale, margin=margin),
#                            line_clustering_config=LineClusteringConfig(min_words_per_line=2),
#                            reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

#     # Save extracted text to a variable
#     extracted_text = ""
#     for read_line in read_lines:
#         line_text = ' '.join(read_word.text for read_word in read_line)
#         extracted_text += line_text + '\n'

#     # Save processed image with text bounding boxes
#     processed_image_name = f"{Path(filename).stem}_processed.png"
#     processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_name)
#     plt.imshow(img, cmap='gray')

#     for read_line in read_lines:
#         for read_word in read_line:
#             aabb = read_word.aabb
#             xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
#             ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
#             plt.plot(xs, ys, c='r')
#             plt.text(aabb.xmin, aabb.ymin - 2, read_word.text, color='white')

#     plt.savefig(processed_image_path)
#     plt.close()

#     # Create a collage of the original and processed images
#     collage_image_name = f"{Path(filename).stem}_collage.png"
#     collage_image_path = os.path.join(app.config['PROCESSED_FOLDER'], collage_image_name)

#     # Create a collage using matplotlib
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#     axs[0].imshow(img, cmap='gray')
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')

#     processed_img = cv2.imread(processed_image_path)
#     axs[1].imshow(processed_img)
#     axs[1].set_title('Processed Image')
#     axs[1].axis('off')

#     plt.tight_layout()
#     plt.savefig(collage_image_path)
#     plt.close()

#     return collage_image_name, extracted_text

# @app.route('/save_text', methods=['POST'])
# def save_text():
#     corrected_text = request.form['corrected_text']
#     # Here you can save the corrected text to a file or database
#     # For simplicity, we'll save it to a file
#     corrected_text_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'corrected_text.txt')
#     with open(corrected_text_filename, 'w') as f:
#         f.write(corrected_text)
#     return redirect(url_for('home'))

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import json
import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from path import Path
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree
import pytesseract
from xray_analysis import function1

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'static/images/'
OUTPUT_FOLDER = 'output/'
DATA_FOLDER = 'data/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])

# Load configurations for the handwritten text recognition pipeline
with open(os.path.join(DATA_FOLDER, 'config.json')) as f:
    sample_config = json.load(f)

with open(os.path.join(DATA_FOLDER, 'words_alpha.txt')) as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# In-memory list to store user documents (can be replaced with a database)
user_documents = []

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

# Handwritten routes
@app.route('/handwritten')
def handwritten():
    return render_template('handwritten.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image and extract text (Handwritten)
            processed_image_name, extracted_text = process_handwritten_image(filepath, filename)

            # Store the document information
            user_documents.append({
                'filename': filename,
                'processed_image': processed_image_name,
                'extracted_text': extracted_text,
                'type': 'handwritten'
            })

            return render_template('upload.html',  
                                   processed_image=processed_image_name, 
                                   extracted_text=extracted_text)

    return render_template('upload.html')

# Printed routes
@app.route('/printed')
def printed():
    return render_template('printed.html')

@app.route('/upload_printed', methods=['GET', 'POST'])
def upload_printed():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image and extract text (Printed)
            processed_image_name, extracted_text = process_printed_image(filepath, filename)

            # Store the document information
            user_documents.append({
                'filename': filename,
                'processed_image': processed_image_name,
                'extracted_text': extracted_text,
                'type': 'printed'
            })

            return render_template('upload_printed.html',  
                                   processed_image=processed_image_name, 
                                   extracted_text=extracted_text)

    return render_template('upload_printed.html')

# Route for X-ray Report Generation page
@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/upload_xray', methods=['GET', 'POST'])
def upload_xray():
    if request.method == 'POST':
        # Handle first compulsory X-ray image
        xray_image1 = request.files.get('file1')
        processed_xray_image1 = None
        file_path1 = None
        if xray_image1:
            file_path1 = os.path.join('static', 'images', xray_image1.filename)
            xray_image1.save(file_path1)
            session['first_xray_path'] = file_path1

            # Call function to process first X-ray image
            # processed_xray_image1 = process_xray_image(file_path1)

        # Handle second optional X-ray image
        xray_image2 = request.files.get('file2')
        processed_xray_image2 = None
        file_path2=None
        if xray_image2:
            file_path2 = os.path.join('static', 'images', xray_image2.filename)
            xray_image2.save(file_path2)
        session['second_xray_path'] = file_path2 if file_path2 else None

            # Call function to process second X-ray image
            # processed_xray_image2 = process_xray_image(file_path2)

        # Extract report from processed X-ray(s)
        predicted_caption = function1([file_path1], [file_path2])
        extracted_report = predicted_caption

        # Render the result page with processed images and report
        return render_template('upload_xray.html', 
                               processed_xray_image1=xray_image1.filename, 
                               processed_xray_image2=xray_image2.filename, 
                               extracted_report=extracted_report)

    return render_template('upload_xray.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', documents=user_documents)

def process_handwritten_image(filepath, filename):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    scale = sample_config.get(filename, {}).get('scale', 1)
    margin = sample_config.get(filename, {}).get('margin', 0)

    # Reading the image and extracting text
    decoder = 'best_path'
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                           reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

    # Save extracted text to a variable
    extracted_text = ""
    for read_line in read_lines:
        line_text = ' '.join(read_word.text for read_word in read_line)
        extracted_text += line_text + '\n'

    # Save processed image with text bounding boxes
    processed_image_name = f"{Path(filename).stem}_processed.png"
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_name)
    plt.imshow(img, cmap='gray')

    for read_line in read_lines:
        for read_word in read_line:
            aabb = read_word.aabb
            xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
            ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
            plt.plot(xs, ys, c='r')
            plt.text(aabb.xmin, aabb.ymin - 2, read_word.text, color='white')

    plt.savefig(processed_image_path)
    plt.close()

    # Create a collage of the original and processed images
    collage_image_name = f"{Path(filename).stem}_collage.png"
    collage_image_path = os.path.join(app.config['PROCESSED_FOLDER'], collage_image_name)

    # Create a collage using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    processed_img = cv2.imread(processed_image_path)
    axs[1].imshow(processed_img)
    axs[1].set_title('Processed Image')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(collage_image_path)
    plt.close()

    return collage_image_name, extracted_text

@app.route('/save_xray_report', methods=['POST'])
def save_xray_report():
    # Get the corrected report from the form
    corrected_report = request.form.get('corrected_report')
    
    # Retrieve the image paths from the session or request
    image_path1 = session.get('first_xray_path')
    image_path2 = session.get('second_xray_path')

    # Save the new collage with corrected text
    collage_image_name = save_collaged_xray_with_text(image_path1, image_path2, corrected_report)

    # # Optionally save the corrected report as a file or to the database
    # corrected_report_path = os.path.join('static', 'reports', 'corrected_report.png')
    # with open(corrected_report_path, 'w') as f:
    #     f.write(corrected_report)
    
    # Redirect back to the x-ray upload page or show a success message
    return redirect(url_for('report'))

def save_collaged_xray_with_text(image_path1, image_path2, extracted_report):
    # Load the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2) if image_path2 else None

    # Check if the images are loaded correctly
    if img1 is None:
        raise ValueError("Image 1 could not be loaded. Check the path.")
    if img2 is not None and img2 is None:
        raise ValueError("Image 2 could not be loaded. Check the path.")

    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 255, 0)
    thickness = 2
    line_type = cv2.LINE_AA

    # Break the report into lines for better placement
    report_lines = extracted_report.split('\n')

    # Calculate the height available for text
    height1, width1 = img1.shape[:2]
    max_height = height1 - 40  # 20 pixels padding from top and bottom
    line_spacing = 30  # Space between lines

    # Determine the maximum font size that fits the image height
    font_scale = 0.6
    while True:
        # Calculate text size
        text_size = cv2.getTextSize(report_lines[0], font, font_scale, thickness)[0]
        total_height = len(report_lines) * line_spacing
        if total_height < max_height:
            break
        font_scale -= 0.1  # Reduce font scale until it fits

    # Add text to the first image
    y0 = 30  # Starting y position
    for line in report_lines:
        cv2.putText(img1, line, (10, y0), font, font_scale, font_color, thickness, line_type)
        y0 += line_spacing

    # If there is a second image, add text to it
    if img2 is not None:
        y0 = 30  # Reset starting y position for second image
        for line in report_lines:
            cv2.putText(img2, line, (10, y0), font, font_scale, font_color, thickness, line_type)
            y0 += line_spacing

    # Ensure both images have the same height for horizontal stacking
    if img2 is not None:
        # Resize the second image to match the height of the first image
        height2, width2 = img2.shape[:2]

        if height1 != height2:
            img2_resized = cv2.resize(img2, (width2, height1))  # Resize img2 to img1's height
        else:
            img2_resized = img2

        # Create a collage of the two images
        collage = np.hstack((img1, img2_resized))  # Side-by-side collage
    else:
        collage = img1

    # Save the collage image
    collage_image_name = 'collaged_xray_with_text.png'
    collage_image_path = os.path.join('static', 'images', collage_image_name)
    cv2.imwrite(collage_image_path, collage)

    return collage_image_name

def process_printed_image(filepath, filename):
    # Load the image using OpenCV
    img = cv2.imread(filepath)
    
    # Convert the image to grayscale for better OCR results
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(thresh_image)

    # Save processed image (if you want to keep the preprocessed image)
    processed_image_name = f"{Path(filename).stem}_printed_processed.png"
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_name)
    cv2.imwrite(processed_image_path, thresh_image)

    # Create a collage of the original and processed images
    collage_image_name = f"{Path(filename).stem}_printed_collage.png"
    collage_image_path = os.path.join(app.config['PROCESSED_FOLDER'], collage_image_name)

    # Create a collage using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(thresh_image, cmap='gray')
    axs[1].set_title('Processed Image')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(collage_image_path)
    plt.close()

    return collage_image_name, extracted_text

@app.route('/save_text', methods=['POST'])
def save_text():
    corrected_text = request.form['corrected_text']
    # Save the corrected text to a file
    corrected_text_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'corrected_text.txt')
    with open(corrected_text_filename, 'w') as f:
        f.write(corrected_text)
    return redirect(url_for('home'))

app.secret_key = 'your_super_secret_key'
if __name__ == '__main__':
    app.run(debug=True)
