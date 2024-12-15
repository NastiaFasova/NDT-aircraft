import json
import os
import time

import cv2
import joblib
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import pythreejs as p3
import streamlit as st
import streamlit.components.v1 as components
import trimesh
import yaml
from huggingface_hub import login
from IPython.display import display
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu
from yaml.loader import SafeLoader

from functions import download_data, generate_answer

# Display the HTML content in Streamlit

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('<p class="small-font">Літайте безпечно!</p>', unsafe_allow_html=True)

with open("data.yaml", mode="r") as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml["names"]
# print(labels)

# Load the ONNX model , fault model
yolo = ort.InferenceSession("Model/weights/best.onnx")
Fault = joblib.load("Model/Wire_Fault.joblib")


def custom_css():
    st.markdown(
        """
        <style>
        body {
            background-image: url('/static/plane.png');
        }
        .big-font {
            font-size:50px !important;
            font-weight: bold;
            color: white;
        }
        .small-font {
            font-family: monospace;
            font-weight: 200;
            font-style: italic;
            font-size:20px !important;
        }
        .normal-font {
            font-family: fangsong;
            font-weight: 800;
            font-size: 30px !important;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


custom_css()


def detect_dents_and_cracks(image):
    # Preprocess the image
    image = image.copy()
    row, col, d = image.shape

    # Convert image to a square image
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Prepare the image blob
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(
        input_image, 1 / 255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False
    )

    # Perform inference using the ONNX model
    yolo.set_providers(["CPUExecutionProvider"])
    yolo_input_name = yolo.get_inputs()[0].name
    yolo_output_name = yolo.get_outputs()[0].name
    preds = yolo.run([yolo_output_name], {yolo_input_name: blob})[0]

    # Debugging information
    st.write(f"Predictions shape: {preds.shape}")

    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    # widht and height of the image (input_image)
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WH_YOLO
    y_factor = image_h / INPUT_WH_YOLO

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detection an object
        if confidence > 0.4:
            class_score = row[5:].max()  # maximum probability from 20 objects
            class_id = row[
                5:
            ].argmax()  # get the index position at which max probabilty occur

            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                # construct bounding from four values
                # left, top, width and height
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])

                # append values into the list
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # Check if there are any detections
    if len(boxes_np) == 0 or len(confidences_np) == 0:
        st.write("No cracks or dents detected in the image.")
        return image, []

    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()
    damage_locations = []
    for ind in index:
        # extract bounding box
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        classes_id = classes[ind]
        class_name = labels[classes_id]

        text = f"{class_name}: {bb_conf}%"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)

        cv2.putText(
            image,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        damage_locations.append((x, y, w, h, class_name))

    return image, damage_locations

    # def render_3d():
    stl_path = "AirplaneAllFiles/AirplaneForFreestl.stl"
    # Load the STL file using trimesh
    mesh = trimesh.load(stl_path)
    # Create a scene
    scene = pyrender.Scene()
    # Create a mesh node with the loaded STL file
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    # Add the mesh node to the scene
    scene.add(mesh_node)
    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    s = np.sqrt(2) / 2
    camera_pose = np.array(
        [
            [0.0, -s, s, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, s, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene.add(camera, pose=camera_pose)
    # Set up the light
    light = pyrender.SpotLight(
        color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0
    )
    scene.add(light, pose=camera_pose)
    # Render the scene
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
    color, _ = r.render(scene)
    image = Image.fromarray(color)
    st.image(image, caption="3D Model Render")


def suggest_repair_actions(damage_locations):
    st.write("### Suggested Repair Actions")
    for loc in damage_locations:
        x, y, w, h, class_name = loc
        st.write(
            f"- **{class_name}** detected at location (x={x}, y={y}, width={w}, height={h}):"
        )
        if class_name == "dent":
            st.write("Inspect the dent for any cracks or paint damage:")
            st.write("- Use appropriate tools to repair the dent.")
            st.write("- Check for structural integrity post-repair.")
        elif class_name == "crack":
            st.write("Assess the extent of the crack:")
            st.write("- Determine if the crack is superficial or structural.")
            st.write("- Measure the length and depth of the crack.")
            st.write("Follow standard procedures to seal or replace the damaged part:")
            st.write("- Clean the area around the crack.")
            st.write("- Apply sealant or adhesive as per manufacturer's instructions.")
            st.write("- If necessary, replace the damaged part.")
            st.write("Perform a thorough inspection to ensure no further damage:")
            st.write("- Check adjacent areas for similar issues.")
            st.write("- Test the repaired area under normal operating conditions.")
        else:
            st.write(
                "Follow appropriate maintenance procedures for the detected issue."
            )
        st.write("---")


def streamlit_config() -> None:
    """
    Configures the Streamlit app for the Resume Analyzer AI.
    """
    st.markdown(
        '<h1 style="text-align: center;">AI-Powered airplain damage detection',
        unsafe_allow_html=True,
    )


def initialize_session_state():
    if "use_openai" not in st.session_state:
        st.session_state.use_openai = False
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "analyze_resume_full" not in st.session_state:
        st.session_state.analyze_resume_full = None
        st.session_state.suggested_jobs = None
    if "scrap_vacancy" not in st.session_state:
        st.session_state.scrap_vacancy = None


def initialize_llm(use_openai=False, openai_api_key=None):
    if use_openai and openai_api_key:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo-0125",
            temperature=0.2,
        )
        model_name = "OpenAI GPT"
    else:
        download_data()
        MODEL_CONTEXT_WINDOW = 8192
        llm = LlamaCpp(
            model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
            n_ctx=MODEL_CONTEXT_WINDOW,
            temperature=0.2,
            verbose=True,
        )
        model_name = "Local LlamaCpp (Mistral 7B)"
    return llm, model_name


def sidebar_model_selection():
    with st.sidebar:
        add_vertical_space(4)
        st.header("Model Selection")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Open Source Model",
                type="primary" if not st.session_state.use_openai else "secondary",
            ):
                st.session_state.use_openai = False
        with col2:
            if st.button(
                "OpenAI GPT",
                type="primary" if st.session_state.use_openai else "secondary",
            ):
                st.session_state.use_openai = True

        if st.session_state.use_openai:
            openai_api_key = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                value=st.session_state.openai_api_key or "",
            )
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key to use GPT model.")
            elif openai_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_api_key
        else:
            st.session_state.openai_api_key = None

    # Check if the model needs to be initialized or updated
    if st.session_state.llm is None or st.session_state.use_openai != (
        st.session_state.model_name == "OpenAI GPT"
    ):
        st.session_state.llm, st.session_state.model_name = initialize_llm(
            st.session_state.use_openai, st.session_state.openai_api_key
        )
        st.sidebar.success(f"Switched to {st.session_state.model_name}")

    st.sidebar.write(f"Current model: {st.session_state.model_name}")


if __name__ == "__main__":
    selected = option_menu(
        menu_title=None,
        options=["Розпізнавання пошкоджень", "AI-Довідник"],
        icons=["activity", "activity", "info"],
        default_index=0,
        orientation="horizontal",
        menu_icon="cast",
    )

    if selected == "Розпізнавання пошкоджень":
        st.markdown(
            '<p class="normal-font">Оцінка ціліcності літака ✈️</p>',
            unsafe_allow_html=True,
        )
        st.info(
            "Завантажте зображення, щоб оцінити деталь на наявність пошкоджень",
            icon="ℹ️",
        )
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

            # create a table
            col1, col2 = st.columns([1, 1])
            # Display the original image
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            # Detect dents and cracks
            marked_image, damage_locations = detect_dents_and_cracks(image)

            # Display the marked image
            with col2:
                st.image(
                    marked_image,
                    caption="Image with Marked Faults",
                    use_column_width=True,
                )

            suggest_repair_actions(damage_locations)

            # st.write("### 3D Model of Detections")
            # Update the call to the embed_3d_model function in your main code

        # render_3d()
        # components.html(html_content, height=600,scrolling=False)

    if selected == "AI-Довідник":
        streamlit_config()
        initialize_session_state()
        sidebar_model_selection()
        st.session_state.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Logos
        logos = {
            "Mistral AI": "https://plugins.matomo.org/MistralAI/images/5.6.3/A_Mistral_AI_logo.png?w=1024",
            "Transformers": "https://miro.medium.com/v2/resize:fit:631/0*ewH4dvb8djn6KenV.png",
            "Qdrant": "https://qdrant.tech/images/logo_with_text.png",
            "LangChain": "https://deepsense.ai/wp-content/uploads/2023/10/LangChain-announces-partnership-with-deepsense.jpeg",
            "Llama CPP": "https://repository-images.githubusercontent.com/612354784/c59e3320-a236-4182-941f-ea3f1a0f50e7",
            "Docker": "https://blog.codewithdan.com/wp-content/uploads/2023/06/Docker-Logo.png",
        }

        # Display logos in a single row with white background
        st.markdown(
            """
        <table style='border-collapse: collapse; width: 100%; padding-top: 20px; padding-bottom: 20px;'>
        <tr>
            <td style='text-align: center; border: 0; padding: 10px 0;'>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style='background-color: white; display: inline-block; padding: 10px;'>
            <img src="{logos["Mistral AI"]}" alt="{"Mistral AI"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
            <img src="{logos["Transformers"]}" alt="{"Transformers"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
            <img src="{logos["Qdrant"]}" alt="{"Qdrant"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
            <img src="{logos["LangChain"]}" alt="{"LangChain"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
            <img src="{logos["Llama CPP"]}" alt="{"Llama CPP"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
            <img src="{logos["Docker"]}" alt="{"Docker"}" style="max-height: 60px; max-width: 150px; width: auto; height: auto; object-fit: contain;">
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            </td>
        </tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        # User input
        user_input = st.text_area("Ask your question")

        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                start_time = time.time()
                response_data = generate_answer(
                    user_input, st.session_state.llm, st.session_state.embeddings
                )
                end_time = time.time()

            if response_data:
                st.markdown(f"Answer: {response_data['answer']}\n")
                st.markdown(
                    f"Elapsed time: {response_data['elapsed_time']:.2f} seconds\n"
                )
                st.markdown("Sources:")
                for i, source in enumerate(response_data["sources"], 1):
                    st.markdown(f"\n-------Source {i}:")
                    st.markdown(source["source"])
                    st.markdown(f"Content: {source['content'][:200]}...")
            else:
                st.error("Error processing your request")
