import streamlit as st
from storage import GCSUtil
import requests
from google.cloud import aiplatform
from google.oauth2 import service_account


bucket = GCSUtil(st.secrets['bucket_name'], st.secrets['google_service_account'])
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_service_account"]
)

aiplatform.init(project=st.secrets['project_id'], location=st.secrets['location'], credentials=credentials)
endpoint = aiplatform.Endpoint(
    f"projects/{st.secrets['project_number']}/locations/{st.secrets['location']}/endpoints/{st.secrets['endpoint_id']}"
)

st.title("Deepfake Video Classifier")
# Allow user to upload a file or select from pre-existing videos
video_option = st.selectbox(
    "Choose an option", ["Upload a video", "Select from sample videos"]
)

if video_option == "Upload a video":
    # File uploader for video files
    video_file = st.file_uploader(
        "Upload a video to check if it's genuine or deepfake",
        type=["mp4", "mov", "avi"],
    )

elif video_option == "Select from sample videos":
    # List of sample videos
    sample_videos = {
        "Mr Beast Speech (REAL)": "videos/mr beast real.mp4",
        "Mr Beast Meme (FAKE)": "videos/mr beast fake.mp4",
    }
    selected_video = st.selectbox("Select a sample video", list(sample_videos.keys()))
    video_file = sample_videos[selected_video]

# Check if a video file or selection is available
if video_file:
    if isinstance(video_file, str):
        video_file = open(video_file, "rb")
    st.video(video_file, muted=True)

    if st.button('Analyze'):
        video_file.seek(0)
        video_url = bucket.upload_video(video_file)

        with st.spinner("Analyzing the video 📽️"):
            result = endpoint.predict(instances=[{"video_url": video_url}])

        # Display the result
        st.snow()
        score = result.predictions[0]["score"]
        st.metric(label="Authenticity Score", value=f"{round(score*100)}%")

        # Interpretation based on score
        if score > 0.6:
            st.success("This video is most likely REAL!")
        elif score < 0.4:
            st.error("This video is suspected to be DEEPFAKE!")
        else:
            st.warning("We are not sure about the video's authenticity.")
