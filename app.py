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
# File uploader for video files
video_file = st.file_uploader(
    "Upload a video to check if it's genuine or deepfake", type=["mp4", "mov", "avi"]
)

if video_file is not None:
    st.video(video_file, muted=True)

    with st.spinner('Analyzing Video Clip 📽️'):
        video_url = bucket.upload_video(video_file)
        print(video_url)
        result = endpoint.predict(instances=[{"video_url": video_url}])
        print(result)

        score = result["score"]
        # Display the result
        st.snow()
        st.metric(label="Authenticity Score", value=f"{round(score*100)}%")

        # Interpretation based on score
        if score > 0.6:
            st.success("This video is most likely REAL!")
        elif score < 0.4:
            st.error("This video is suspected to be DEEPFAKE!")
        else:
            st.warning("We are not sure about the video's authenticity.")
