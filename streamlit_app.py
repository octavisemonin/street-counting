import streamlit as st
from streamlit_gsheets import GSheetsConnection

import cv2
import pickle
import numpy as np
import pandas as pd
import time

from datetime import datetime
from google.cloud import videointelligence
from google.oauth2 import service_account

from ffmpy import FFmpeg

colors = {
    'car': (0,0,255),
    'person': (0,255,0), 
    'bicycle': (255,0,0),
    # 'toy vehicle': (250,128,114),
    # 'lighting': (255,255,0), 
    # 'tire': (0,0,0), 
    # 'luggage & bags': (150,75,0), 
    # 'building': (128,128,128), 
    # 'shoe': (128,128,128), 
    # 'mirror': (127, 0, 0),
    # 'wheel': (0, 127, 0)
}

def analyze(result):
    """Analyze the objects and tracks from Google"""

    object_annotations = result.annotation_results[0].object_annotations
    objects = []
    tracks = []

    for n,obj in enumerate(object_annotations):
        
        objects.append(obj.entity.description)

        for f in obj.frames:

            track = [
                n,
                obj.entity.description,
                f.time_offset.total_seconds(),
                f.normalized_bounding_box.left,
                f.normalized_bounding_box.right,
                f.normalized_bounding_box.top,
                f.normalized_bounding_box.bottom,
            ]
            tracks.append(track) 

    objects = pd.Series(objects, name='type')
    tracks = pd.DataFrame(tracks, columns=['n','description','time','left','right','top','bottom'])

    tracks['on_left'] = tracks['right'] < 0.5
    tracks['on_right'] = tracks['left'] > 0.5
    first_left = tracks.loc[tracks['on_left']].groupby('n').first()['time']
    first_left = first_left.rename('first_left')
    first_right = tracks.loc[tracks['on_right']].groupby('n').first()['time']
    first_right = first_right.rename('first_right')

    objects = pd.concat([objects, first_left, first_right], axis=1)

    objects['crossing'] = None
    objects.loc[objects['first_left'] < objects['first_right'], 'crossing'] = 'right'
    objects.loc[objects['first_left'] > objects['first_right'], 'crossing'] = 'left'

    n_crossing_right = sum(objects['crossing']=='right')
    n_crossing_left = sum(objects['crossing']=='left')
    counts = objects.groupby(['type','crossing']).size().unstack()

    # Log run history in Google Sheets
    df = sheets_client.read(ttl=0)
    df.loc[len(df)] = [datetime.now(), video.name, video.size, 
                         tracks['time'].max(), process_time,
                         len(objects), len(tracks)]
    sheets_client.update(worksheet=0, data=df)

    # Return
    r = {
        'objects':objects, 'tracks':tracks,
        'n_crossing_right':n_crossing_right,
        'n_crossing_left':n_crossing_left,
        'counts':counts,
    }

    return r

# Session state initialization
if 'annotation_result' not in st.session_state:
    st.session_state['annotation_result'] = None

# Create a GCP service account.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
sheets_client = st.connection("gsheets", type=GSheetsConnection)

# Show title and description.
st.title("🚸 Street Counting")
st.write(
    "Upload a video below and the robots will count stuff! "
)

# Let the user upload a file via `st.file_uploader`.
video = st.file_uploader(
    "Upload a video (.mp4, .mov, or .avi). We will not keep it.", type=("mp4", "mov", "avi")
)

if video and st.session_state['annotation_result'] is None:

    with st.spinner("The robots are annotating (hang in there, this takes awhile)."):
        start_time = time.time()

        # Send to Google
        features = [videointelligence.Feature.OBJECT_TRACKING]
        operation = video_client.annotate_video(
            request={"features": features, 
                        "input_content": video.getvalue()}
        )
        result = operation.result(timeout=500)

        # Use this for debugging instead of using API calls:
        # temp = 'result-4a2e7a146c8f7bc1dd61acac940c5b04.pickle'
        # temp = 'result-96b3339212e7b4771f6c002cc97cff78.pickle'
        # with open(temp, 'rb') as handle:
        #     result = pickle.load(handle)

        st.session_state['annotation_result'] = result
        process_time = time.time() - start_time
        st.write("Finished processing.")

    # Analyze
    r = analyze(result)
    objects = r['objects']
    tracks = r['tracks']
    st.write(f"{len(objects)} objects found:")
    st.dataframe(objects)
    st.write(f"{r['n_crossing_right']} crossing right, {r['n_crossing_left']} crossing left:")
    st.dataframe(r['counts'])

    # User can download their data
    st.download_button(
        "Download objects",
        objects.to_csv().encode('utf-8'),
        "objects.csv",
        "text/csv",
    )

    st.download_button(
        "Download tracks",
        tracks.to_csv().encode('utf-8'),
        "tracks.csv",
        "text/csv",
    )

    with open(video.name, mode='wb') as f:
        f.write(video.read()) # save to read by cv2

    cap = cv2.VideoCapture(video.name)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = []
    frame_number = 0

    width = int(cap.get(3))
    height = int(cap.get(4))

    # Heatmap
    heatmap = np.zeros((int(height), int(width), 3))

    # Draw annotations
    output_path = f'{video.name}.annotated.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width,  height))

    progress_text = "Drawing annotations. Please wait."
    my_bar = st.progress(0, text=progress_text)
    frame_img = st.empty()
    heat_img = st.empty()

    # let's consolidate this later and use tracks below
    object_annotations = result.annotation_results[0].object_annotations

    while(cap.isOpened()):
        ret, img = cap.read()

        if ret:
            for n,obj in enumerate(object_annotations):
                description = obj.entity.description
                entity_id = obj.entity.entity_id
                
                for f in obj.frames:
                    obj_frame = round(f.time_offset.total_seconds() * fps)
                    
                    if abs(frame_number-obj_frame) < 2:
                        left = int(f.normalized_bounding_box.left * width)
                        right = int(f.normalized_bounding_box.right * width)
                        top = int(f.normalized_bounding_box.top * height)
                        bottom = int(f.normalized_bounding_box.bottom * height)
        
                        if description in colors.keys():
                            color = colors[description]
                            heatmap[top:bottom,left:right] += color
                        else: 
                            color = (255,255,255)
                            
                        cv2.rectangle(img, (left, top), (right, bottom), 
                                    color, 2)
                        cv2.putText(img, f'Obj #{n}, {description}', (left, bottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    color, 2)

            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_number += 1

            out.write(img)

            my_bar.progress(frame_number/length, text=progress_text)
            if frame_number % 10 == 0: 
                frame_img.image(img)
                heatmap_norm = np.log(heatmap + 1)
                heatmap_norm = heatmap_norm/heatmap_norm.max(axis=0).max(axis=0)
                heat_img.image(heatmap_norm)

            # cv2.imshow("annotations", img)
            # if cv2.waitKey(1) == 27:
            #     break
        else:
            cap.release()
            out.release()
            my_bar.empty()
            break

    with open(output_path, 'rb') as f:
        st.download_button('Download annotated video', f, file_name=output_path)  # Defaults to 'application/octet-stream'

    # This doesn't seem to work on streamlit cloud...
    # https://discuss.streamlit.io/t/processing-video-with-opencv-and-write-it-to-disk-to-display-in-st-video/28891/2
    # ff = FFmpeg(
    #     inputs={output_path: '-y -i'},
    #     outputs={output_path: '-c:v libx264'}
    # )
    # ff.run()
    # st.video(output_path)
    # st.image(img)

elif video is None:
    st.session_state['annotation_result'] = None

st.write("Run history:")
df = sheets_client.read(ttl=0)
st.dataframe(df)
st.write('Questions? Email octavi@gmail.com')