import streamlit as st
from PIL import Image

def display_query_image(query_image):
    st.image(query_image, caption="Query Image", use_container_width=True)

def display_results(results):
    st.markdown("### Retrieved Images")
    
    max_cols = 2
    num_imgs = len(results)
    cols = st.columns(min(num_imgs, max_cols))
    
    for i, (img_path, dist) in enumerate(results):
        col = cols[i % max_cols]  
        try:
            img = Image.open(img_path)
            col.image(img, caption=f"Distance: {dist:.4f}", width=350)  
        except Exception as e:
            col.write("Error loading image")
            col.text(img_path)