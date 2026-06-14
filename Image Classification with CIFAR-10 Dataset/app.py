import streamlit as st
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load CIFAR-10 data
(x_train, y_train), (_, _) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Viewer")
st.write("Select a class and see images")

# User choice dropdown
choice = st.selectbox("Choose class:", class_names)

# Get index of chosen class
class_idx = class_names.index(choice)

# Find 9 images of that class
indices = np.where(y_train.flatten() == class_idx)[0][:9]

st.subheader(f"Showing 9 '{choice}' images")

# Display images in 3x3 grid
cols = st.columns(3)
for i, idx in enumerate(indices):
    with cols[i % 3]:
        st.image(x_train[idx], caption=class_names[y_train[idx][0]], use_column_width=True)