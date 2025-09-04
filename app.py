# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:38:01 2025

Python code to create Streamlit app for Somerville Open Datasets

@author: Kevin Monk
"""

import streamlit as st

def app():
    """
    All code for the app goes here

    """
    st.title('Test Title to Get a Hang of Things')
    st.sidebar.markdown('**This is a sidebar text:**')
    
    st.markdown(""" This is a Streamlit app that will show the basics of how to
                use the website. These will eventually relate to data exploration
                provided by Somerville, MA.""")
    
    
if __name__ == '__main__':
    app()

