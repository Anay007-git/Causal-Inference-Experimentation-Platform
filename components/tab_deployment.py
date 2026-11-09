"""
UI Component: Deployment Hub (Vercel, Streamlit Cloud, Docker)
"""

import streamlit as st
import sys
import platform


def render_deployment_tab():
    st.markdown("### 🚀 Deployment & Production Guide")
    st.markdown(
        """
        This platform features a **hybrid production architecture** designed for zero-friction deployment 
        across multiple modern hosting environments:
        - **Vercel**: Deploy serverless Python causal engine via Vercel Serverless Functions (`api/index.py`) and modern web UI.
        - **Streamlit Community Cloud**: Free 1-click cloud hosting directly from your GitHub repository.
        - **Docker / GCP Cloud Run**: Containerized deployment for enterprise VPCs and Kubernetes.
        """
    )

    t1, t2, t3 = st.tabs(["▲ Deploy to Vercel", "☁️ Streamlit Cloud", "🐳 Docker & GCP Cloud Run"])

    with t1:
        st.markdown("#### Deploying on Vercel")
        st.markdown(
            """
            This repository comes pre-configured with `vercel.json` and a serverless causal engine in `api/index.py`.
            
            **Method A: Deploy via Vercel CLI**
            ```bash
            # 1. Install Vercel CLI (if not already installed)
            npm install -g vercel

            # 2. Login to your Vercel account
            vercel login

            # 3. Deploy from the project root
            vercel --prod
            ```

            **Method B: 1-Click Git Integration via Vercel Dashboard**
            1. Push this repository to GitHub or GitLab.
            2. Open [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New Project**.
            3. Select your repository. Vercel will automatically detect `vercel.json` and deploy the serverless Python functions in `api/index.py`.
            4. Your causal inference application will be live instantly on `https://your-project.vercel.app`!
            """
        )

    with t2:
        st.markdown("#### Deploying on Streamlit Community Cloud (Free)")
        st.markdown(
            """
            Streamlit Community Cloud offers native, zero-setup hosting for this full interactive dashboard.

            1. Push this code to a public or private GitHub repository.
            2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
            3. Select your repository, branch (`main`), and set the main file path to:
               ```text
               app.py
               ```
            4. Click **Deploy!** Streamlit Cloud installs `requirements.txt` and boots the dashboard within 2 minutes.
            """
        )

    with t3:
        st.markdown("#### Enterprise Deployment with Docker / GCP Cloud Run")
        st.markdown(
            """
            For internal enterprise experimentation platforms running inside a VPC:
            
            ```bash
            # 1. Build the Docker image
            docker build -t causal-experimentation-platform:latest .

            # 2. Run locally or in a test VM
            docker run -p 8501:8501 causal-experimentation-platform:latest

            # 3. Deploy to Google Cloud Run
            gcloud builds submit --tag gcr.io/[PROJECT-ID]/causal-platform
            gcloud run deploy causal-platform --image gcr.io/[PROJECT-ID]/causal-platform --platform managed --port 8501
            ```
            """
        )

    # Runtime Environment Diagnostics
    st.markdown("#### ⚙️ Runtime Diagnostic Information")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Python Version", platform.python_version())
    with d2:
        st.metric("Platform OS", platform.system())
    with d3:
        st.metric("Streamlit Version", st.__version__)
    with d4:
        st.metric("Architecture", platform.machine())

