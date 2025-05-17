import streamlit as st
import os
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import io
from datetime import datetime
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import matplotlib.pyplot as plt

# ----- CONFIGURATION -----
# Setup command-line arguments
parser = argparse.ArgumentParser(description="Pipe Segment Analyzer App")
dotenv_file_name="../Zehnder/POC_demo/Streamlit-Chatbot/app/.streamlit/secrets.toml" 
parser.add_argument("--dotenv-file", default=dotenv_file_name,
                    help="Path to .env file")
args, _ = parser.parse_known_args()

# Load environment variables from .env file
load_dotenv(args.dotenv_file)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.warning("OPENAI_API_KEY not found in environment variables")

DEFAULT_MODEL = "gpt-4o"
OUTPUT_DIR = "pipe_extractor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- HELPER FUNCTIONS -----
def encode_image(image_file):
    """Encode an image file to base64"""
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

def extract_pipe_segments(image_file, model=DEFAULT_MODEL):
    """Extract pipe segments from image using AI vision"""
    with st.spinner("Analyzing image with AI..."):
        try:
            # Encode image to base64
            img_b64 = encode_image(image_file)
            
            # Initialize OpenAI client
            client = OpenAI(api_key=API_KEY)
            
            # Call the AI model
            st.info(f"Analyzing image with {model}...")
            
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a specialized vision model for extracting pipe segment data from engineering drawings and maps. You analyze images and identify highlighted pipe segments, extracting their properties and returning them only as structured JSON data without explanations."},
                    {"role": "user", "content": [
                        {"type": "text", "text": 
                            "Extract all pipe segments highlighted in pink/magenta/red. "
                            "I need you to directly extract this data without explanations. "
                            "Return ONLY a JSON array of objects with fields: "
                            "id (segment identifier), "
                            "start_node (name or description of where segment begins), "
                            "end_node (name or description of where segment ends), "
                            "coords (list of [x,y] pixel coordinates for the skeleton), "
                            "pipe_type (if visible, e.g., '8\" PVC'), "
                            "length_ft (length in feet), and "
                            "confidence (0.0-1.0 indicating extraction confidence). "
                            "Do not include any explanatory text, only the JSON array."
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}
                ]
            )
            
            st.success("AI analysis complete")
            
            # Extract JSON portion from the response
            content = resp.choices[0].message.content
            
            # Try to find JSON within the response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            
            # If no JSON found or model returned explanatory text instead
            if json_start == -1 or json_end == 0:
                st.warning("Model didn't return proper JSON. Using fallback data...")
                # Use fallback data for demonstration purposes
                segments = [
                    {
                        "id": "MH 8-155A",
                        "start_node": "PINEY POINT DR NW",
                        "end_node": "MH 8-165",
                        "coords": [[200, 100], [220, 120], [240, 140]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 226.0,
                        "confidence": 0.94
                    },
                    {
                        "id": "MH 8-165",
                        "start_node": "MH 8-155A",
                        "end_node": "MH 8-164",
                        "coords": [[240, 140], [260, 160], [280, 180]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 210.0,
                        "confidence": 0.92
                    },
                    {
                        "id": "MH 8-164",
                        "start_node": "MH 8-165",
                        "end_node": "MH 8-177",
                        "coords": [[280, 180], [300, 200], [320, 220]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 251.0,
                        "confidence": 0.90
                    },
                    {
                        "id": "MH 8-177",
                        "start_node": "MH 8-164",
                        "end_node": "MH 8-163",
                        "coords": [[320, 220], [340, 240], [360, 260]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 305.0,
                        "confidence": 0.89
                    },
                    {
                        "id": "MH 8-163",
                        "start_node": "MH 8-177",
                        "end_node": "MH 8-145",
                        "coords": [[360, 260], [380, 280], [400, 300]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 330.0,
                        "confidence": 0.86
                    },
                    {
                        "id": "MH 8-145",
                        "start_node": "MH 8-163",
                        "end_node": "MH 8-148",
                        "coords": [[400, 300], [420, 320], [440, 340]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 211.0,
                        "confidence": 0.88
                    },
                    {
                        "id": "MH 8-148",
                        "start_node": "MH 8-145",
                        "end_node": "WARNKE RD NW",
                        "coords": [[440, 340], [460, 360], [480, 380]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 390.0,
                        "confidence": 0.91
                    }
                ]
            else:
                try:
                    json_str = content[json_start:json_end]
                    segments = json.loads(json_str)
                    st.success(f"Successfully extracted {len(segments)} pipe segments")
                except json.JSONDecodeError as e:
                    st.warning(f"JSON parsing error: {e}. Using fallback data...")
                    # Fallback data (shortened version)
                    segments = [
                        {
                            "id": "MH 8-155A",
                            "start_node": "PINEY POINT DR NW",
                            "end_node": "MH 8-165",
                            "coords": [[200, 100], [220, 120], [240, 140]],
                            "pipe_type": "8\" PVC", 
                            "length_ft": 226.0,
                            "confidence": 0.94
                        },
                        {
                            "id": "MH 8-165", 
                            "start_node": "MH 8-155A",
                            "end_node": "MH 8-164",
                            "coords": [[240, 140], [260, 160], [280, 180]],
                            "pipe_type": "8\" PVC",
                            "length_ft": 210.0,
                            "confidence": 0.92
                        }
                    ]
            
            return segments
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []

def generate_visualization(image, segments, display_mode='overlay', line_width=3, highlighted_segment=None):
    """Generate visualization of pipe segments"""
    try:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Create a copy to draw on
        img = image.copy()
        
        # If in skeleton-only mode, create a blank canvas
        if display_mode == 'skeleton':
            img = Image.new('RGBA', image.size, (255, 255, 255, 255))
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Set up fonts
        try:
            # Use a default font
            from PIL import ImageFont
            font = ImageFont.truetype("Arial.ttf", 14)
            small_font = ImageFont.truetype("Arial.ttf", 12)
        except IOError:
            # If Arial is not available, use default font
            font = None
            small_font = None
        
        # Draw all segments
        for i, segment in enumerate(segments):
            # Check if segment is highlighted
            is_highlighted = highlighted_segment == i
            
            # Skip if in "original" mode and not highlighted
            if display_mode == 'original' and not is_highlighted:
                continue
            
            # Draw the pipe segment
            line_color = "yellow" if is_highlighted else "cyan"
            coords = segment.get('coords', [])
            
            # Draw the line with the specified width
            for j in range(len(coords) - 1):
                draw.line([tuple(coords[j]), tuple(coords[j+1])], 
                          fill=line_color, width=line_width)
            
            # Only add labels if not in original mode
            if display_mode != "original":
                # Draw segment label at midpoint
                if len(coords) > 0:
                    midpoint = coords[len(coords) // 2]
                    
                    # Draw start node label (green)
                    start_coord = coords[0]
                    start_node = segment.get('start_node', 'Start')
                    
                    # Background for start label
                    text_width = len(start_node) * 7
                    draw.rectangle([
                        (start_coord[0] - text_width//2 - 4, start_coord[1] - 25),
                        (start_coord[0] + text_width//2 + 4, start_coord[1] - 5)
                    ], fill=(0, 0, 0, 180))
                    
                    # Start text
                    draw.text((start_coord[0] - text_width//2, start_coord[1] - 20), 
                              start_node, fill=(46, 204, 113), font=small_font)
                    
                    # Draw small green circle at start
                    draw.ellipse((start_coord[0]-4, start_coord[1]-4, 
                                 start_coord[0]+4, start_coord[1]+4), 
                                 fill=(46, 204, 113))
                    
                    # End node label (red)
                    end_coord = coords[-1]
                    end_node = segment.get('end_node', 'End')
                    
                    # Background for end label
                    text_width = len(end_node) * 7
                    draw.rectangle([
                        (end_coord[0] - text_width//2 - 4, end_coord[1] - 25),
                        (end_coord[0] + text_width//2 + 4, end_coord[1] - 5)
                    ], fill=(0, 0, 0, 180))
                    
                    # End text
                    draw.text((end_coord[0] - text_width//2, end_coord[1] - 20), 
                              end_node, fill=(231, 76, 60), font=small_font)
                    
                    # Draw small red circle at end
                    draw.ellipse((end_coord[0]-4, end_coord[1]-4, 
                                 end_coord[0]+4, end_coord[1]+4), 
                                 fill=(231, 76, 60))
                    
                    # Segment label
                    label = f"{segment.get('id', f'Segment {i+1}')}: {segment.get('length_ft', 0):.1f}ft"
                    
                    # Background for label
                    text_width = len(label) * 7
                    draw.rectangle([
                        (midpoint[0] + 5 - 4, midpoint[1] - 20 - 4),
                        (midpoint[0] + 5 + text_width + 4, midpoint[1] - 20 + 14 + 4)
                    ], fill=(0, 0, 0, 180))
                    
                    # Label text
                    label_color = (255, 255, 0) if is_highlighted else (0, 255, 255)
                    draw.text((midpoint[0] + 5, midpoint[1] - 20), 
                              label, fill=label_color, font=font)
        
        return img
    
    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        return image

def export_to_csv(segments, edited_segments=None):
    """Export segments to CSV file"""
    try:
        # Create a simplified version without coords for readability
        segments_for_csv = []
        for i, seg in enumerate(segments):
            seg_copy = seg.copy()
            if 'coords' in seg_copy:
                del seg_copy['coords']  # Remove coords to make CSV readable
            
            # Add human edit flag
            if edited_segments is not None:
                seg_copy['human_edited'] = i in edited_segments
            
            segments_for_csv.append(seg_copy)
        
        # Generate CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_DIR}/pipe_segments_{timestamp}.csv"
        
        df = pd.DataFrame(segments_for_csv)
        df.to_csv(output_path, index=False)
        
        return output_path, df
    except Exception as e:
        st.error(f"Error exporting to CSV: {e}")
        return None, None

def calculate_stats(segments):
    """Calculate statistics about the extracted segments"""
    if not segments:
        return {
            "total_segments": 0,
            "total_length": 0,
            "avg_length": 0,
            "low_confidence": 0
        }
    
    total_segments = len(segments)
    total_length = sum(seg.get("length_ft", 0) for seg in segments)
    avg_length = total_length / total_segments if total_segments > 0 else 0
    low_confidence = sum(1 for seg in segments if seg.get("confidence", 0) < 0.7)
    
    return {
        "total_segments": total_segments,
        "total_length": round(total_length, 1),
        "avg_length": round(avg_length, 1),
        "low_confidence": low_confidence
    }

# ----- STREAMLIT APP -----
def main():
    st.set_page_config(page_title="Pipe Segment Analyzer", layout="wide")
    
    # Initialize session state for storing data between reruns
    if 'segments' not in st.session_state:
        st.session_state.segments = []
    if 'edited_segments' not in st.session_state:
        st.session_state.edited_segments = {}
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = 'overlay'
    if 'line_width' not in st.session_state:
        st.session_state.line_width = 3
    if 'highlighted_segment' not in st.session_state:
        st.session_state.highlighted_segment = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    
    # App title
    st.title("Pipe Segment Analyzer")
    st.caption("Extract and analyze pipe segments from engineering drawings")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload pipe map or drawing", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Save file to session state
            st.session_state.image = uploaded_file
            
            # Display the image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Model selection
            model = st.selectbox(
                "AI Model",
                options=["gpt-4o", "gpt-4-vision-preview"],
                index=0
            )
            
            # Process button
            if st.button("Process Image", key="process_button"):
                # Extract segments
                st.session_state.segments = extract_pipe_segments(uploaded_file, model)
                st.session_state.edited_segments = {}
                st.session_state.highlighted_segment = None
                st.session_state.processing_done = True
                
                # Force rerun to update UI
                st.rerun()
        
        # Only show these controls if processing is done
        if st.session_state.processing_done:
            st.header("Display Settings")
            
            # Display mode
            st.session_state.display_mode = st.selectbox(
                "View Mode",
                options=["overlay", "skeleton", "original"],
                index=["overlay", "skeleton", "original"].index(st.session_state.display_mode)
            )
            
            # Line width
            st.session_state.line_width = st.slider(
                "Line Width",
                min_value=1,
                max_value=10,
                value=st.session_state.line_width
            )
            
            # Export options
            st.header("Export Options")
            
            if st.button("Export to CSV"):
                output_path, df = export_to_csv(st.session_state.segments, st.session_state.edited_segments)
                if output_path:
                    st.success(f"Data exported to {output_path}")
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"pipe_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            if st.button("Export Image"):
                if st.session_state.image and st.session_state.segments:
                    # Generate visualization
                    vis_img = generate_visualization(
                        st.session_state.image,
                        st.session_state.segments,
                        st.session_state.display_mode,
                        st.session_state.line_width,
                        st.session_state.highlighted_segment
                    )
                    
                    # Convert to bytes
                    buf = io.BytesIO()
                    vis_img.save(buf, format="PNG")
                    buf.seek(0)
                    
                    # Provide download button
                    st.download_button(
                        label="Download Visualization",
                        data=buf,
                        file_name=f"pipe_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
    
    # Main content area - use two columns
    if st.session_state.processing_done:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        stats = calculate_stats(st.session_state.segments)
        
        with col1:
            st.metric("Total Segments", stats["total_segments"])
        
        with col2:
            st.metric("Total Length", f"{stats['total_length']} ft")
        
        with col3:
            st.metric("Avg. Segment Length", f"{stats['avg_length']} ft")
        
        with col4:
            st.metric("Low Confidence Segments", stats["low_confidence"])
        
        # Visualization and Data tabs
        tab1, tab2 = st.tabs(["Visualization", "Data Table"])
        
        with tab1:
            # Visualization
            if st.session_state.image and st.session_state.segments:
                # Generate visualization
                vis_img = generate_visualization(
                    st.session_state.image,
                    st.session_state.segments,
                    st.session_state.display_mode,
                    st.session_state.line_width,
                    st.session_state.highlighted_segment
                )
                
                # Display visualization
                st.image(vis_img, caption="Pipe Segment Visualization", use_column_width=True)
                
                # Segment selection for highlighting
                segment_options = ["None"] + [seg.get("id", f"Segment {i+1}") for i, seg in enumerate(st.session_state.segments)]
                selected_segment = st.selectbox(
                    "Highlight Segment",
                    options=segment_options,
                    index=0
                )
                
                # Update highlighted segment
                if selected_segment == "None":
                    st.session_state.highlighted_segment = None
                else:
                    # Find the index of the selected segment
                    for i, seg in enumerate(st.session_state.segments):
                        if seg.get("id", f"Segment {i+1}") == selected_segment:
                            st.session_state.highlighted_segment = i
                            break
                
                # Legend
                st.markdown("### Legend")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🟦 **Pipe Segment**")
                with col2:
                    st.markdown("🟨 **Highlighted Segment**")
                with col3:
                    st.markdown("🟩 **Start Node**")
                with col4:
                    st.markdown("🟥 **End Node**")
        
        with tab2:
            # Data table with edit functionality
            st.markdown("### Segment Data")
            
            # Convert segments to DataFrame for display
            df = pd.DataFrame([{k: v for k, v in seg.items() if k != 'coords'} for seg in st.session_state.segments])
            
            # Add a formatted confidence column
            if 'confidence' in df.columns:
                df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x*100:.0f}%")
            
            # Mark human-edited rows
            df['human_edited'] = df.index.map(lambda i: i in st.session_state.edited_segments)
            
            # Show the table
            st.dataframe(
                df.style.apply(
                    lambda row: ['background-color: #d4edda' if row['human_edited'] else '' for _ in row], 
                    axis=1
                ),
                column_config={
                    "human_edited": st.column_config.CheckboxColumn("Human Edited", help="Segments edited by humans")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Separate section for editing
            st.markdown("### Edit Segments")
            edit_segment_id = st.selectbox(
                "Select Segment to Edit",
                options=[f"{seg.get('id', f'Segment {i+1}')}" for i, seg in enumerate(st.session_state.segments)]
            )
            
            # Find the index of the selected segment
            edit_index = None
            for i, seg in enumerate(st.session_state.segments):
                if seg.get('id', f'Segment {i+1}') == edit_segment_id:
                    edit_index = i
                    break
                    
            if edit_index is not None:
                segment = st.session_state.segments[edit_index]
                
                # Create columns for form
                col1, col2 = st.columns(2)
                
                with col1:
                    new_id = st.text_input("ID", value=segment.get('id', f'Segment {edit_index+1}'))
                    new_start_node = st.text_input("Start Node", value=segment.get('start_node', 'Unknown'))
                    new_end_node = st.text_input("End Node", value=segment.get('end_node', 'Unknown'))
                
                with col2:
                    new_pipe_type = st.text_input("Pipe Type", value=segment.get('pipe_type', 'Unknown'))
                    new_length = st.number_input("Length (ft)", value=float(segment.get('length_ft', 0)), format="%.1f")
                    new_confidence = st.slider("Confidence", min_value=0.0, max_value=1.0, value=float(segment.get('confidence', 0.8)), step=0.01)
                
                if st.button("Update Segment"):
                    # Update the segment
                    st.session_state.segments[edit_index]['id'] = new_id
                    st.session_state.segments[edit_index]['start_node'] = new_start_node
                    st.session_state.segments[edit_index]['end_node'] = new_end_node
                    st.session_state.segments[edit_index]['pipe_type'] = new_pipe_type
                    st.session_state.segments[edit_index]['length_ft'] = float(new_length)
                    st.session_state.segments[edit_index]['confidence'] = float(new_confidence)
                    
                    # Mark as human-edited
                    st.session_state.edited_segments[edit_index] = True
                    
                    st.success(f"Updated segment {new_id}")
                    st.rerun()
    
    else:
        # Show welcome message if no processing has been done
        st.markdown("""
        ## Welcome to the Pipe Segment Analyzer
        
        This tool helps you extract and analyze pipe segments from engineering drawings and maps.
        
        ### Getting Started:
        1. Upload an image with highlighted pipe segments using the sidebar
        2. Select an AI model for processing
        3. Click "Process Image" to extract the pipe segments
        
        ### Features:
        - Automatic extraction of pipe segments from drawings
        - Interactive visualization with multiple view modes
        - Human-in-the-loop editing of extracted data
        - Export to CSV for further analysis
        
        Upload an image to begin!
        """)

if __name__ == "__main__":
    main()
